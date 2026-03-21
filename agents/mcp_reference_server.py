"""MCP Reference Data Server — Reference Implementation (§10.3).

This is the REFERENCE IMPLEMENTATION of an MCP reference data server.
Teams MUST implement their own servers following the same contract
(MCPLookupRequest → MCPLookupResponse) but can use this as a template.

The reference server provides:
- In-memory reference data store with code/description/details per entry
- Single-attribute lookups: GET /<CATEGORY> — list all items in a category
- Hierarchical lookups: GET /<CATEGORY>/<code>/<CHILD> — e.g. /CTRY/US/STATE
- Value resolution: POST /lookup — resolve raw values to canonical entries
- Example datasets (entities, currencies, jurisdictions, countries, states, cities)

Teams override this with their own reference data sources:
- Database lookups
- External API calls
- File-based reference data
- Domain-specific normalization logic

CONTRACT REQUIREMENTS (teams MUST adhere to):
1. Accept POST /lookup with JSON body matching MCPLookupRequest
2. Return JSON matching MCPLookupResponse (must include code, description, details)
3. Support GET /<CATEGORY> for single-attribute listing
4. Support GET /<CATEGORY>/<code>/<CHILD_CATEGORY> for hierarchical lookups (up to 2 levels)
5. Always return a response (even on no-match: matched=False)
6. Include confidence score (1.0 = exact match, <1.0 = fuzzy)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field as dc_field
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional, Tuple

from agents.contracts import MCPLookupRequest, MCPLookupResponse

logger = logging.getLogger(__name__)


# ── Rich Reference Entry ─────────────────────────────────────────────


@dataclass
class ReferenceEntry:
    """A single reference data item with code, description, details, and aliases.

    Every entry MUST have code + description.  Details is a free-form dict
    for domain-specific metadata (e.g. continent, population, ISO codes).
    """
    code: str                                        # Canonical identifier (e.g. "US", "USD")
    description: str                                 # Human-readable label
    details: Dict[str, Any] = dc_field(default_factory=dict)
    aliases: List[str] = dc_field(default_factory=list)


# ── Reference Data Store ─────────────────────────────────────────────


class ReferenceDataStore:
    """In-memory reference data store with hierarchical lookup support.

    Teams replace this with their own data sources (DB, API, file).

    Supports:
    - Single-attribute lookup:  list_category("CTRY") → all countries
    - Value resolution:         lookup("entity_name", "RBC") → canonical entry
    - Hierarchical lookup:      lookup_hierarchical("CTRY", "US", "STATE") → states in US
    """

    def __init__(self) -> None:
        # Category → {normalized_code → ReferenceEntry}
        self._entries: Dict[str, Dict[str, ReferenceEntry]] = {}
        # Category → {normalized_alias → normalized_code}
        self._aliases: Dict[str, Dict[str, str]] = {}
        # Hierarchical: parent_category → parent_code → child_category → [child_codes]
        self._hierarchy: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

    def add_entry(
        self,
        category: str,
        entry: ReferenceEntry,
    ) -> None:
        """Add a reference entry to a category."""
        if category not in self._entries:
            self._entries[category] = {}
            self._aliases[category] = {}

        key = entry.code.lower().strip()
        self._entries[category][key] = entry
        # Also index by description
        self._aliases[category][entry.description.lower().strip()] = key
        # Index aliases
        for alias in entry.aliases:
            self._aliases[category][alias.lower().strip()] = key

    def add_reference(
        self,
        category: str,
        canonical_value: str,
        aliases: Optional[List[str]] = None,
        code: Optional[str] = None,
        description: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a reference data entry (backwards-compatible with old API).

        If code/description not provided, derives them from canonical_value.
        """
        entry = ReferenceEntry(
            code=code or canonical_value,
            description=description or canonical_value,
            details=details or {},
            aliases=aliases or [],
        )
        self.add_entry(category, entry)

    def add_hierarchy(
        self,
        parent_category: str,
        parent_code: str,
        child_category: str,
        child_codes: List[str],
    ) -> None:
        """Register a hierarchical relationship.

        Example: add_hierarchy("CTRY", "US", "STATE", ["CA", "NY", "TX"])
        """
        if parent_category not in self._hierarchy:
            self._hierarchy[parent_category] = {}
        parent_key = parent_code.lower().strip()
        if parent_key not in self._hierarchy[parent_category]:
            self._hierarchy[parent_category][parent_key] = {}
        self._hierarchy[parent_category][parent_key][child_category.upper()] = [
            c.lower().strip() for c in child_codes
        ]

    def list_category(self, category: str) -> List[ReferenceEntry]:
        """List all entries in a category (single-attribute lookup)."""
        cat_entries = self._entries.get(category.lower(), {})
        if not cat_entries:
            # Try uppercase key
            cat_entries = self._entries.get(category.upper(), {})
        if not cat_entries:
            cat_entries = self._entries.get(category, {})
        return list(cat_entries.values())

    def lookup_hierarchical(
        self,
        parent_category: str,
        parent_code: str,
        child_category: str,
    ) -> List[ReferenceEntry]:
        """Hierarchical lookup: e.g. CTRY/US/STATE → states in US.

        Returns child entries from the child_category that belong to parent_code.
        """
        parent_cat = parent_category.upper()
        parent_key = parent_code.lower().strip()
        child_cat = child_category.upper()

        # Find child codes from hierarchy
        hier = self._hierarchy.get(parent_cat, {}).get(parent_key, {})
        child_codes = hier.get(child_cat, [])

        if not child_codes:
            return []

        # Resolve child codes to entries in the child category
        child_entries = self._entries.get(child_cat, {})
        results = []
        for code in child_codes:
            entry = child_entries.get(code)
            if entry:
                results.append(entry)
        return results

    def lookup(
        self, category: str, value: str
    ) -> Tuple[str, float, List[str], bool, ReferenceEntry | None]:
        """Look up a value in a category.

        Returns:
            (canonical_value, confidence, alternatives, matched, entry)
        """
        if category not in self._entries:
            return value, 0.0, [], False, None

        normalized = value.lower().strip()

        # Exact code match
        if normalized in self._entries[category]:
            entry = self._entries[category][normalized]
            return entry.code, 1.0, [], True, entry

        # Alias/description match
        if normalized in self._aliases[category]:
            code_key = self._aliases[category][normalized]
            entry = self._entries[category][code_key]
            return entry.code, 0.95, [], True, entry

        # Fuzzy match (prefix/substring)
        alternatives = []
        matched_entry = None
        for key, entry in self._entries[category].items():
            if normalized in key or key in normalized:
                alternatives.append(entry.code)
                if matched_entry is None:
                    matched_entry = entry
        for alias, code_key in self._aliases[category].items():
            if normalized in alias or alias in normalized:
                entry = self._entries[category][code_key]
                if entry.code not in alternatives:
                    alternatives.append(entry.code)
                    if matched_entry is None:
                        matched_entry = entry

        if alternatives:
            return alternatives[0], 0.7, alternatives[1:5], True, matched_entry

        return value, 0.0, [], False, None

    def load_from_dict(self, data: Dict[str, Dict[str, List[str]]]) -> None:
        """Bulk load reference data (backwards-compatible).

        Format: {category: {canonical_value: [alias1, alias2, ...]}}
        """
        for category, entries in data.items():
            for canonical, aliases in entries.items():
                self.add_reference(category, canonical, aliases)

    def load_rich_entries(
        self,
        category: str,
        entries: List[ReferenceEntry],
    ) -> None:
        """Bulk load rich reference entries into a category."""
        for entry in entries:
            self.add_entry(category, entry)


# ── Default Reference Data (ships with the platform) ─────────────────

_DEFAULT_REFERENCE_DATA = {
    "entity_name": {
        "Royal Bank of Canada": ["RBC", "Royal Bank", "RBC Financial Group"],
        "Toronto-Dominion Bank": ["TD Bank", "TD", "Toronto Dominion"],
        "Canadian Imperial Bank of Commerce": ["CIBC", "CI Bank of Commerce"],
        "Bank of Montreal": ["BMO", "Bank of Montréal", "BMO Financial Group"],
        "Bank of Nova Scotia": ["Scotiabank", "BNS", "Scotia"],
        "JPMorgan Chase & Co.": ["JPMorgan", "JP Morgan", "Chase", "JPM"],
        "Goldman Sachs Group, Inc.": ["Goldman Sachs", "Goldman", "GS"],
        "Morgan Stanley": ["MS", "Morgan Stanley & Co."],
    },
    "currency_code": {
        "USD": ["US Dollar", "US Dollars", "United States Dollar", "$", "US$"],
        "CAD": ["Canadian Dollar", "Canadian Dollars", "C$", "CA$"],
        "EUR": ["Euro", "Euros", "€"],
        "GBP": ["British Pound", "Pound Sterling", "£", "Sterling"],
        "JPY": ["Japanese Yen", "Yen", "¥"],
        "CHF": ["Swiss Franc", "Swiss Francs", "SFr"],
    },
    "jurisdiction": {
        "United States": ["US", "USA", "U.S.", "U.S.A.", "United States of America"],
        "Canada": ["CA", "CAN"],
        "United Kingdom": ["UK", "U.K.", "Great Britain", "GB"],
        "European Union": ["EU", "E.U."],
        "Switzerland": ["CH", "Swiss Confederation"],
        "Japan": ["JP", "JPN"],
    },
    "document_type": {
        "10-K": ["Annual Report (10-K)", "Form 10-K", "10K"],
        "10-Q": ["Quarterly Report (10-Q)", "Form 10-Q", "10Q"],
        "Basel III Pillar 3": ["Pillar 3", "Basel Pillar 3", "P3 Disclosure"],
        "Annual Report": ["AR", "Annual Financial Report"],
        "Proxy Statement": ["DEF 14A", "Proxy", "Form DEF 14A"],
    },
}


# ── Rich Geographic Reference Data ──────────────────────────────────

_COUNTRIES: List[ReferenceEntry] = [
    ReferenceEntry(code="US", description="United States", details={"iso3": "USA", "continent": "North America", "currency": "USD"}, aliases=["USA", "U.S.", "U.S.A.", "United States of America", "America"]),
    ReferenceEntry(code="CA", description="Canada", details={"iso3": "CAN", "continent": "North America", "currency": "CAD"}, aliases=["CAN"]),
    ReferenceEntry(code="GB", description="United Kingdom", details={"iso3": "GBR", "continent": "Europe", "currency": "GBP"}, aliases=["UK", "U.K.", "Great Britain", "Britain"]),
    ReferenceEntry(code="DE", description="Germany", details={"iso3": "DEU", "continent": "Europe", "currency": "EUR"}, aliases=["Deutschland"]),
    ReferenceEntry(code="FR", description="France", details={"iso3": "FRA", "continent": "Europe", "currency": "EUR"}, aliases=[]),
    ReferenceEntry(code="JP", description="Japan", details={"iso3": "JPN", "continent": "Asia", "currency": "JPY"}, aliases=["JPN"]),
    ReferenceEntry(code="CH", description="Switzerland", details={"iso3": "CHE", "continent": "Europe", "currency": "CHF"}, aliases=["Swiss Confederation"]),
    ReferenceEntry(code="AU", description="Australia", details={"iso3": "AUS", "continent": "Oceania", "currency": "AUD"}, aliases=["AUS"]),
    ReferenceEntry(code="SG", description="Singapore", details={"iso3": "SGP", "continent": "Asia", "currency": "SGD"}, aliases=["SGP"]),
    ReferenceEntry(code="HK", description="Hong Kong", details={"iso3": "HKG", "continent": "Asia", "currency": "HKD"}, aliases=["HKG"]),
    ReferenceEntry(code="IN", description="India", details={"iso3": "IND", "continent": "Asia", "currency": "INR"}, aliases=["IND"]),
    ReferenceEntry(code="BR", description="Brazil", details={"iso3": "BRA", "continent": "South America", "currency": "BRL"}, aliases=["BRA"]),
    ReferenceEntry(code="MX", description="Mexico", details={"iso3": "MEX", "continent": "North America", "currency": "MXN"}, aliases=["MEX"]),
    ReferenceEntry(code="CN", description="China", details={"iso3": "CHN", "continent": "Asia", "currency": "CNY"}, aliases=["CHN", "People's Republic of China", "PRC"]),
    ReferenceEntry(code="KR", description="South Korea", details={"iso3": "KOR", "continent": "Asia", "currency": "KRW"}, aliases=["KOR", "Republic of Korea"]),
]

_STATES: List[ReferenceEntry] = [
    # US States
    ReferenceEntry(code="CA", description="California", details={"country": "US", "fips": "06"}, aliases=["Calif."]),
    ReferenceEntry(code="NY", description="New York", details={"country": "US", "fips": "36"}, aliases=["N.Y."]),
    ReferenceEntry(code="TX", description="Texas", details={"country": "US", "fips": "48"}, aliases=["Tex."]),
    ReferenceEntry(code="FL", description="Florida", details={"country": "US", "fips": "12"}, aliases=["Fla."]),
    ReferenceEntry(code="IL", description="Illinois", details={"country": "US", "fips": "17"}, aliases=["Ill."]),
    ReferenceEntry(code="PA", description="Pennsylvania", details={"country": "US", "fips": "42"}, aliases=["Penn.", "Pa."]),
    ReferenceEntry(code="OH", description="Ohio", details={"country": "US", "fips": "39"}, aliases=[]),
    ReferenceEntry(code="GA", description="Georgia", details={"country": "US", "fips": "13"}, aliases=["Ga."]),
    ReferenceEntry(code="NC", description="North Carolina", details={"country": "US", "fips": "37"}, aliases=["N.C."]),
    ReferenceEntry(code="MA", description="Massachusetts", details={"country": "US", "fips": "25"}, aliases=["Mass."]),
    ReferenceEntry(code="NJ", description="New Jersey", details={"country": "US", "fips": "34"}, aliases=["N.J."]),
    ReferenceEntry(code="CT", description="Connecticut", details={"country": "US", "fips": "09"}, aliases=["Conn."]),
    ReferenceEntry(code="DE", description="Delaware", details={"country": "US", "fips": "10"}, aliases=["Del."]),
    # Canadian Provinces
    ReferenceEntry(code="ON", description="Ontario", details={"country": "CA"}, aliases=["Ont."]),
    ReferenceEntry(code="QC", description="Quebec", details={"country": "CA"}, aliases=["Que.", "Québec"]),
    ReferenceEntry(code="BC", description="British Columbia", details={"country": "CA"}, aliases=["B.C."]),
    ReferenceEntry(code="AB", description="Alberta", details={"country": "CA"}, aliases=["Alta."]),
    # UK Countries/Regions
    ReferenceEntry(code="ENG", description="England", details={"country": "GB"}, aliases=[]),
    ReferenceEntry(code="SCT", description="Scotland", details={"country": "GB"}, aliases=[]),
    ReferenceEntry(code="WLS", description="Wales", details={"country": "GB"}, aliases=[]),
]

_CITIES: List[ReferenceEntry] = [
    # US Cities
    ReferenceEntry(code="NYC", description="New York City", details={"state": "NY", "country": "US"}, aliases=["New York", "Manhattan", "NY City"]),
    ReferenceEntry(code="LAX", description="Los Angeles", details={"state": "CA", "country": "US"}, aliases=["LA", "L.A."]),
    ReferenceEntry(code="CHI", description="Chicago", details={"state": "IL", "country": "US"}, aliases=[]),
    ReferenceEntry(code="HOU", description="Houston", details={"state": "TX", "country": "US"}, aliases=[]),
    ReferenceEntry(code="SFO", description="San Francisco", details={"state": "CA", "country": "US"}, aliases=["SF", "S.F."]),
    ReferenceEntry(code="BOS", description="Boston", details={"state": "MA", "country": "US"}, aliases=[]),
    ReferenceEntry(code="MIA", description="Miami", details={"state": "FL", "country": "US"}, aliases=[]),
    ReferenceEntry(code="PHI", description="Philadelphia", details={"state": "PA", "country": "US"}, aliases=["Philly"]),
    ReferenceEntry(code="ATL", description="Atlanta", details={"state": "GA", "country": "US"}, aliases=[]),
    ReferenceEntry(code="CLT", description="Charlotte", details={"state": "NC", "country": "US"}, aliases=[]),
    ReferenceEntry(code="WIL", description="Wilmington", details={"state": "DE", "country": "US"}, aliases=[]),
    # Canadian Cities
    ReferenceEntry(code="YTO", description="Toronto", details={"state": "ON", "country": "CA"}, aliases=["TO"]),
    ReferenceEntry(code="YMQ", description="Montreal", details={"state": "QC", "country": "CA"}, aliases=["Montréal", "MTL"]),
    ReferenceEntry(code="YVR", description="Vancouver", details={"state": "BC", "country": "CA"}, aliases=["Van"]),
    ReferenceEntry(code="YOW", description="Ottawa", details={"state": "ON", "country": "CA"}, aliases=[]),
    ReferenceEntry(code="YYC", description="Calgary", details={"state": "AB", "country": "CA"}, aliases=[]),
    # UK Cities
    ReferenceEntry(code="LON", description="London", details={"state": "ENG", "country": "GB"}, aliases=[]),
    ReferenceEntry(code="EDI", description="Edinburgh", details={"state": "SCT", "country": "GB"}, aliases=[]),
    # Other
    ReferenceEntry(code="TYO", description="Tokyo", details={"country": "JP"}, aliases=[]),
    ReferenceEntry(code="HKG", description="Hong Kong", details={"country": "HK"}, aliases=[]),
    ReferenceEntry(code="SGP", description="Singapore City", details={"country": "SG"}, aliases=["Singapore"]),
    ReferenceEntry(code="SYD", description="Sydney", details={"country": "AU"}, aliases=[]),
    ReferenceEntry(code="FRA", description="Frankfurt", details={"country": "DE"}, aliases=[]),
    ReferenceEntry(code="ZRH", description="Zurich", details={"state": "ZH", "country": "CH"}, aliases=["Zürich"]),
]

# Entity type reference (single-attribute example)
_ENTITY_TYPES: List[ReferenceEntry] = [
    ReferenceEntry(code="BANK", description="Bank / Financial Institution", details={"sector": "financial_services"}),
    ReferenceEntry(code="INSURANCE", description="Insurance Company", details={"sector": "financial_services"}),
    ReferenceEntry(code="ASSET_MGR", description="Asset Management Firm", details={"sector": "financial_services"}),
    ReferenceEntry(code="CORP", description="Non-Financial Corporation", details={"sector": "corporate"}),
    ReferenceEntry(code="GOVT", description="Government / Sovereign", details={"sector": "public"}),
    ReferenceEntry(code="REGULATOR", description="Regulatory Authority", details={"sector": "public"}),
    ReferenceEntry(code="SPV", description="Special Purpose Vehicle", details={"sector": "structured_finance"}),
    ReferenceEntry(code="FUND", description="Investment Fund", details={"sector": "financial_services"}),
]


# ── Hierarchy Builder ────────────────────────────────────────────────

def _build_hierarchy(store: ReferenceDataStore) -> None:
    """Build hierarchical relationships between categories.

    Supports up to 2 levels:
      /CTRY/<code>/STATE  — states/provinces in a country
      /CTRY/<code>/CITY   — cities in a country
      /STATE/<code>/CITY  — cities in a state
    """
    # CTRY → STATE: group states by country
    country_states: Dict[str, List[str]] = {}
    for entry in _STATES:
        ctry = entry.details.get("country", "")
        if ctry:
            country_states.setdefault(ctry, []).append(entry.code)
    for ctry_code, state_codes in country_states.items():
        store.add_hierarchy("CTRY", ctry_code, "STATE", state_codes)

    # CTRY → CITY: group cities by country
    country_cities: Dict[str, List[str]] = {}
    for entry in _CITIES:
        ctry = entry.details.get("country", "")
        if ctry:
            country_cities.setdefault(ctry, []).append(entry.code)
    for ctry_code, city_codes in country_cities.items():
        store.add_hierarchy("CTRY", ctry_code, "CITY", city_codes)

    # STATE → CITY: group cities by state
    state_cities: Dict[str, List[str]] = {}
    for entry in _CITIES:
        state = entry.details.get("state", "")
        if state:
            state_cities.setdefault(state, []).append(entry.code)
    for state_code, city_codes in state_cities.items():
        store.add_hierarchy("STATE", state_code, "CITY", city_codes)


# ── Singleton Store ───────────────────────────────────────────────────

_store: Optional[ReferenceDataStore] = None


def get_reference_store() -> ReferenceDataStore:
    """Get or create the singleton reference data store."""
    global _store
    if _store is None:
        _store = ReferenceDataStore()
        # Load backwards-compatible simple entries
        _store.load_from_dict(_DEFAULT_REFERENCE_DATA)
        # Load rich geographic entries
        _store.load_rich_entries("CTRY", _COUNTRIES)
        _store.load_rich_entries("STATE", _STATES)
        _store.load_rich_entries("CITY", _CITIES)
        _store.load_rich_entries("ENTITY_TYPES", _ENTITY_TYPES)
        # Build hierarchical relationships
        _build_hierarchy(_store)
    return _store


def _entry_to_dict(entry: ReferenceEntry) -> Dict[str, Any]:
    """Serialize a ReferenceEntry to a response dict."""
    return {
        "code": entry.code,
        "description": entry.description,
        "details": entry.details,
    }


# ── HTTP Request Handler (Reference Server) ──────────────────────────


class MCPRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for the MCP reference data server.

    Teams can subclass or replace this handler with their own implementation.

    Supported routes:
      POST /lookup                      — resolve a raw value to canonical entry
      GET  /<CATEGORY>                  — list all items (single-attribute)
      GET  /<CATEGORY>/<code>/<CHILD>   — hierarchical lookup (up to 2 levels)
    """

    store: ReferenceDataStore = None  # type: ignore[assignment]

    def do_GET(self) -> None:
        store = self.store or get_reference_store()
        path = self.path.rstrip("/")
        parts = [p for p in path.split("/") if p]

        if len(parts) == 1:
            # Single-attribute: GET /ENTITY_TYPES → list all entity types
            category = parts[0]
            entries = store.list_category(category)
            results = [_entry_to_dict(e) for e in entries]
            self._send_json({"category": category, "count": len(results), "entries": results})

        elif len(parts) == 3:
            # Hierarchical: GET /CTRY/US/STATE → states in US
            parent_category, parent_code, child_category = parts
            entries = store.lookup_hierarchical(parent_category, parent_code, child_category)
            results = [_entry_to_dict(e) for e in entries]
            self._send_json({
                "parent_category": parent_category.upper(),
                "parent_code": parent_code.upper(),
                "child_category": child_category.upper(),
                "count": len(results),
                "entries": results,
            })

        else:
            self.send_error(400, "Use GET /<CATEGORY> or GET /<CATEGORY>/<code>/<CHILD>")

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/lookup":
            self.send_error(404, "Not Found")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        lookup_key = data.get("lookup_key", "")
        lookup_value = data.get("lookup_value", "")
        context = data.get("context", {})

        store = self.store or get_reference_store()
        canonical, confidence, alternatives, matched, entry = store.lookup(lookup_key, lookup_value)

        response = {
            "lookup_key": lookup_key,
            "original_value": lookup_value,
            "canonical_value": canonical,
            "code": entry.code if entry else canonical,
            "description": entry.description if entry else canonical,
            "details": entry.details if entry else {},
            "confidence": confidence,
            "source": "reference_implementation",
            "alternatives": alternatives,
            "matched": matched,
        }

        self._send_json(response)

    def _send_json(self, data: Dict[str, Any]) -> None:
        """Send a JSON response."""
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def log_message(self, format: str, *args: Any) -> None:
        """Route HTTP logs through Python logging."""
        logger.debug(format, *args)


def create_mcp_server(host: str = "0.0.0.0", port: int = 8100) -> HTTPServer:
    """Create an MCP reference data HTTP server.

    Usage:
        server = create_mcp_server(port=8100)
        server.serve_forever()
    """
    store = get_reference_store()
    MCPRequestHandler.store = store
    server = HTTPServer((host, port), MCPRequestHandler)
    logger.info("MCP Reference Server listening on %s:%d", host, port)
    return server


# ── Direct Lookup (bypass HTTP for in-process use) ────────────────────


def direct_lookup(
    lookup_key: str,
    lookup_value: str,
    context: Optional[Dict[str, str]] = None,
) -> MCPLookupResponse:
    """Perform a reference data lookup without HTTP (in-process).

    This is useful when the Transformer Agent and reference data are
    co-located.  For distributed deployments, use the HTTP server.
    """
    store = get_reference_store()
    canonical, confidence, alternatives, matched, entry = store.lookup(lookup_key, lookup_value)
    return MCPLookupResponse(
        lookup_key=lookup_key,
        original_value=lookup_value,
        canonical_value=canonical,
        code=entry.code if entry else canonical,
        description=entry.description if entry else canonical,
        details=entry.details if entry else {},
        confidence=confidence,
        source="reference_implementation_direct",
        alternatives=alternatives,
        matched=matched,
    )


def direct_list(category: str) -> List[Dict[str, Any]]:
    """List all entries in a category without HTTP (in-process).

    Single-attribute lookup: direct_list("ENTITY_TYPES") → all entity types.
    """
    store = get_reference_store()
    return [_entry_to_dict(e) for e in store.list_category(category)]


def direct_hierarchical_lookup(
    parent_category: str,
    parent_code: str,
    child_category: str,
) -> List[Dict[str, Any]]:
    """Hierarchical lookup without HTTP (in-process).

    Example: direct_hierarchical_lookup("CTRY", "US", "STATE") → US states.
    """
    store = get_reference_store()
    entries = store.lookup_hierarchical(parent_category, parent_code, child_category)
    return [_entry_to_dict(e) for e in entries]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    server = create_mcp_server()
    print("MCP Reference Data Server running on port 8100")
    print("  POST /lookup                     — resolve raw value to canonical entry")
    print("  GET  /<CATEGORY>                 — list all items (e.g. /ENTITY_TYPES)")
    print("  GET  /<CATEGORY>/<code>/<CHILD>  — hierarchical (e.g. /CTRY/US/STATE)")
    server.serve_forever()
