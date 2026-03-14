"""MCP Reference Data Server — Reference Implementation (§10.3).

This is the REFERENCE IMPLEMENTATION of an MCP reference data server.
Teams MUST implement their own servers following the same contract
(MCPLookupRequest → MCPLookupResponse) but can use this as a template.

The reference server provides:
- In-memory reference data store
- REST endpoint at POST /lookup
- Standard contract compliance
- Example datasets (entity names, currency codes, jurisdictions)

Teams override this with their own reference data sources:
- Database lookups
- External API calls
- File-based reference data
- Domain-specific normalization logic

CONTRACT REQUIREMENTS (teams MUST adhere to):
1. Accept POST /lookup with JSON body matching MCPLookupRequest
2. Return JSON matching MCPLookupResponse
3. Always return a response (even on no-match: matched=False)
4. Include confidence score (1.0 = exact match, <1.0 = fuzzy)
"""

from __future__ import annotations

import json
import logging
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional

from agents.contracts import MCPLookupRequest, MCPLookupResponse

logger = logging.getLogger(__name__)


class ReferenceDataStore:
    """In-memory reference data store.

    Teams replace this with their own data sources (DB, API, file).
    The interface is simple: lookup(key, value) → (canonical, confidence, alternatives).
    """

    def __init__(self) -> None:
        # Category → {normalized_key → canonical_value}
        self._data: Dict[str, Dict[str, str]] = {}
        # Category → {alias → canonical_value}
        self._aliases: Dict[str, Dict[str, str]] = {}

    def add_reference(
        self,
        category: str,
        canonical_value: str,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Add a reference data entry with optional aliases."""
        if category not in self._data:
            self._data[category] = {}
            self._aliases[category] = {}

        key = canonical_value.lower().strip()
        self._data[category][key] = canonical_value
        if aliases:
            for alias in aliases:
                self._aliases[category][alias.lower().strip()] = canonical_value

    def lookup(
        self, category: str, value: str
    ) -> tuple:
        """Look up a value in a category.

        Returns:
            (canonical_value, confidence, alternatives, matched)
        """
        if category not in self._data:
            return value, 0.0, [], False

        normalized = value.lower().strip()

        # Exact match
        if normalized in self._data[category]:
            return self._data[category][normalized], 1.0, [], True

        # Alias match
        if normalized in self._aliases[category]:
            canonical = self._aliases[category][normalized]
            return canonical, 0.95, [], True

        # Fuzzy match (simple prefix/substring)
        alternatives = []
        for key, canonical in self._data[category].items():
            if normalized in key or key in normalized:
                alternatives.append(canonical)
        for alias, canonical in self._aliases[category].items():
            if normalized in alias or alias in normalized:
                if canonical not in alternatives:
                    alternatives.append(canonical)

        if alternatives:
            return alternatives[0], 0.7, alternatives[1:5], True

        return value, 0.0, [], False

    def load_from_dict(self, data: Dict[str, Dict[str, List[str]]]) -> None:
        """Bulk load reference data.

        Format: {category: {canonical_value: [alias1, alias2, ...]}}
        """
        for category, entries in data.items():
            for canonical, aliases in entries.items():
                self.add_reference(category, canonical, aliases)


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


# ── Singleton Store ───────────────────────────────────────────────────

_store: Optional[ReferenceDataStore] = None


def get_reference_store() -> ReferenceDataStore:
    """Get or create the singleton reference data store."""
    global _store
    if _store is None:
        _store = ReferenceDataStore()
        _store.load_from_dict(_DEFAULT_REFERENCE_DATA)
    return _store


# ── HTTP Request Handler (Reference Server) ──────────────────────────


class MCPRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for the MCP reference data server.

    Teams can subclass or replace this handler with their own implementation.
    The contract is: POST /lookup accepts MCPLookupRequest JSON, returns MCPLookupResponse JSON.
    """

    store: ReferenceDataStore = None  # type: ignore[assignment]

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
        canonical, confidence, alternatives, matched = store.lookup(lookup_key, lookup_value)

        response = {
            "lookup_key": lookup_key,
            "original_value": lookup_value,
            "canonical_value": canonical,
            "confidence": confidence,
            "source": "reference_implementation",
            "alternatives": alternatives,
            "matched": matched,
        }

        response_bytes = json.dumps(response).encode("utf-8")
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
    canonical, confidence, alternatives, matched = store.lookup(lookup_key, lookup_value)
    return MCPLookupResponse(
        lookup_key=lookup_key,
        original_value=lookup_value,
        canonical_value=canonical,
        confidence=confidence,
        source="reference_implementation_direct",
        alternatives=alternatives,
        matched=matched,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    server = create_mcp_server()
    print(f"MCP Reference Data Server running on port 8100")
    print("POST /lookup — accepts MCPLookupRequest, returns MCPLookupResponse")
    server.serve_forever()
