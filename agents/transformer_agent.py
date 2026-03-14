"""Transformer Agent — value normalization via MCP reference data (§10.2).

Normalizes extracted field values using:
1. MCP reference data lookup (team-specific servers)
2. Built-in transformations (date formatting, currency, regex, case)

Teams implement their own MCP servers for reference data but MUST adhere
to the MCPLookupRequest/MCPLookupResponse contracts.  This module contains
a reference implementation that teams can use as a starting point.

All MCP lookups are logged for auditability.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from agents.contracts import (
    AgentMessage,
    ExtractedField,
    MCPLookupRequest,
    MCPLookupResponse,
    TransformResult,
    TransformationRule,
    new_id,
)
from agents.message_bus import MessageBus
from agents.model_gateway import ModelGateway

logger = logging.getLogger(__name__)


# ── MCP Client Protocol ──────────────────────────────────────────────
# Teams MUST implement servers that accept MCPLookupRequest and return
# MCPLookupResponse.  The client below calls any MCP-compliant server.


class MCPReferenceClient:
    """Client for calling MCP reference data servers.

    This is the platform-provided client.  It calls any MCP server that
    implements the standard MCPLookupRequest/MCPLookupResponse contract.

    Teams deploy their own MCP servers (e.g. for entity resolution,
    currency codes, jurisdiction mapping) and configure the URL.
    """

    def __init__(self, timeout_s: int = 10) -> None:
        self._timeout_s = timeout_s

    def lookup(
        self,
        server_url: str,
        request: MCPLookupRequest,
    ) -> MCPLookupResponse:
        """Call an MCP reference data server.

        Args:
            server_url: The MCP server endpoint (e.g. http://localhost:8100/lookup)
            request: The lookup request (standard contract)

        Returns:
            MCPLookupResponse from the server

        The server MUST accept POST with JSON body matching MCPLookupRequest
        and return JSON matching MCPLookupResponse.
        """
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "lookup_key": request.lookup_key,
            "lookup_value": request.lookup_value,
            "context": request.context,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{server_url.rstrip('/')}/lookup",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                canonical = data.get("canonical_value", request.lookup_value)
                return MCPLookupResponse(
                    lookup_key=data.get("lookup_key", request.lookup_key),
                    original_value=data.get("original_value", request.lookup_value),
                    canonical_value=canonical,
                    code=data.get("code", canonical),
                    description=data.get("description", canonical),
                    details=data.get("details", {}),
                    confidence=float(data.get("confidence", 1.0)),
                    source=data.get("source", ""),
                    alternatives=data.get("alternatives", []),
                    matched=data.get("matched", True),
                )
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError) as exc:
            logger.warning("MCP lookup failed for %s=%s: %s", request.lookup_key, request.lookup_value, exc)
            return MCPLookupResponse(
                lookup_key=request.lookup_key,
                original_value=request.lookup_value,
                canonical_value=request.lookup_value,
                confidence=0.0,
                matched=False,
            )


# ── Transformation Rule Registry ─────────────────────────────────────

_TRANSFORM_RULES: Dict[str, List[TransformationRule]] = {}


def register_transform_rules(schema_id: str, rules: List[TransformationRule]) -> None:
    """Register transformation rules for an extraction schema."""
    _TRANSFORM_RULES[schema_id] = rules


def get_transform_rules(schema_id: str) -> List[TransformationRule]:
    """Get transformation rules for a schema."""
    return _TRANSFORM_RULES.get(schema_id, [])


class TransformerAgent(BaseAgent):
    """Value normalization agent with MCP reference data lookup (§10.2).

    Flow:
    1. Receives ExtractedFields from the Extractor Agent
    2. For each field that needs normalization:
       a. Check if field has normalize_via_mcp=True → call MCP server
       b. Apply any registered TransformationRules (regex, date, case)
    3. Returns TransformResult with normalized_value populated
    """

    agent_name = "transformer"

    def __init__(
        self,
        bus: MessageBus,
        gateway: Optional[ModelGateway] = None,
        mcp_client: Optional[MCPReferenceClient] = None,
        default_mcp_url: str = "http://localhost:8100",
    ) -> None:
        super().__init__(bus, gateway)
        self._mcp_client = mcp_client or MCPReferenceClient()
        self._default_mcp_url = default_mcp_url

    def handle_message(self, message: AgentMessage) -> TransformResult:
        """Handle a transform request."""
        payload = message.payload
        doc_id = payload["doc_id"]
        query_id = message.query_id
        schema_id = payload.get("schema_id", "")

        # Deserialize fields from payload
        raw_fields = payload.get("fields", [])
        fields = []
        for f in raw_fields:
            if isinstance(f, ExtractedField):
                fields.append(f)
            elif isinstance(f, dict):
                fields.append(ExtractedField(**f))

        rules = get_transform_rules(schema_id)
        return self.transform(doc_id, fields, rules, query_id)

    def transform(
        self,
        doc_id: str,
        fields: List[ExtractedField],
        rules: List[TransformationRule],
        query_id: str = "",
    ) -> TransformResult:
        """Apply transformations and MCP lookups to extracted fields."""
        transformed: List[ExtractedField] = []
        mcp_lookups = 0
        mcp_failures = 0
        warnings: List[str] = []

        rule_map = {r.field_name: r for r in rules}

        for field in fields:
            result_field = field
            rule = rule_map.get(field.field_name)

            if rule and rule.transform_type == "mcp_lookup":
                # MCP reference data lookup
                mcp_lookups += 1
                mcp_response = self._mcp_lookup(
                    server_url=rule.mcp_server_url or self._default_mcp_url,
                    lookup_key=rule.mcp_lookup_key or field.field_name,
                    lookup_value=field.raw_value,
                    context={"doc_id": doc_id, "field_name": field.field_name},
                )
                if mcp_response.matched:
                    result_field = replace(
                        field,
                        normalized_value=mcp_response.canonical_value,
                        confidence=max(field.confidence, mcp_response.confidence),
                        extraction_method="mcp_normalized",
                    )
                else:
                    mcp_failures += 1
                    warnings.append(f"MCP lookup failed for {field.field_name}={field.raw_value}")
                    result_field = replace(field, normalized_value=field.raw_value)

            elif rule and rule.transform_type == "regex_replace":
                if rule.regex_pattern and field.raw_value:
                    normalized = re.sub(rule.regex_pattern, rule.regex_replacement, field.raw_value)
                    result_field = replace(field, normalized_value=normalized)
                else:
                    result_field = replace(field, normalized_value=field.raw_value)

            elif rule and rule.transform_type == "date_format":
                result_field = self._transform_date(field, rule)

            elif rule and rule.transform_type == "uppercase":
                result_field = replace(field, normalized_value=field.raw_value.upper())

            elif rule and rule.transform_type == "lowercase":
                result_field = replace(field, normalized_value=field.raw_value.lower())

            elif rule and rule.transform_type == "currency_convert":
                # Currency conversion via MCP
                mcp_lookups += 1
                mcp_response = self._mcp_lookup(
                    server_url=rule.mcp_server_url or self._default_mcp_url,
                    lookup_key="currency_code",
                    lookup_value=field.raw_value,
                    context={"doc_id": doc_id},
                )
                if mcp_response.matched:
                    result_field = replace(
                        field,
                        normalized_value=mcp_response.canonical_value,
                        extraction_method="mcp_normalized",
                    )
                else:
                    mcp_failures += 1
                    result_field = replace(field, normalized_value=field.raw_value)

            else:
                # No transformation rule — keep raw value
                result_field = replace(field, normalized_value=field.raw_value)

            transformed.append(result_field)

        return TransformResult(
            doc_id=doc_id,
            query_id=query_id,
            transformed_fields=transformed,
            mcp_lookups_performed=mcp_lookups,
            mcp_lookups_failed=mcp_failures,
            warnings=warnings,
        )

    def _mcp_lookup(
        self,
        server_url: str,
        lookup_key: str,
        lookup_value: str,
        context: Dict[str, str],
    ) -> MCPLookupResponse:
        """Perform an MCP reference data lookup."""
        request = MCPLookupRequest(
            lookup_key=lookup_key,
            lookup_value=lookup_value,
            context=context,
        )
        logger.info("MCP lookup: %s=%s → %s", lookup_key, lookup_value, server_url)
        return self._mcp_client.lookup(server_url, request)

    def _transform_date(
        self, field: ExtractedField, rule: TransformationRule
    ) -> ExtractedField:
        """Transform date format."""
        if not field.raw_value or not rule.date_input_format:
            return replace(field, normalized_value=field.raw_value)
        try:
            dt = datetime.strptime(field.raw_value, rule.date_input_format)
            output_fmt = rule.date_output_format or "%Y-%m-%d"
            return replace(field, normalized_value=dt.strftime(output_fmt))
        except ValueError:
            return replace(field, normalized_value=field.raw_value)
