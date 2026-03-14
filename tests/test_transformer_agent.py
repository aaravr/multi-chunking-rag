"""Tests for the Transformer Agent and MCP reference data (§10.2, §10.3)."""

import pytest
from datetime import datetime, timezone

from agents.contracts import (
    AgentMessage,
    ExtractedField,
    MCPLookupRequest,
    MCPLookupResponse,
    TransformResult,
    TransformationRule,
    new_id,
)
from agents.transformer_agent import (
    TransformerAgent,
    MCPReferenceClient,
    register_transform_rules,
    get_transform_rules,
    _TRANSFORM_RULES,
)
from agents.mcp_reference_server import (
    ReferenceDataStore,
    direct_lookup,
    get_reference_store,
)
from agents.message_bus import MessageBus


@pytest.fixture(autouse=True)
def clear_rules():
    _TRANSFORM_RULES.clear()
    yield
    _TRANSFORM_RULES.clear()


@pytest.fixture
def bus():
    return MessageBus()


@pytest.fixture
def sample_fields():
    return [
        ExtractedField(
            field_name="company_name",
            raw_value="RBC",
            confidence=0.8,
            source_chunk_ids=["c1"],
            page_numbers=[1],
        ),
        ExtractedField(
            field_name="currency",
            raw_value="US Dollar",
            confidence=0.9,
            source_chunk_ids=["c2"],
            page_numbers=[2],
        ),
        ExtractedField(
            field_name="filing_date",
            raw_value="12/31/2024",
            confidence=0.85,
            source_chunk_ids=["c3"],
            page_numbers=[1],
        ),
        ExtractedField(
            field_name="jurisdiction",
            raw_value="us",
            confidence=0.7,
            source_chunk_ids=["c4"],
            page_numbers=[3],
        ),
    ]


@pytest.fixture
def transform_rules():
    return [
        TransformationRule(
            field_name="company_name",
            transform_type="mcp_lookup",
            mcp_lookup_key="entity_name",
        ),
        TransformationRule(
            field_name="currency",
            transform_type="mcp_lookup",
            mcp_lookup_key="currency_code",
        ),
        TransformationRule(
            field_name="filing_date",
            transform_type="date_format",
            date_input_format="%m/%d/%Y",
            date_output_format="%Y-%m-%d",
        ),
        TransformationRule(
            field_name="jurisdiction",
            transform_type="uppercase",
        ),
    ]


# ── MCP Reference Store Tests ────────────────────────────────────────


def test_reference_store_exact_match():
    store = ReferenceDataStore()
    store.add_reference("entity_name", "Royal Bank of Canada", ["RBC"])
    canonical, confidence, alternatives, matched = store.lookup("entity_name", "RBC")
    assert canonical == "Royal Bank of Canada"
    assert confidence == 0.95
    assert matched is True


def test_reference_store_no_match():
    store = ReferenceDataStore()
    store.add_reference("entity_name", "Royal Bank of Canada", ["RBC"])
    canonical, confidence, alternatives, matched = store.lookup("entity_name", "Unknown Corp")
    assert matched is False
    assert confidence == 0.0


def test_reference_store_fuzzy_match():
    store = ReferenceDataStore()
    store.add_reference("entity_name", "Royal Bank of Canada", ["Royal Bank"])
    canonical, confidence, alternatives, matched = store.lookup("entity_name", "Royal Bank Group")
    assert matched is True
    assert confidence == 0.7


def test_direct_lookup_entity():
    response = direct_lookup("entity_name", "RBC")
    assert isinstance(response, MCPLookupResponse)
    assert response.canonical_value == "Royal Bank of Canada"
    assert response.matched is True


def test_direct_lookup_currency():
    response = direct_lookup("currency_code", "US Dollar")
    assert isinstance(response, MCPLookupResponse)
    assert response.canonical_value == "USD"
    assert response.matched is True


def test_direct_lookup_jurisdiction():
    response = direct_lookup("jurisdiction", "US")
    assert isinstance(response, MCPLookupResponse)
    assert response.canonical_value == "United States"
    assert response.matched is True


def test_direct_lookup_no_match():
    response = direct_lookup("entity_name", "Completely Unknown Entity XYZ123")
    assert isinstance(response, MCPLookupResponse)
    assert response.matched is False


# ── Transformer Agent Tests ──────────────────────────────────────────


class MockMCPClient(MCPReferenceClient):
    """Mock MCP client that uses direct_lookup instead of HTTP."""

    def lookup(self, server_url, request):
        return direct_lookup(request.lookup_key, request.lookup_value)


def test_transformer_mcp_lookup(bus, sample_fields, transform_rules):
    agent = TransformerAgent(bus, mcp_client=MockMCPClient())
    result = agent.transform("doc-001", sample_fields, transform_rules, "q-001")

    assert isinstance(result, TransformResult)
    assert result.mcp_lookups_performed == 2  # company_name + currency

    company = next(f for f in result.transformed_fields if f.field_name == "company_name")
    assert company.normalized_value == "Royal Bank of Canada"
    assert company.extraction_method == "mcp_normalized"


def test_transformer_date_format(bus, sample_fields, transform_rules):
    agent = TransformerAgent(bus, mcp_client=MockMCPClient())
    result = agent.transform("doc-001", sample_fields, transform_rules, "q-001")

    date_field = next(f for f in result.transformed_fields if f.field_name == "filing_date")
    assert date_field.normalized_value == "2024-12-31"


def test_transformer_uppercase(bus, sample_fields, transform_rules):
    agent = TransformerAgent(bus, mcp_client=MockMCPClient())
    result = agent.transform("doc-001", sample_fields, transform_rules, "q-001")

    jurisdiction = next(f for f in result.transformed_fields if f.field_name == "jurisdiction")
    assert jurisdiction.normalized_value == "US"


def test_transformer_no_rules(bus, sample_fields):
    agent = TransformerAgent(bus, mcp_client=MockMCPClient())
    result = agent.transform("doc-001", sample_fields, [], "q-001")

    # All fields should have normalized_value = raw_value
    for f in result.transformed_fields:
        assert f.normalized_value == f.raw_value


def test_transformer_message_bus(bus, sample_fields, transform_rules):
    register_transform_rules("schema-001", transform_rules)
    agent = TransformerAgent(bus, mcp_client=MockMCPClient())

    payload_fields = []
    for f in sample_fields:
        payload_fields.append({
            "field_name": f.field_name,
            "raw_value": f.raw_value,
            "confidence": f.confidence,
            "source_chunk_ids": f.source_chunk_ids,
            "page_numbers": f.page_numbers,
        })

    message = AgentMessage(
        message_id=new_id(),
        query_id="q-001",
        from_agent="orchestrator",
        to_agent="transformer",
        message_type="transform",
        payload={
            "doc_id": "doc-001",
            "schema_id": "schema-001",
            "fields": payload_fields,
        },
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    result = bus.send(message)
    assert isinstance(result, TransformResult)
    assert result.doc_id == "doc-001"


# ── Rule Registry Tests ──────────────────────────────────────────────


def test_register_and_get_rules(transform_rules):
    register_transform_rules("schema-001", transform_rules)
    rules = get_transform_rules("schema-001")
    assert len(rules) == 4


def test_get_rules_empty():
    assert get_transform_rules("nonexistent") == []


# ── MCP Contract Compliance ──────────────────────────────────────────


def test_mcp_request_contract():
    """Verify MCPLookupRequest is frozen and has required fields."""
    req = MCPLookupRequest(
        lookup_key="entity_name",
        lookup_value="RBC",
        context={"doc_type": "10-K"},
    )
    assert req.lookup_key == "entity_name"
    assert req.lookup_value == "RBC"
    with pytest.raises(AttributeError):
        req.lookup_key = "changed"  # type: ignore[misc]


def test_mcp_response_contract():
    """Verify MCPLookupResponse is frozen and has required fields."""
    resp = MCPLookupResponse(
        lookup_key="entity_name",
        original_value="RBC",
        canonical_value="Royal Bank of Canada",
        confidence=0.95,
        matched=True,
    )
    assert resp.canonical_value == "Royal Bank of Canada"
    assert resp.confidence == 0.95
    with pytest.raises(AttributeError):
        resp.canonical_value = "changed"  # type: ignore[misc]
