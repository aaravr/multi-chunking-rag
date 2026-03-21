"""Tests for the Data Extractor Agent (§10.1)."""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

from agents.contracts import (
    AgentMessage,
    ExtractionSchema,
    FieldDefinition,
    ExtractedField,
    ExtractionResult,
    new_id,
)
from agents.extractor_agent import (
    ExtractorAgent,
    register_schema,
    get_schema,
    list_schemas,
    _SCHEMA_REGISTRY,
)
from agents.message_bus import MessageBus


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear schema registry between tests."""
    _SCHEMA_REGISTRY.clear()
    yield
    _SCHEMA_REGISTRY.clear()


@pytest.fixture
def sample_schema():
    return ExtractionSchema(
        schema_id="test-schema-001",
        schema_name="10-K Annual Report",
        document_type="10-K",
        classification_label="sec_filing",
        fields=[
            FieldDefinition(
                field_name="total_revenue",
                display_name="Total Revenue",
                field_type="currency",
                required=True,
                description="Total revenue for the reporting period",
                validation_regex=r"\$?[\d,]+\.?\d*",
                extraction_hint="Look in the income statement or financial highlights",
            ),
            FieldDefinition(
                field_name="fiscal_year_end",
                display_name="Fiscal Year End",
                field_type="date",
                required=True,
                description="End date of the fiscal year",
            ),
            FieldDefinition(
                field_name="company_name",
                display_name="Company Name",
                field_type="text",
                required=True,
                description="Legal name of the reporting entity",
            ),
            FieldDefinition(
                field_name="document_status",
                display_name="Document Status",
                field_type="text",
                required=False,
                allowed_values=["draft", "final", "amended"],
            ),
        ],
    )


@pytest.fixture
def mock_chunk():
    """Create a mock chunk with the expected attributes."""
    chunk = MagicMock()
    chunk.chunk_id = "chunk-001"
    chunk.text_content = (
        "ACME Corporation reported total revenue of $5,432,100.00 "
        "for the fiscal year ending December 31, 2024. "
        "This annual report (10-K) is filed with the SEC."
    )
    chunk.page_numbers = [1, 2]
    return chunk


@pytest.fixture
def bus():
    return MessageBus()


@pytest.fixture
def gateway():
    def mock_caller(model_id, messages, temperature):
        return {
            "content": '{"fiscal_year_end": "December 31, 2024", "company_name": "ACME Corporation"}',
            "input_tokens": 100,
            "output_tokens": 50,
        }
    from agents.model_gateway import ModelGateway
    return ModelGateway(llm_caller=mock_caller)


# ── Schema Registry Tests ────────────────────────────────────────────


def test_register_and_get_schema(sample_schema):
    register_schema(sample_schema)
    result = get_schema("10-K", "sec_filing")
    assert result is not None
    assert result.schema_id == "test-schema-001"
    assert len(result.fields) == 4


def test_list_schemas(sample_schema):
    register_schema(sample_schema)
    schemas = list_schemas()
    assert len(schemas) == 1
    assert schemas[0].schema_name == "10-K Annual Report"


def test_get_schema_not_found():
    assert get_schema("unknown", "unknown") is None


# ── Deterministic Extraction Tests ───────────────────────────────────


def test_deterministic_regex_extraction(bus, gateway, sample_schema, mock_chunk):
    register_schema(sample_schema)
    agent = ExtractorAgent(bus, gateway)

    result = agent.extract(
        doc_id="doc-001",
        chunks=[mock_chunk],
        schema=sample_schema,
        query_id="q-001",
    )

    assert isinstance(result, ExtractionResult)
    assert result.doc_id == "doc-001"
    assert result.schema_id == "test-schema-001"

    # total_revenue should be extracted via regex (first match of the pattern)
    revenue_field = next((f for f in result.fields if f.field_name == "total_revenue"), None)
    assert revenue_field is not None
    assert revenue_field.extraction_method == "regex"
    assert revenue_field.confidence == 0.9
    # The regex \$?[\d,]+\.?\d* matches currency values in the text
    assert revenue_field.raw_value  # Should find some numeric match


def test_llm_fallback_extraction(bus, gateway, sample_schema, mock_chunk):
    register_schema(sample_schema)
    agent = ExtractorAgent(bus, gateway)

    result = agent.extract(
        doc_id="doc-001",
        chunks=[mock_chunk],
        schema=sample_schema,
        query_id="q-001",
    )

    # Fields without regex should fall back to LLM
    company_field = next((f for f in result.fields if f.field_name == "company_name"), None)
    assert company_field is not None
    assert company_field.extraction_method == "llm"
    assert company_field.raw_value == "ACME Corporation"


# ── Validation Tests ─────────────────────────────────────────────────


def test_allowed_values_validation(bus, gateway, mock_chunk):
    schema = ExtractionSchema(
        schema_id="val-schema",
        schema_name="Validation Test",
        document_type="test",
        classification_label="test",
        fields=[
            FieldDefinition(
                field_name="status",
                display_name="Status",
                field_type="text",
                allowed_values=["active", "inactive"],
                validation_regex=r"active|inactive",
            ),
        ],
    )

    # Mock chunk with invalid value
    chunk = MagicMock()
    chunk.chunk_id = "c1"
    chunk.text_content = "The status is pending review."
    chunk.page_numbers = [1]

    agent = ExtractorAgent(bus, gateway)
    result = agent.extract("doc-001", [chunk], schema, "q-001")
    # No regex match → falls to LLM → validation may fail
    assert isinstance(result, ExtractionResult)


# ── Message Bus Integration ──────────────────────────────────────────


def test_handle_message(bus, gateway, sample_schema, mock_chunk):
    register_schema(sample_schema)
    agent = ExtractorAgent(bus, gateway)

    message = AgentMessage(
        message_id=new_id(),
        query_id="q-001",
        from_agent="orchestrator",
        to_agent="extractor",
        message_type="extract",
        payload={
            "doc_id": "doc-001",
            "chunks": [mock_chunk],
            "document_type": "10-K",
            "classification_label": "sec_filing",
        },
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    result = bus.send(message)
    assert isinstance(result, ExtractionResult)
    assert result.doc_id == "doc-001"


def test_no_schema_returns_warning(bus, gateway, mock_chunk):
    agent = ExtractorAgent(bus, gateway)

    message = AgentMessage(
        message_id=new_id(),
        query_id="q-001",
        from_agent="orchestrator",
        to_agent="extractor",
        message_type="extract",
        payload={
            "doc_id": "doc-001",
            "chunks": [mock_chunk],
            "document_type": "unknown",
            "classification_label": "unknown",
        },
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    result = bus.send(message)
    assert isinstance(result, ExtractionResult)
    assert len(result.warnings) > 0


# ── Overall Confidence ───────────────────────────────────────────────


def test_overall_confidence(bus, gateway, sample_schema, mock_chunk):
    register_schema(sample_schema)
    agent = ExtractorAgent(bus, gateway)
    result = agent.extract("doc-001", [mock_chunk], sample_schema, "q-001")
    assert 0.0 <= result.overall_confidence <= 1.0
