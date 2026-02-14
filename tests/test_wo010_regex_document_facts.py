"""WO-010: Regression tests for document_facts regex patterns."""

import pytest

from ingestion.document_facts import (
    extract_document_facts,
    _match_default_currency,
    _match_units,
    _match_reporting_period,
)


def _chunk(text: str):
    from core.contracts import RetrievedChunk
    return RetrievedChunk(
        chunk_id="c1",
        doc_id="doc",
        page_numbers=[1],
        macro_id=0,
        child_id=0,
        chunk_type="narrative",
        text_content=text,
        char_start=0,
        char_end=len(text),
        polygons=[],
        source_type="native",
        score=0.0,
        heading_path="",
        section_id="",
    )


def test_default_currency_canadian():
    assert _match_default_currency("All amounts are in Canadian dollars unless otherwise stated.") == "Canadian dollars"


def test_default_currency_us():
    assert _match_default_currency("All amounts are in U.S. dollars.") == "U.S. dollars"


def test_default_currency_negative():
    assert _match_default_currency("Revenue increased 5%.") is None


def test_units_millions():
    assert _match_units("All amounts are in Canadian millions unless noted.") == "millions"


def test_units_billions():
    assert _match_units("Amounts in U.S. billions.") is None  # "amounts in" not "all amounts are in"


def test_units_all_amounts_billions():
    assert _match_units("All amounts are in millions except per-share data.") == "millions"


def test_reporting_period_positive():
    assert _match_reporting_period("For the year ended October 31, 2024") == "October 31, 2024"


def test_reporting_period_negative():
    assert _match_reporting_period("Q3 2024 revenue") is None


def test_extract_currency_from_fixture_style():
    """Integration: extract_document_facts finds Canadian dollars."""
    chunks = [_chunk("All amounts are in Canadian dollars unless otherwise stated.")]
    facts = extract_document_facts("doc", chunks)
    currency = next(f for f in facts if f.fact_name == "default_currency")
    assert currency.status == "found"
    assert currency.value == "Canadian dollars"
