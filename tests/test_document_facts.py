from pathlib import Path

from core import config
from core.contracts import DocumentFact, RetrievedChunk
from ingestion.document_facts import extract_document_facts
from retrieval import metadata


def _chunk(text: str) -> RetrievedChunk:
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
        polygons=[{"page_number": 1, "polygon": [{"x": 0, "y": 0}]}],
        source_type="native",
        score=0.0,
        heading_path="Front matter",
        section_id="front-matter",
    )


def test_extract_default_currency_from_fixture():
    fixture = Path(__file__).parent / "fixtures" / "document_facts_currency.txt"
    chunk = _chunk(fixture.read_text())
    facts = extract_document_facts("doc", [chunk])
    currency = next(f for f in facts if f.fact_name == "default_currency")
    assert currency.status == "found"
    assert currency.value == "Canadian dollars"
    assert currency.source_chunk_id == "c1"


def test_metadata_query_returns_cached_value(monkeypatch):
    fact = DocumentFact(
        doc_id="doc",
        fact_name="default_currency",
        value="Canadian dollars",
        status="found",
        confidence=0.9,
        source_chunk_id="c1",
        page_numbers=[1],
        polygons=[{"page_number": 1, "polygon": [{"x": 0, "y": 0}]}],
        evidence_excerpt="All amounts are in Canadian dollars unless otherwise stated.",
    )
    monkeypatch.setattr(metadata, "_fetch_fact", lambda *_args, **_kwargs: fact)
    answer, chunks, info = metadata.handle_metadata_query(
        "doc", "default currency"
    )
    assert "Canadian dollars" in answer
    assert "[C1]" in answer
    assert chunks
    assert info["status"] == "found"


def test_metadata_query_not_found(monkeypatch):
    monkeypatch.setattr(metadata, "_fetch_fact", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(metadata.vector_search, "search_on_pages", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(metadata, "_heading_phrase_candidates", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(metadata, "_bm25_narrative_candidates", lambda *_args, **_kwargs: [])
    answer, chunks, info = metadata.handle_metadata_query(
        "doc", "default currency"
    )
    assert "not found" in answer.lower()
    assert info["status"] == "not_found"
    assert chunks == []


def test_metadata_query_finds_currency_after_page_three(monkeypatch):
    config.settings.front_matter_pages = 10
    from dataclasses import replace

    late_chunk = _chunk("All amounts are in Canadian dollars unless otherwise stated.")
    late_chunk = replace(late_chunk, page_numbers=[5])
    monkeypatch.setattr(metadata, "_fetch_fact", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(metadata.vector_search, "search_on_pages", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(metadata, "_heading_phrase_candidates", lambda *_args, **_kwargs: [late_chunk])
    monkeypatch.setattr(metadata, "_bm25_narrative_candidates", lambda *_args, **_kwargs: [])
    answer, chunks, info = metadata.handle_metadata_query(
        "doc", "default currency"
    )
    assert "Canadian dollars" in answer
    assert "[C1]" in answer
    assert info["status"] == "found"
