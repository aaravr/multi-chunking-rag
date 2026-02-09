from typing import List

from core.contracts import RetrievedChunk
from retrieval import router, vector_search
from synthesis.coverage import extract_coverage_items, format_coverage_answer
from synthesis.verifier import verify_coverage


def _chunk(
    chunk_id: str,
    text: str,
    pages: List[int],
    heading_path: str = "",
    section_id: str = "",
    macro_id: int = 0,
    chunk_type: str = "narrative",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        page_numbers=pages,
        macro_id=macro_id,
        child_id=0,
        chunk_type=chunk_type,
        text_content=text,
        char_start=0,
        char_end=len(text),
        polygons=[],
        source_type="native",
        score=0.0,
        heading_path=heading_path,
        section_id=section_id,
    )


def test_coverage_intent_detection():
    intent = router.classify_query("list me all litigation events")
    assert intent.intent == "coverage"
    assert intent.coverage_type == "list"


def test_coverage_section_expansion(monkeypatch):
    anchor = [
        _chunk(
            "c1",
            "Note 21 - Litigation",
            [120],
            heading_path="Notes > Note 21 > Litigation",
            section_id="note-21-litigation",
            macro_id=3,
        )
    ]
    expanded = [
        _chunk(
            "c2",
            "Fresco litigation matter",
            [120],
            heading_path="Notes > Note 21 > Litigation",
            section_id="note-21-litigation",
            macro_id=3,
        ),
        _chunk(
            "c3",
            "Cerberus litigation matter",
            [121],
            heading_path="Notes > Note 21 > Litigation",
            section_id="note-21-litigation",
            macro_id=3,
        ),
    ]

    monkeypatch.setattr(router.settings, "enable_hybrid_retrieval", False)
    monkeypatch.setattr(router, "bm25_heading_anchor", lambda *args, **kwargs: None)
    monkeypatch.setattr(vector_search, "search", lambda *args, **kwargs: anchor)
    monkeypatch.setattr(vector_search, "fetch_by_section", lambda *args, **kwargs: expanded)

    results = router.search_with_intent("doc-1", "list me all litigation events")
    assert len(results) == 2
    assert all(r.section_id == "note-21-litigation" for r in results)


def test_coverage_page_window_fallback(monkeypatch):
    anchor = [_chunk("c1", "Litigation", [50], macro_id=7)]
    fallback = [
        _chunk("c2", "Frayce litigation matter", [49], macro_id=7),
        _chunk("c3", "Fresco litigation matter", [51], macro_id=7),
    ]

    monkeypatch.setattr(router.settings, "enable_hybrid_retrieval", False)
    monkeypatch.setattr(router, "bm25_heading_anchor", lambda *args, **kwargs: None)
    monkeypatch.setattr(vector_search, "search", lambda *args, **kwargs: anchor)
    monkeypatch.setattr(vector_search, "fetch_by_section", lambda *args, **kwargs: [])
    monkeypatch.setattr(vector_search, "fetch_by_macro_id", lambda *args, **kwargs: [])
    monkeypatch.setattr(vector_search, "fetch_by_page_window", lambda *args, **kwargs: fallback)

    try:
        router.search_with_intent("doc-1", "list me all litigation events")
        assert False, "Expected CoverageListQuery expansion error"
    except RuntimeError as exc:
        assert "CoverageListQuery expansion returned no chunks" in str(exc)


def test_coverage_list_extraction_fixture():
    chunks = [
        _chunk(
            "c1",
            "Note 21 - Litigation\nFresco matter details",
            [120],
            heading_path="Notes > Note 21 > Litigation",
            section_id="note-21-litigation",
        ),
        _chunk(
            "c2",
            "Cerberus matter details\nFrayce class action",
            [121],
            heading_path="Notes > Note 21 > Litigation",
            section_id="note-21-litigation",
        ),
    ]
    items = extract_coverage_items("list me all litigation events", chunks)
    joined = " ".join(i["display"] for i in items).lower()
    assert "fresco" in joined
    assert "cerberus" in joined
    assert "frayce" in joined


def test_coverage_verifier_checks_items():
    chunks = [
        _chunk(
            "c1",
            "Fresco matter details",
            [120],
            heading_path="Notes > Note 21 > Litigation",
            section_id="note-21-litigation",
        ),
    ]
    answer = format_coverage_answer("list all litigation events", chunks)
    verdict, rationale = verify_coverage(
        "list all litigation events", answer, chunks
    )
    assert verdict == "YES"
    assert "normalized token" in rationale.lower()
