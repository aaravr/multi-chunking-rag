from pathlib import Path

from core.contracts import RetrievedChunk
from retrieval import router, vector_search
from synthesis.coverage import extract_coverage_items, format_coverage_answer


def _chunk(
    chunk_id: str,
    text: str,
    pages,
    heading_path: str,
    section_id: str,
    macro_id: int = 0,
    chunk_type: str = "narrative",
):
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="cibc-2024",
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


def test_coverage_query_integration_fixture(monkeypatch):
    fixture = Path(__file__).parent / "fixtures" / "cibc_note21_excerpt.txt"
    text = fixture.read_text().strip().splitlines()
    heading_path = "Notes > Note 21 > Contingent Liabilities and Litigation"
    section_id = "note-21-litigation"

    anchor = [_chunk("anchor", text[0], [120], heading_path, section_id, macro_id=4)]
    expanded = [
        _chunk("c1", "\n".join(text[:2]), [120], heading_path, section_id, macro_id=4),
        _chunk("c2", "\n".join(text[2:]), [121], heading_path, section_id, macro_id=4),
    ]

    monkeypatch.setattr(router.settings, "enable_hybrid_retrieval", False)
    monkeypatch.setattr(router, "bm25_heading_anchor", lambda *args, **kwargs: None)
    monkeypatch.setattr(vector_search, "search", lambda *args, **kwargs: anchor)
    monkeypatch.setattr(vector_search, "fetch_by_section", lambda *args, **kwargs: expanded)

    intent = router.classify_query("list all litigation events")
    results = router.search_with_intent("cibc-2024", "list all litigation events")
    items = extract_coverage_items("list all litigation events", results)
    answer = format_coverage_answer("list all litigation events", results)

    assert intent.intent == "coverage"
    assert len(results) > 1
    assert all(r.section_id == section_id for r in results)
    joined = " ".join(i["display"] for i in items).lower()
    assert "fresco" in joined
    assert "cerberus" in joined
    assert "frayce" in joined
    assert "chunk_id=c1" in answer or "chunk_id=c2" in answer
