from core.contracts import RetrievedChunk
from retrieval import router


def _chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc",
        page_numbers=[120],
        macro_id=1,
        child_id=0,
        chunk_type="narrative",
        text_content=text,
        char_start=0,
        char_end=len(text),
        polygons=[],
        source_type="native",
        score=0.0,
        heading_path="Notes/Note 21/Significant legal proceedings",
        section_id="note-21",
    )


def test_closed_matters_plan_section_read(monkeypatch):
    query = "Which matters are explicitly closed (and what closed them)?"
    anchor = _chunk("a1", "Note 21 Significant legal proceedings")
    expanded = [
        _chunk("c1", "Fresco/Gaudet closed via settlement."),
        _chunk("c2", "Cerberus closed via settlement."),
    ]

    monkeypatch.setattr(router, "bm25_heading_anchor", lambda *_args, **_kwargs: anchor)
    monkeypatch.setattr(
        router.vector_search,
        "fetch_by_section",
        lambda *_args, **_kwargs: expanded,
    )
    results, debug = router.search_with_intent_debug("doc", query, top_k=3)
    assert results
    assert debug["expansion"]["method"] == "section"
    assert "note-21" in (debug["expansion"]["section_id"] or "")
