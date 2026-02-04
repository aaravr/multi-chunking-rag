from core.contracts import RetrievedChunk
from retrieval import router


def _chunk(chunk_id: str, text: str, chunk_type: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc",
        page_numbers=[100],
        macro_id=0,
        chunk_type=chunk_type,
        text_content=text,
        char_start=0,
        char_end=len(text),
        polygons=[],
        source_type="native",
        score=0.0,
        heading_path="MD&A/Items of note",
        section_id="items-of-note",
    )


def test_items_of_note_anchor_excludes_table(monkeypatch):
    query = "List the items of note affecting 2024 net income"
    table_candidate = _chunk(
        "c-table",
        "[TABLE] Items of note consolidated financial statements",
        "table",
    )
    narrative_candidate = _chunk(
        "c-note",
        "Items of note include FDIC special assessment.",
        "narrative",
    )
    monkeypatch.setattr(
        router.settings, "enable_hybrid_retrieval", True
    )
    monkeypatch.setattr(
        router,
        "bm25_heading_anchor_candidates",
        lambda *args, **kwargs: [table_candidate, narrative_candidate],
    )
    monkeypatch.setattr(
        router,
        "_expand_from_anchor",
        lambda *_args, **_kwargs: ([narrative_candidate], {"method": "section"}),
    )
    results, debug = router.search_with_intent_debug("doc", query, top_k=3)
    assert results
    assert debug["anchor"]["chunk_type"] != "table"
    snippet = debug["top_chunks"][0]["snippet"].lower()
    assert "items of note" in snippet or "fdic special assessment" in snippet
