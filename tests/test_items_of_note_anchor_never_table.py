from pathlib import Path

from core.contracts import RetrievedChunk
from retrieval import router


def _chunk(chunk_id: str, text: str, chunk_type: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc",
        page_numbers=[1],
        macro_id=0,
        child_id=0,
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


def test_items_of_note_anchor_never_table(monkeypatch):
    fixture = Path(__file__).parent / "fixtures" / "items_of_note_fixture.txt"
    text = fixture.read_text()
    narrative = _chunk("c-mdna", text, "narrative")
    table = _chunk("c-note12", "[TABLE] Note 12 Derivative instruments", "table")

    monkeypatch.setattr(router.settings, "enable_hybrid_retrieval", True)
    monkeypatch.setattr(
        router,
        "bm25_heading_anchor_candidates",
        lambda *args, **kwargs: [table, narrative],
    )
    monkeypatch.setattr(
        router,
        "_expand_from_anchor",
        lambda *_args, **_kwargs: ([narrative], {"method": "section"}),
    )
    results, debug = router.search_with_intent_debug(
        "doc", "List the items of note affecting 2024 net income", top_k=3
    )
    assert results
    assert debug["anchor"]["chunk_type"] != "table"
    snippet = debug["top_chunks"][0]["snippet"].lower()
    assert "items of note" in snippet or "fdic special assessment" in snippet
