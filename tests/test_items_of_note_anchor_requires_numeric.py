from core.contracts import RetrievedChunk
from retrieval import router


def _chunk(chunk_id: str, text: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc",
        page_numbers=[page],
        macro_id=0,
        child_id=0,
        chunk_type="narrative",
        text_content=text,
        char_start=0,
        char_end=len(text),
        polygons=[],
        source_type="native",
        score=0.0,
        heading_path="MD&A/Items of note",
        section_id="items-of-note",
    )


def test_items_of_note_anchor_requires_numeric(monkeypatch):
    front = _chunk("c-front", "Items of note section overview.", 1)
    later = _chunk("c-later", "Items of note: 1) FDIC special assessment", 5)

    monkeypatch.setattr(router.settings, "enable_hybrid_retrieval", True)
    monkeypatch.setattr(
        router,
        "bm25_heading_anchor_candidates",
        lambda *args, **kwargs: [front, later],
    )
    monkeypatch.setattr(
        router,
        "_expand_from_anchor",
        lambda *_args, **_kwargs: ([later], {"method": "section"}),
    )
    results, debug = router.search_with_intent_debug(
        "doc", "List the items of note affecting 2024 net income", top_k=3
    )
    assert results
    assert debug["anchor"]["chunk_id"] == "c-later"


def test_items_of_note_anchor_rejects_front_matter(monkeypatch):
    front = _chunk(
        "c-front",
        "Items of note. See glossary for definitions.",
        1,
    )
    later = _chunk(
        "c-later",
        "Items of note: 1) FDIC special assessment 2) Acquisition-related intangibles",
        6,
    )

    monkeypatch.setattr(router.settings, "enable_hybrid_retrieval", True)
    monkeypatch.setattr(
        router,
        "bm25_heading_anchor_candidates",
        lambda *args, **kwargs: [front, later],
    )
    monkeypatch.setattr(
        router,
        "_expand_from_anchor",
        lambda *_args, **_kwargs: ([later], {"method": "section"}),
    )
    results, debug = router.search_with_intent_debug(
        "doc", "List the items of note affecting 2024 net income", top_k=3
    )
    assert results
    assert debug["anchor"]["chunk_id"] == "c-later"
