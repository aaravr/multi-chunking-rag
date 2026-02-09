from core.contracts import RetrievedChunk
from retrieval import router


def _chunk(chunk_id: str, text: str, heading: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
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
        heading_path=heading,
        section_id=heading,
    )


def test_items_of_note_anchor_not_adjusted_measures(monkeypatch):
    query = "List the items of note affecting 2024 net income and the aggregate impact"
    bad_adjusted = _chunk(
        "c-lcr",
        "Adjusted measures are non-GAAP measures used to assess performance. "
        "Common shareholders' equity divided by risk-weighted assets.",
        "Risk/LCR ratio",
    )
    bad_front = _chunk(
        "c-front",
        "Items of note. See the Glossary for definitions and cross-reference.",
        "Annual/Report",
    )
    good = _chunk(
        "c-mdna",
        "Items of note reconciliation to net income: FDIC special assessment ($123), "
        "Acquisition-related intangibles ($45). Aggregate impact ($168).",
        "MD&A/Items of note",
    )

    intent = router.classify_query(query)
    assert intent.intent == "coverage"
    assert intent.coverage_type == "numeric_list"

    monkeypatch.setattr(router.settings, "enable_hybrid_retrieval", True)
    monkeypatch.setattr(
        router,
        "bm25_heading_anchor_candidates",
        lambda *args, **kwargs: [bad_adjusted, bad_front, good],
    )
    monkeypatch.setattr(
        router,
        "_expand_from_anchor",
        lambda *_args, **_kwargs: ([good], {"method": "section"}),
    )
    results, debug = router.search_with_intent_debug("doc", query, top_k=3)
    assert results
    assert debug["anchor"]["chunk_id"] == "c-mdna"

    decisions = debug["anchor_decisions"]
    lcr_rejected = [
        item
        for item in decisions
        if item["chunk_id"] == "c-lcr"
        and "reject_adjusted_measures_definition" in item["reasons"]
    ]
    front_rejected = [
        item
        for item in decisions
        if item["chunk_id"] == "c-front"
        and "front_matter_reference" in item["reasons"]
    ]
    assert lcr_rejected
    assert front_rejected
