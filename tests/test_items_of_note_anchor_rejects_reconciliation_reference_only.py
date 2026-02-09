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


def test_items_of_note_anchor_rejects_reconciliation_reference_only(monkeypatch):
    query = "List the items of note affecting 2024 net income and the aggregate impact"
    bad_reference = _chunk(
        "c-ref",
        "The calculation of adjusted measures is adjusted to exclude the impact of items of note. "
        "For additional information and a reconciliation of reported results to adjusted measures, see the MD&A.",
        "Risk/LCR ratio",
    )
    good = _chunk(
        "c-mdna",
        "Items of note affecting 2024 net income: "
        "FDIC special assessment ($0.3 billion after tax), "
        "Acquisition-related intangibles ($0.1 billion after tax). "
        "Aggregate impact ($0.4 billion after tax).",
        "MD&A/Items of note",
    )

    intent = router.classify_query(query)
    assert intent.intent == "coverage"
    assert intent.coverage_type == "numeric_list"

    monkeypatch.setattr(router.settings, "enable_hybrid_retrieval", True)
    monkeypatch.setattr(
        router,
        "bm25_heading_anchor_candidates",
        lambda *args, **kwargs: [bad_reference, good],
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
    rejected = [
        item
        for item in decisions
        if item["chunk_id"] == "c-ref"
        and "reject_reconciliation_reference_only" in item["reasons"]
    ]
    assert rejected


def test_itemization_detector_only_counts_financial_impacts():
    false_cases = [
        "LCR (12) … 118% … (6) Adjusted measures are non-GAAP…",
        "Ratios for 2020, 2021 and 2022 reflect…",
    ]
    true_cases = [
        "FDIC special assessment ($0.3 billion after tax) and acquisition-related intangibles ($0.1 billion after tax).",
        "Aggregate impact ($0.4 billion after tax).",
    ]

    for text in false_cases:
        assert router._count_financial_impact_mentions(text) == 0
    for text in true_cases:
        assert router._count_financial_impact_mentions(text) >= 1
