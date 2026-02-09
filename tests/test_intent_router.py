from core.contracts import RetrievedChunk
from retrieval import router


def test_classify_location_query():
    intent = router.classify_query("See page 22 for details")
    assert intent.intent == "location"
    assert intent.pages == [22]


def test_classify_coverage_query():
    intent = router.classify_query("list all significant events")
    assert intent.intent == "coverage"
    assert intent.coverage_type == "list"


def test_classify_closed_status_query():
    intent = router.classify_query(
        "Which matters are explicitly closed and what closed them"
    )
    assert intent.intent == "coverage"
    assert intent.coverage_type == "list"
    assert intent.status_filter == "closed"


def test_classify_semantic_query():
    intent = router.classify_query("What is the CET1 ratio?")
    assert intent.intent == "semantic"


def test_location_query_routes_to_page_filter(monkeypatch):
    called = {}

    def _search_on_pages(doc_id, query, pages, top_k=3):
        called["pages"] = pages
        return []

    monkeypatch.setattr(router.vector_search, "search_on_pages", _search_on_pages)
    router.search_with_intent("doc-1", "page 3 revenue", top_k=3)
    assert called["pages"] == [3]


def test_coverage_query_expands_section(monkeypatch):
    def _search(doc_id, query, top_k=1):
        class Dummy:
            chunk_id = "c1"
            heading_path = "doc/Section A"
            section_id = "Section A"
            chunk_type = "heading"
            page_numbers = [1]
            macro_id = 0
            child_id = 0
        return [Dummy()]

    def _fetch_by_section(doc_id, heading_path, section_id):
        return [
            RetrievedChunk(
                chunk_id="c1",
                doc_id=doc_id,
                page_numbers=[1],
                macro_id=0,
                child_id=0,
                chunk_type="narrative",
                text_content="Expanded content",
                char_start=0,
                char_end=10,
                polygons=[],
                source_type="native",
                score=0.0,
                heading_path=heading_path,
                section_id=section_id,
            )
        ]

    monkeypatch.setattr(router.vector_search, "search", _search)
    monkeypatch.setattr(router.vector_search, "fetch_by_section", _fetch_by_section)
    monkeypatch.setattr(router, "bm25_heading_anchor", lambda *args, **kwargs: None)
    results = router.search_with_intent("doc-1", "list all significant events", top_k=3)
    assert len(results) == 1
    assert results[0].text_content == "Expanded content"
