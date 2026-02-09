from core.contracts import RetrievedChunk
from retrieval import router


def _chunk(text: str, chunk_type: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
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


def test_table_filtering_excludes_tables_for_items_of_note():
    query = "List the items of note affecting net income"
    chunks = [
        _chunk("[TABLE] Items of note table", "table"),
        _chunk("Items of note include FDIC special assessment.", "narrative"),
    ]
    filtered = router._apply_table_filter(query, chunks)
    assert len(filtered) == 1
    assert filtered[0].chunk_type == "narrative"


def test_table_filtering_allows_tables_for_explicit_note():
    query = "Note 12 derivative instruments"
    chunks = [
        _chunk("[TABLE] Note 12 Derivative instruments", "table"),
        _chunk("Derivative instruments narrative", "narrative"),
    ]
    filtered = router._apply_table_filter(query, chunks)
    assert len(filtered) == 2
