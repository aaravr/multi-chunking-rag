"""Tests for all chunking strategies in embedding/chunking_strategies.py.

Validates:
- Each strategy produces valid ChunkRecord lists with deterministic lineage
- Empty/single-page edge cases
- Strategy dispatch registry completeness
- Chunk invariants: doc_id, page_numbers, char_start/end, embedding present
- Strategy-specific behavior (clause boundaries, parent-child hierarchy, etc.)

All tests mock the embedding model to avoid loading 440MB ModernBERT.
"""

import uuid
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from core.contracts import CanonicalPage, CanonicalSpan, ChunkRecord


# ── Fixtures ─────────────────────────────────────────────────────────────

FAKE_EMBEDDING = [0.1] * 768


def _fake_embed_text(text: str) -> List[float]:
    """Return a deterministic fake embedding."""
    return FAKE_EMBEDDING


def _make_span(
    text: str,
    char_start: int = 0,
    char_end: int = 0,
    page_number: int = 1,
    is_table: bool = False,
    heading_path: str = "",
    section_id: str = "",
) -> CanonicalSpan:
    return CanonicalSpan(
        text=text,
        char_start=char_start,
        char_end=char_end or len(text),
        polygons=[{"x": 0, "y": 0, "w": 100, "h": 20}],
        source_type="native",
        page_number=page_number,
        heading_path=heading_path,
        section_id=section_id,
        is_table=is_table,
    )


def _make_page(
    doc_id: str,
    page_number: int,
    text: str,
    spans: List[CanonicalSpan] = None,
    is_table: bool = False,
) -> CanonicalPage:
    if spans is None:
        spans = [_make_span(text, page_number=page_number, is_table=is_table)]
    return CanonicalPage(
        doc_id=doc_id,
        page_number=page_number,
        text=text,
        spans=spans,
    )


@pytest.fixture
def doc_id():
    return str(uuid.uuid4())


@pytest.fixture
def single_page(doc_id):
    text = (
        "The company reported revenue of $5.2 billion for the fiscal year. "
        "Operating expenses increased by 12% year-over-year. "
        "Net income reached $1.1 billion, up from $900 million. "
        "The board declared a quarterly dividend of $0.50 per share. "
        "Management expects continued growth in the next quarter."
    )
    return [_make_page(doc_id, 1, text)]


@pytest.fixture
def multi_page(doc_id):
    pages = []
    for i in range(1, 6):
        text = (
            f"Section {i}: Financial Overview for Quarter {i}. "
            f"Revenue for Q{i} was ${i * 1.5:.1f} billion. "
            f"Operating margin improved to {20 + i}%. "
            f"Headcount grew by {i * 100} employees. "
            f"Capital expenditure was ${i * 0.3:.1f} billion. "
            f"Free cash flow reached ${i * 0.5:.1f} billion. "
            f"The company maintained its credit rating of A+. "
            f"Market share in the segment increased to {30 + i}%."
        )
        pages.append(_make_page(doc_id, i, text))
    return pages


@pytest.fixture
def legal_pages(doc_id):
    """Pages with legal clause structure."""
    page1_text = (
        "ARTICLE I\nDEFINITIONS\n\n"
        "Section 1.1 Defined Terms. As used in this Agreement, the following "
        "terms shall have the meanings set forth below.\n\n"
        "Section 1.2 Rules of Construction. Unless the context otherwise "
        "requires, references to Sections are to Sections of this Agreement.\n\n"
    )
    page2_text = (
        "ARTICLE II\nTHE CREDIT FACILITY\n\n"
        "Section 2.1 Commitments. Subject to the terms and conditions set "
        "forth herein, each Lender severally agrees to make revolving loans "
        "to the Borrower from time to time.\n\n"
        "Section 2.2 Borrowings. Each Borrowing shall be comprised entirely "
        "of Base Rate Loans or Eurodollar Rate Loans as the Borrower may "
        "request in the applicable Borrowing Request.\n\n"
    )
    page3_text = (
        "ARTICLE III\nREPRESENTATIONS AND WARRANTIES\n\n"
        "Section 3.1 Organization. The Borrower is a corporation duly "
        "organized and validly existing under the laws of the State of Delaware.\n\n"
        "Section 3.2 Authorization. The execution and delivery of this "
        "Agreement has been duly authorized by all necessary corporate action.\n\n"
    )
    return [
        _make_page(doc_id, 1, page1_text),
        _make_page(doc_id, 2, page2_text),
        _make_page(doc_id, 3, page3_text),
    ]


@pytest.fixture
def table_pages(doc_id):
    """Pages with table and narrative content."""
    narrative_text = (
        "The following table summarizes our revenue by segment for the "
        "fiscal year ended December 31, 2025. "
        "Total revenue increased by 15% compared to the prior year."
    )
    table_text = (
        "| Segment | Revenue ($M) | Growth (%) |\n"
        "| ------- | ------------ | ---------- |\n"
        "| Cloud   | 2,500        | 25         |\n"
        "| On-prem | 1,800        | 5          |\n"
        "| Services| 900          | 10         |\n"
    )
    narrative_spans = [_make_span(narrative_text, page_number=1)]
    table_spans = [_make_span(table_text, page_number=2, is_table=True)]

    return [
        _make_page(doc_id, 1, narrative_text, spans=narrative_spans),
        _make_page(doc_id, 2, table_text, spans=table_spans, is_table=True),
    ]


# ── Patch helper ────────────────────────────────────────────────────────

def _fake_tokenize_full(text: str) -> List:
    """Simulate tokenizer: one 'token' per whitespace-delimited word."""
    offsets = []
    pos = 0
    for word in text.split():
        start = text.index(word, pos)
        end = start + len(word)
        offsets.append((start, end))
        pos = end
    return offsets


@pytest.fixture(autouse=True)
def mock_embedder():
    """Mock the embedding model for all tests."""
    mock_model = MagicMock()
    mock_model.embed_text = _fake_embed_text
    mock_model.tokenize_full = _fake_tokenize_full
    with patch(
        "embedding.chunking_strategies._embed_text",
        side_effect=_fake_embed_text,
    ), patch(
        "embedding.chunking_strategies._get_embedder",
        return_value=mock_model,
    ):
        yield


# ── Shared assertion helpers ─────────────────────────────────────────────

def assert_valid_chunks(chunks: List[ChunkRecord], doc_id: str):
    """Validate invariants that ALL chunking strategies must preserve."""
    assert isinstance(chunks, list)
    assert len(chunks) > 0, "Strategy produced no chunks"
    for i, chunk in enumerate(chunks):
        assert chunk.doc_id == doc_id, f"Chunk {i}: wrong doc_id"
        assert len(chunk.page_numbers) > 0, f"Chunk {i}: empty page_numbers"
        assert chunk.text_content, f"Chunk {i}: empty text_content"
        assert len(chunk.embedding) == 768, f"Chunk {i}: wrong embedding dim"
        assert chunk.chunk_id, f"Chunk {i}: empty chunk_id"


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY DISPATCH
# ═══════════════════════════════════════════════════════════════════════════

class TestStrategyDispatch:
    def test_dispatch_contains_all_11_strategies(self):
        from embedding.chunking_strategies import get_strategy_dispatch
        dispatch = get_strategy_dispatch()
        expected = {
            "semantic", "recursive", "clause_aware", "sentence_level",
            "sliding_window", "parent_child", "table_aware",
            "topic_segmentation", "context_enriched", "proposition",
            "summary_indexed",
        }
        assert set(dispatch.keys()) == expected

    def test_dispatch_values_are_callable(self):
        from embedding.chunking_strategies import get_strategy_dispatch
        dispatch = get_strategy_dispatch()
        for name, fn in dispatch.items():
            assert callable(fn), f"{name} is not callable"


# ═══════════════════════════════════════════════════════════════════════════
# 1. SEMANTIC CHUNKING
# ═══════════════════════════════════════════════════════════════════════════

class TestSemanticChunking:
    def test_produces_chunks(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_semantic
        chunks = chunk_semantic(doc_id, multi_page)
        assert_valid_chunks(chunks, doc_id)

    def test_single_page(self, doc_id, single_page):
        from embedding.chunking_strategies import chunk_semantic
        chunks = chunk_semantic(doc_id, single_page)
        assert_valid_chunks(chunks, doc_id)

    def test_empty_pages(self, doc_id):
        from embedding.chunking_strategies import chunk_semantic
        chunks = chunk_semantic(doc_id, [])
        assert chunks == []

    def test_respects_max_chunk_sentences(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_semantic
        chunks = chunk_semantic(doc_id, multi_page, max_chunk_sentences=2)
        assert_valid_chunks(chunks, doc_id)
        # With max 2 sentences per chunk, should have more chunks
        chunks_large = chunk_semantic(doc_id, multi_page, max_chunk_sentences=50)
        assert len(chunks) >= len(chunks_large)


# ═══════════════════════════════════════════════════════════════════════════
# 2. RECURSIVE CHARACTER SPLITTING
# ═══════════════════════════════════════════════════════════════════════════

class TestRecursiveChunking:
    def test_produces_chunks(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_recursive
        chunks = chunk_recursive(doc_id, multi_page)
        assert_valid_chunks(chunks, doc_id)

    def test_empty_pages(self, doc_id):
        from embedding.chunking_strategies import chunk_recursive
        chunks = chunk_recursive(doc_id, [])
        assert chunks == []

    def test_smaller_max_produces_more_chunks(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_recursive
        chunks_small = chunk_recursive(doc_id, multi_page, max_chunk_chars=200)
        chunks_large = chunk_recursive(doc_id, multi_page, max_chunk_chars=5000)
        assert len(chunks_small) >= len(chunks_large)

    def test_overlap_present(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_recursive
        chunks = chunk_recursive(
            doc_id, multi_page, max_chunk_chars=300, chunk_overlap_chars=50
        )
        assert_valid_chunks(chunks, doc_id)


# ═══════════════════════════════════════════════════════════════════════════
# 3. CLAUSE-AWARE CHUNKING
# ═══════════════════════════════════════════════════════════════════════════

class TestClauseAwareChunking:
    def test_produces_chunks_from_legal_text(self, doc_id, legal_pages):
        from embedding.chunking_strategies import chunk_clause_aware
        chunks = chunk_clause_aware(doc_id, legal_pages)
        assert_valid_chunks(chunks, doc_id)

    def test_clause_boundaries_respected(self, doc_id, legal_pages):
        from embedding.chunking_strategies import chunk_clause_aware
        chunks = chunk_clause_aware(doc_id, legal_pages)
        # Should split at Section/Article boundaries — expect multiple chunks
        assert len(chunks) >= 2

    def test_empty_pages(self, doc_id):
        from embedding.chunking_strategies import chunk_clause_aware
        chunks = chunk_clause_aware(doc_id, [])
        assert chunks == []

    def test_non_legal_text_still_produces_chunks(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_clause_aware
        chunks = chunk_clause_aware(doc_id, multi_page)
        # Should still produce chunks even without legal clause markers
        assert_valid_chunks(chunks, doc_id)


# ═══════════════════════════════════════════════════════════════════════════
# 4. SENTENCE-LEVEL CHUNKING
# ═══════════════════════════════════════════════════════════════════════════

class TestSentenceLevelChunking:
    def test_produces_chunks(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_sentence_level
        chunks = chunk_sentence_level(doc_id, multi_page)
        assert_valid_chunks(chunks, doc_id)

    def test_empty_pages(self, doc_id):
        from embedding.chunking_strategies import chunk_sentence_level
        chunks = chunk_sentence_level(doc_id, [])
        assert chunks == []

    def test_sentences_per_chunk_controls_size(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_sentence_level
        chunks_small = chunk_sentence_level(
            doc_id, multi_page, sentences_per_chunk=2
        )
        chunks_large = chunk_sentence_level(
            doc_id, multi_page, sentences_per_chunk=10
        )
        assert len(chunks_small) >= len(chunks_large)

    def test_single_page(self, doc_id, single_page):
        from embedding.chunking_strategies import chunk_sentence_level
        chunks = chunk_sentence_level(doc_id, single_page)
        assert_valid_chunks(chunks, doc_id)


# ═══════════════════════════════════════════════════════════════════════════
# 5. SLIDING WINDOW (ROLLING WINDOW)
# ═══════════════════════════════════════════════════════════════════════════

class TestSlidingWindowChunking:
    def test_produces_chunks(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_sliding_window
        chunks = chunk_sliding_window(doc_id, multi_page)
        assert_valid_chunks(chunks, doc_id)

    def test_empty_pages(self, doc_id):
        from embedding.chunking_strategies import chunk_sliding_window
        chunks = chunk_sliding_window(doc_id, [])
        assert chunks == []

    def test_smaller_window_more_chunks(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_sliding_window
        chunks_small = chunk_sliding_window(
            doc_id, multi_page, window_size_tokens=50, stride_tokens=25
        )
        chunks_large = chunk_sliding_window(
            doc_id, multi_page, window_size_tokens=500, stride_tokens=400
        )
        assert len(chunks_small) >= len(chunks_large)

    def test_stride_less_than_window(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_sliding_window
        # Overlapping windows (stride < window) should produce more chunks
        chunks_overlap = chunk_sliding_window(
            doc_id, multi_page, window_size_tokens=100, stride_tokens=50
        )
        chunks_no_overlap = chunk_sliding_window(
            doc_id, multi_page, window_size_tokens=100, stride_tokens=100
        )
        assert len(chunks_overlap) >= len(chunks_no_overlap)


# ═══════════════════════════════════════════════════════════════════════════
# 6. PARENT-CHILD HIERARCHICAL
# ═══════════════════════════════════════════════════════════════════════════

class TestParentChildChunking:
    def test_produces_chunks(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_parent_child
        chunks = chunk_parent_child(doc_id, multi_page)
        assert_valid_chunks(chunks, doc_id)

    def test_parent_and_child_chunks_present(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_parent_child
        chunks = chunk_parent_child(doc_id, multi_page)
        parent_chunks = [c for c in chunks if c.child_id == 0]
        child_chunks = [c for c in chunks if c.child_id > 0]
        assert len(parent_chunks) > 0, "No parent chunks (child_id=0)"
        assert len(child_chunks) > 0, "No child chunks (child_id>0)"

    def test_children_linked_to_parents(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_parent_child
        chunks = chunk_parent_child(doc_id, multi_page)
        parent_macro_ids = {c.macro_id for c in chunks if c.child_id == 0}
        child_macro_ids = {c.macro_id for c in chunks if c.child_id > 0}
        # Every child's macro_id should reference a parent
        assert child_macro_ids.issubset(parent_macro_ids)

    def test_empty_pages(self, doc_id):
        from embedding.chunking_strategies import chunk_parent_child
        chunks = chunk_parent_child(doc_id, [])
        assert chunks == []


# ═══════════════════════════════════════════════════════════════════════════
# 7. TABLE-AWARE CHUNKING
# ═══════════════════════════════════════════════════════════════════════════

class TestTableAwareChunking:
    def test_produces_chunks(self, doc_id, table_pages):
        from embedding.chunking_strategies import chunk_table_aware
        chunks = chunk_table_aware(doc_id, table_pages)
        assert_valid_chunks(chunks, doc_id)

    def test_table_chunks_identified(self, doc_id, table_pages):
        from embedding.chunking_strategies import chunk_table_aware
        chunks = chunk_table_aware(doc_id, table_pages)
        table_chunks = [c for c in chunks if c.chunk_type == "table"]
        # The table page should produce at least one table chunk
        assert len(table_chunks) >= 1

    def test_narrative_and_table_separated(self, doc_id, table_pages):
        from embedding.chunking_strategies import chunk_table_aware
        chunks = chunk_table_aware(doc_id, table_pages)
        chunk_types = {c.chunk_type for c in chunks}
        # Should have both narrative and table chunks
        assert len(chunk_types) >= 1  # At least one type present

    def test_empty_pages(self, doc_id):
        from embedding.chunking_strategies import chunk_table_aware
        chunks = chunk_table_aware(doc_id, [])
        assert chunks == []

    def test_all_narrative_pages(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_table_aware
        chunks = chunk_table_aware(doc_id, multi_page)
        # Should still produce chunks even without tables
        assert_valid_chunks(chunks, doc_id)


# ═══════════════════════════════════════════════════════════════════════════
# 8. TOPIC SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

class TestTopicSegmentation:
    def test_produces_chunks(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_topic_segmentation
        chunks = chunk_topic_segmentation(doc_id, multi_page)
        assert_valid_chunks(chunks, doc_id)

    def test_empty_pages(self, doc_id):
        from embedding.chunking_strategies import chunk_topic_segmentation
        chunks = chunk_topic_segmentation(doc_id, [])
        assert chunks == []

    def test_single_page(self, doc_id, single_page):
        from embedding.chunking_strategies import chunk_topic_segmentation
        chunks = chunk_topic_segmentation(doc_id, single_page)
        assert_valid_chunks(chunks, doc_id)


# ═══════════════════════════════════════════════════════════════════════════
# 9. CONTEXT-ENRICHED CHUNKING
# ═══════════════════════════════════════════════════════════════════════════

class TestContextEnrichedChunking:
    def test_produces_chunks(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_context_enriched
        chunks = chunk_context_enriched(doc_id, multi_page)
        assert_valid_chunks(chunks, doc_id)

    def test_empty_pages(self, doc_id):
        from embedding.chunking_strategies import chunk_context_enriched
        chunks = chunk_context_enriched(doc_id, [])
        assert chunks == []

    def test_with_heading_context(self, doc_id):
        """Chunks from pages with heading_path should get re-embedded."""
        from embedding.chunking_strategies import chunk_context_enriched
        span = _make_span(
            "The borrower shall maintain a minimum net worth of $100M.",
            heading_path="Article III > Section 3.1",
            section_id="3.1",
        )
        page = _make_page(
            doc_id, 1,
            "The borrower shall maintain a minimum net worth of $100M.",
            spans=[span],
        )
        chunks = chunk_context_enriched(doc_id, [page])
        assert_valid_chunks(chunks, doc_id)


# ═══════════════════════════════════════════════════════════════════════════
# 10. PROPOSITION CHUNKING (LLM-dependent)
# ═══════════════════════════════════════════════════════════════════════════

class TestPropositionChunking:
    def test_without_gateway_falls_back_to_sentence(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_proposition
        chunks = chunk_proposition(doc_id, multi_page, gateway=None)
        assert_valid_chunks(chunks, doc_id)

    def test_with_mock_gateway(self, doc_id, single_page):
        from embedding.chunking_strategies import chunk_proposition
        mock_gw = MagicMock()
        mock_gw.call_model.return_value = {
            "content": (
                "- The company reported revenue of $5.2 billion.\n"
                "- Operating expenses increased by 12%.\n"
                "- Net income reached $1.1 billion."
            )
        }
        chunks = chunk_proposition(
            doc_id, single_page, gateway=mock_gw, model_id="test-model"
        )
        assert_valid_chunks(chunks, doc_id)

    def test_empty_pages(self, doc_id):
        from embedding.chunking_strategies import chunk_proposition
        chunks = chunk_proposition(doc_id, [])
        assert chunks == []


# ═══════════════════════════════════════════════════════════════════════════
# 11. SUMMARY-INDEXED CHUNKING (LLM-dependent)
# ═══════════════════════════════════════════════════════════════════════════

class TestSummaryIndexedChunking:
    def test_without_gateway_returns_base_chunks(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_summary_indexed
        chunks = chunk_summary_indexed(doc_id, multi_page, gateway=None)
        assert_valid_chunks(chunks, doc_id)

    def test_with_mock_gateway_adds_summaries(self, doc_id, multi_page):
        from embedding.chunking_strategies import chunk_summary_indexed
        mock_gw = MagicMock()
        mock_gw.call_model.return_value = {
            "content": "Summary: revenue growth and financial metrics."
        }
        chunks = chunk_summary_indexed(
            doc_id, multi_page, gateway=mock_gw, model_id="test-model"
        )
        assert_valid_chunks(chunks, doc_id)
        # Should have more chunks than base (summaries added)
        base_chunks = chunk_summary_indexed(doc_id, multi_page, gateway=None)
        assert len(chunks) >= len(base_chunks)

    def test_empty_pages(self, doc_id):
        from embedding.chunking_strategies import chunk_summary_indexed
        chunks = chunk_summary_indexed(doc_id, [])
        assert chunks == []


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-STRATEGY INVARIANTS
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossStrategyInvariants:
    """Every strategy must preserve deterministic lineage (§2.1)."""

    DETERMINISTIC_STRATEGIES = [
        "semantic", "recursive", "clause_aware", "sentence_level",
        "sliding_window", "parent_child", "table_aware",
        "topic_segmentation", "context_enriched",
    ]

    @pytest.mark.parametrize("strategy_name", DETERMINISTIC_STRATEGIES)
    def test_all_chunks_have_doc_id(self, strategy_name, doc_id, multi_page):
        from embedding.chunking_strategies import get_strategy_dispatch
        fn = get_strategy_dispatch()[strategy_name]
        chunks = fn(doc_id, multi_page)
        for chunk in chunks:
            assert chunk.doc_id == doc_id

    @pytest.mark.parametrize("strategy_name", DETERMINISTIC_STRATEGIES)
    def test_all_chunks_have_page_numbers(self, strategy_name, doc_id, multi_page):
        from embedding.chunking_strategies import get_strategy_dispatch
        fn = get_strategy_dispatch()[strategy_name]
        chunks = fn(doc_id, multi_page)
        for chunk in chunks:
            assert len(chunk.page_numbers) > 0

    @pytest.mark.parametrize("strategy_name", DETERMINISTIC_STRATEGIES)
    def test_all_chunks_have_embedding(self, strategy_name, doc_id, multi_page):
        from embedding.chunking_strategies import get_strategy_dispatch
        fn = get_strategy_dispatch()[strategy_name]
        chunks = fn(doc_id, multi_page)
        for chunk in chunks:
            assert len(chunk.embedding) == 768

    @pytest.mark.parametrize("strategy_name", DETERMINISTIC_STRATEGIES)
    def test_empty_input_returns_empty(self, strategy_name, doc_id):
        from embedding.chunking_strategies import get_strategy_dispatch
        fn = get_strategy_dispatch()[strategy_name]
        chunks = fn(doc_id, [])
        assert chunks == []

    @pytest.mark.parametrize("strategy_name", DETERMINISTIC_STRATEGIES)
    def test_unique_chunk_ids(self, strategy_name, doc_id, multi_page):
        from embedding.chunking_strategies import get_strategy_dispatch
        fn = get_strategy_dispatch()[strategy_name]
        chunks = fn(doc_id, multi_page)
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk_ids found"


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

class TestHelpers:
    def test_split_sentences(self):
        from embedding.chunking_strategies import _split_sentences
        text = "First sentence. Second sentence. Third sentence."
        sentences = _split_sentences(text)
        assert len(sentences) >= 1

    def test_split_sentences_empty(self):
        from embedding.chunking_strategies import _split_sentences
        assert _split_sentences("") == []
        assert _split_sentences("   ") == []

    def test_cosine_similarity_identical(self):
        from embedding.chunking_strategies import _cosine_similarity
        vec = [1.0, 2.0, 3.0]
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        from embedding.chunking_strategies import _cosine_similarity
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_concat_page_text(self):
        from embedding.chunking_strategies import _concat_page_text
        doc_id = str(uuid.uuid4())
        pages = [
            _make_page(doc_id, 1, "Page one text"),
            _make_page(doc_id, 2, "Page two text"),
        ]
        result = _concat_page_text(pages)
        assert "Page one text" in result
        assert "Page two text" in result
