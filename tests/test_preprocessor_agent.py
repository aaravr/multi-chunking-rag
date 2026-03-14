"""Tests for the Preprocessor Agent (MASTER_PROMPT §4.9).

Validates:
- 3-tier decision flow: skip → deterministic → learned → default
- Document type → strategy mapping (all known types)
- OutcomeStore learning loop: record + lookup
- Triage-based adjustments (DI ratio, image coverage)
- Message bus integration
- Edge cases: empty docs, unknown types, zero pages
"""

import pytest

from agents.contracts import (
    ChunkingOutcome,
    ChunkingStrategy,
    PreprocessorInput,
    PreprocessorResult,
    new_id,
)
from agents.message_bus import MessageBus
from agents.preprocessor_agent import (
    MIN_OUTCOMES_FOR_LEARNING,
    QUALITY_THRESHOLD,
    OutcomeStore,
    PreprocessorAgent,
    _DEFAULT_STRATEGY,
    _SKIP_STRATEGY,
    _STRATEGY_RULES,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def outcome_store():
    return OutcomeStore(use_db=False)


@pytest.fixture
def agent(outcome_store):
    bus = MessageBus()
    return PreprocessorAgent(bus=bus, gateway=None, outcome_store=outcome_store)


def _make_input(**kwargs) -> PreprocessorInput:
    defaults = {
        "doc_id": new_id(),
        "filename": "test.pdf",
        "page_count": 10,
    }
    defaults.update(kwargs)
    return PreprocessorInput(**defaults)


def _make_outcome(**kwargs) -> ChunkingOutcome:
    defaults = {
        "doc_id": new_id(),
        "strategy_name": "late_chunking",
        "document_type": "annual_report",
        "classification_label": "financial_report",
        "page_count": 50,
        "chunk_count": 200,
        "avg_chunk_tokens": 180.0,
        "table_chunk_ratio": 0.15,
        "heading_chunk_ratio": 0.08,
        "boilerplate_ratio": 0.05,
        "processing_time_ms": 5000.0,
        "quality_score": 0.85,
    }
    defaults.update(kwargs)
    return ChunkingOutcome(**defaults)


# ── Tier 0: Skip check ──────────────────────────────────────────────────

class TestSkipDecision:
    def test_zero_pages_skips_chunking(self, agent):
        inp = _make_input(page_count=0)
        result = agent.determine_strategy(inp)
        assert result.requires_chunking is False
        assert result.chunking_strategy.strategy_name == "skip"
        assert result.confidence == 1.0
        assert result.decision_method == "deterministic"

    def test_zero_text_length_skips_chunking(self, agent):
        inp = _make_input(
            page_count=5,
            triage_summary={"total_text_length": 0},
        )
        result = agent.determine_strategy(inp)
        assert result.requires_chunking is False
        assert result.chunking_strategy.strategy_name == "skip"

    def test_nonzero_pages_does_not_skip(self, agent):
        inp = _make_input(page_count=5)
        result = agent.determine_strategy(inp)
        assert result.requires_chunking is True


# ── Tier 1: Deterministic rules ─────────────────────────────────────────

class TestDeterministicStrategy:
    @pytest.mark.parametrize("doc_type,expected_strategy", [
        ("10-K", "sec_filing"),
        ("10-Q", "sec_filing"),
        ("20-F", "sec_filing"),
        ("annual_report", "financial_report"),
        ("pillar3_disclosure", "regulatory_section"),
        ("contract", "contract_clause"),
        ("loan_agreement", "contract_clause"),
        ("esg_report", "sustainability_report"),
    ])
    def test_known_document_types(self, agent, doc_type, expected_strategy):
        inp = _make_input(document_type=doc_type)
        result = agent.determine_strategy(inp)
        assert result.requires_chunking is True
        assert result.chunking_strategy.strategy_name == expected_strategy
        assert result.decision_method == "deterministic"
        assert result.confidence == 0.95

    @pytest.mark.parametrize("label,expected_strategy", [
        ("sec_filing", "sec_filing"),
        ("financial_report", "financial_report"),
        ("basel_regulatory", "regulatory_section"),
        ("legal_document", "contract_clause"),
        ("sustainability", "sustainability_report"),
    ])
    def test_classification_label_fallback(self, agent, label, expected_strategy):
        inp = _make_input(classification_label=label)
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.strategy_name == expected_strategy
        assert result.decision_method == "deterministic"

    def test_document_type_takes_priority_over_label(self, agent):
        inp = _make_input(
            document_type="10-K",
            classification_label="financial_report",
        )
        result = agent.determine_strategy(inp)
        # 10-K maps to sec_filing, not financial_report
        assert result.chunking_strategy.strategy_name == "sec_filing"

    def test_sec_filing_has_larger_overlap(self, agent):
        inp = _make_input(document_type="10-K")
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.macro_overlap_tokens == 512
        assert result.chunking_strategy.child_target_tokens == 384

    def test_regulatory_has_smaller_macro(self, agent):
        inp = _make_input(document_type="pillar3_disclosure")
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.macro_max_tokens == 4096
        assert result.chunking_strategy.table_extraction == "full_page"

    def test_contract_has_no_table_extraction(self, agent):
        inp = _make_input(document_type="contract")
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.table_extraction == "none"
        assert result.chunking_strategy.child_target_tokens == 192


# ── Tier 2: Learned outcomes ────────────────────────────────────────────

class TestLearnedStrategy:
    def test_insufficient_outcomes_falls_to_default(self, agent):
        # Only 1 outcome (need MIN_OUTCOMES_FOR_LEARNING)
        agent.record_outcome(_make_outcome(
            document_type="research_paper",
            classification_label="academic",
        ))
        inp = _make_input(
            document_type="research_paper",
            classification_label="academic",
        )
        result = agent.determine_strategy(inp)
        assert result.decision_method == "default"

    def test_sufficient_outcomes_uses_learned(self, agent):
        # Record enough outcomes for learning
        for _ in range(MIN_OUTCOMES_FOR_LEARNING):
            agent.record_outcome(_make_outcome(
                document_type="research_paper",
                classification_label="academic",
                strategy_name="custom_research",
                quality_score=0.9,
            ))
        inp = _make_input(
            document_type="research_paper",
            classification_label="academic",
        )
        result = agent.determine_strategy(inp)
        assert result.decision_method == "learned"
        assert result.confidence == 0.75
        assert len(result.learned_from_doc_ids) > 0

    def test_low_quality_outcomes_not_used(self, agent):
        for _ in range(MIN_OUTCOMES_FOR_LEARNING):
            agent.record_outcome(_make_outcome(
                document_type="research_paper",
                classification_label="academic",
                quality_score=0.1,  # Below QUALITY_THRESHOLD
            ))
        inp = _make_input(
            document_type="research_paper",
            classification_label="academic",
        )
        result = agent.determine_strategy(inp)
        assert result.decision_method == "default"

    def test_best_strategy_selected_by_quality(self, agent):
        # Record outcomes for two strategies, one better than the other
        for _ in range(MIN_OUTCOMES_FOR_LEARNING):
            agent.record_outcome(_make_outcome(
                document_type="memo",
                classification_label="internal",
                strategy_name="strategy_a",
                quality_score=0.6,
            ))
            agent.record_outcome(_make_outcome(
                document_type="memo",
                classification_label="internal",
                strategy_name="strategy_b",
                quality_score=0.9,
            ))
        inp = _make_input(
            document_type="memo",
            classification_label="internal",
        )
        result = agent.determine_strategy(inp)
        assert result.decision_method == "learned"
        # strategy_b has higher avg quality
        assert "0.9" in result.chunking_strategy.rationale or \
               result.chunking_strategy.strategy_name == "strategy_b"

    def test_deterministic_overrides_learned(self, agent):
        """Known doc types always use deterministic, even if outcomes exist."""
        for _ in range(MIN_OUTCOMES_FOR_LEARNING):
            agent.record_outcome(_make_outcome(
                document_type="10-K",
                classification_label="sec_filing",
                strategy_name="custom_strategy",
                quality_score=0.99,
            ))
        inp = _make_input(document_type="10-K")
        result = agent.determine_strategy(inp)
        # Deterministic tier fires first
        assert result.decision_method == "deterministic"


# ── Tier 3: Default fallback ────────────────────────────────────────────

class TestDefaultFallback:
    def test_unknown_type_gets_default(self, agent):
        inp = _make_input(
            document_type="unknown_type",
            classification_label="unknown_label",
        )
        result = agent.determine_strategy(inp)
        assert result.decision_method == "default"
        assert result.chunking_strategy.strategy_name == "late_chunking"
        assert result.confidence == 0.5

    def test_no_type_gets_default(self, agent):
        inp = _make_input()
        result = agent.determine_strategy(inp)
        assert result.decision_method == "default"
        assert result.chunking_strategy.macro_max_tokens == 8192


# ── Triage adjustments ──────────────────────────────────────────────────

class TestTriageAdjustments:
    def test_high_di_ratio_upgrades_table_extraction(self, agent):
        inp = _make_input(
            document_type="annual_report",
            triage_summary={"di_page_ratio": 0.7, "avg_image_coverage": 0.3},
        )
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.table_extraction == "full_page"
        assert any("DI ratio" in w for w in result.warnings)

    def test_low_di_ratio_keeps_span_extraction(self, agent):
        inp = _make_input(
            document_type="annual_report",
            triage_summary={"di_page_ratio": 0.1, "avg_image_coverage": 0.1},
        )
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.table_extraction == "span"

    def test_high_image_coverage_adds_warning(self, agent):
        inp = _make_input(
            triage_summary={"di_page_ratio": 0.0, "avg_image_coverage": 0.8},
        )
        result = agent.determine_strategy(inp)
        assert any("image coverage" in w for w in result.warnings)

    def test_no_triage_data_no_adjustment(self, agent):
        inp = _make_input(document_type="10-K")
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.macro_overlap_tokens == 512
        assert len(result.warnings) == 0


# ── OutcomeStore ─────────────────────────────────────────────────────────

class TestOutcomeStore:
    def test_record_and_lookup(self, outcome_store):
        o1 = _make_outcome(quality_score=0.8)
        o2 = _make_outcome(quality_score=0.9)
        outcome_store.record(o1)
        outcome_store.record(o2)
        result = outcome_store.lookup("annual_report", "financial_report")
        assert result is not None

    def test_lookup_empty_returns_none(self, outcome_store):
        result = outcome_store.lookup("nonexistent", "type")
        assert result is None

    def test_total_outcomes_count(self, outcome_store):
        assert outcome_store.total_outcomes == 0
        outcome_store.record(_make_outcome())
        assert outcome_store.total_outcomes == 1

    def test_get_doc_ids(self, outcome_store):
        o = _make_outcome()
        outcome_store.record(o)
        ids = outcome_store.get_doc_ids_for_type("annual_report", "financial_report")
        assert o.doc_id in ids

    def test_quality_below_threshold_returns_none(self, outcome_store):
        for _ in range(MIN_OUTCOMES_FOR_LEARNING):
            outcome_store.record(_make_outcome(quality_score=0.1))
        result = outcome_store.lookup(
            "annual_report", "financial_report",
            min_quality=QUALITY_THRESHOLD,
        )
        assert result is None


# ── Message bus integration ──────────────────────────────────────────────

class TestMessageBusIntegration:
    def test_handle_message(self, agent):
        from agents.contracts import AgentMessage
        msg = AgentMessage(
            message_id=new_id(),
            query_id=new_id(),
            from_agent="orchestrator",
            to_agent="preprocessor",
            message_type="preprocess_request",
            payload={
                "doc_id": new_id(),
                "filename": "annual_report_2024.pdf",
                "page_count": 100,
                "document_type": "annual_report",
                "classification_label": "financial_report",
            },
            timestamp="2026-01-01T00:00:00Z",
        )
        result = agent.handle_message(msg)
        assert isinstance(result, PreprocessorResult)
        assert result.requires_chunking is True
        assert result.chunking_strategy.strategy_name == "financial_report"


# ── Strategy rule coverage ───────────────────────────────────────────────

class TestStrategyRules:
    def test_all_rules_have_rationale(self):
        for key, strategy in _STRATEGY_RULES.items():
            assert strategy.rationale, f"Strategy for {key} has no rationale"

    def test_all_rules_have_valid_table_extraction(self):
        valid = {"span", "full_page", "none"}
        for key, strategy in _STRATEGY_RULES.items():
            assert strategy.table_extraction in valid, (
                f"Strategy for {key} has invalid table_extraction: "
                f"{strategy.table_extraction}"
            )

    def test_default_strategy_is_late_chunking(self):
        assert _DEFAULT_STRATEGY.strategy_name == "late_chunking"
        assert _DEFAULT_STRATEGY.macro_max_tokens == 8192

    def test_skip_strategy_has_zero_tokens(self):
        assert _SKIP_STRATEGY.macro_max_tokens == 0
        assert _SKIP_STRATEGY.child_target_tokens == 0
