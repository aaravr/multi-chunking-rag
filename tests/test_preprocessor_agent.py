"""Tests for the Preprocessor Agent (MASTER_PROMPT §4.9).

Validates:
- 3-tier decision flow: skip → deterministic → learned → heuristic
- Document type → strategy mapping for ALL investment banking document types
- Processing levels: skip, metadata_only, single_chunk, page_level, late_chunking
- OutcomeStore learning loop: record + lookup
- Triage-based adjustments (DI ratio, image coverage)
- Complexity assessment heuristic
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
    PROCESSING_LEVELS,
    QUALITY_THRESHOLD,
    OutcomeStore,
    PreprocessorAgent,
    _DEFAULT_STRATEGY,
    _METADATA_ONLY_STRATEGY,
    _PAGE_LEVEL_STRATEGY,
    _SINGLE_CHUNK_STRATEGY,
    _SINGLE_CHUNK_MAX_PAGES,
    _PAGE_LEVEL_MAX_PAGES,
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
    def test_zero_pages_skips_processing(self, agent):
        inp = _make_input(page_count=0)
        result = agent.determine_strategy(inp)
        assert result.requires_chunking is False
        assert result.chunking_strategy.strategy_name == "skip"
        assert result.chunking_strategy.processing_level == "skip"
        assert result.confidence == 1.0
        assert result.decision_method == "deterministic"

    def test_zero_text_length_skips_processing(self, agent):
        inp = _make_input(
            page_count=5,
            triage_summary={"total_text_length": 0},
        )
        result = agent.determine_strategy(inp)
        assert result.requires_chunking is False
        assert result.chunking_strategy.strategy_name == "skip"
        assert result.chunking_strategy.processing_level == "skip"

    def test_nonzero_pages_does_not_skip(self, agent):
        inp = _make_input(page_count=5)
        result = agent.determine_strategy(inp)
        assert result.requires_chunking is True


# ── Tier 1: Deterministic rules — Identity & KYC (metadata_only) ─────

class TestIdentityDocuments:
    """Identity/KYC documents should use metadata_only processing."""

    @pytest.mark.parametrize("doc_type", [
        "passport",
        "driving_licence",
        "driving_license",
        "national_id",
        "visa",
        "identity_document",
        "kyc_form",
        "beneficial_ownership",
        "sanctions_screening",
        "pep_declaration",
    ])
    def test_identity_docs_are_metadata_only(self, agent, doc_type):
        inp = _make_input(document_type=doc_type, page_count=1)
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.processing_level == "metadata_only"
        assert result.requires_chunking is False
        assert result.decision_method == "deterministic"
        assert result.confidence == 0.95


class TestProofDocuments:
    """Proof-of-existence and simple certificates."""

    @pytest.mark.parametrize("doc_type,expected_level", [
        ("utility_bill", "metadata_only"),
        ("proof_of_address", "metadata_only"),
        ("certificate_of_incorporation", "single_chunk"),
        ("certificate_of_good_standing", "single_chunk"),
        ("certificate_of_incumbency", "single_chunk"),
        ("pay_slip", "metadata_only"),
    ])
    def test_proof_docs(self, agent, doc_type, expected_level):
        inp = _make_input(document_type=doc_type, page_count=1)
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.processing_level == expected_level
        assert result.decision_method == "deterministic"


# ── Tier 1: Corporate governance ─────────────────────────────────────

class TestCorporateGovernance:
    @pytest.mark.parametrize("doc_type,expected_level", [
        ("board_resolution", "single_chunk"),
        ("power_of_attorney", "single_chunk"),
        ("signing_authority", "single_chunk"),
        ("corporate_structure_chart", "metadata_only"),
        ("shareholder_register", "page_level"),
        ("articles_of_association", "late_chunking"),
    ])
    def test_corporate_governance_docs(self, agent, doc_type, expected_level):
        inp = _make_input(document_type=doc_type, page_count=5)
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.processing_level == expected_level
        assert result.decision_method == "deterministic"


# ── Tier 1: Trading agreements (late_chunking) ───────────────────────

class TestTradingAgreements:
    @pytest.mark.parametrize("doc_type", [
        "isda_master_agreement",
        "isda_schedule",
        "credit_support_annex",
        "gmra",
        "gmsla",
        "prime_brokerage_agreement",
        "futures_agreement",
        "trading_agreement",
    ])
    def test_trading_agreements_use_late_chunking(self, agent, doc_type):
        inp = _make_input(document_type=doc_type, page_count=50)
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.processing_level == "late_chunking"
        assert result.chunking_strategy.strategy_name == "clause_aware"
        assert result.chunking_strategy.macro_max_tokens == 4096
        assert result.decision_method == "deterministic"


# ── Tier 1: Credit & lending ────────────────────────────────────────

class TestCreditDocuments:
    @pytest.mark.parametrize("doc_type,expected_level", [
        ("credit_agreement", "late_chunking"),
        ("facility_agreement", "late_chunking"),
        ("loan_agreement", "late_chunking"),
        ("security_agreement", "late_chunking"),
        ("guarantee", "late_chunking"),
        ("intercreditor_agreement", "late_chunking"),
        ("term_sheet", "single_chunk"),
        ("commitment_letter", "single_chunk"),
    ])
    def test_credit_docs(self, agent, doc_type, expected_level):
        inp = _make_input(document_type=doc_type, page_count=50)
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.processing_level == expected_level
        assert result.decision_method == "deterministic"


# ── Tier 1: Capital markets ─────────────────────────────────────────

class TestCapitalMarkets:
    @pytest.mark.parametrize("doc_type,expected_level", [
        ("prospectus", "late_chunking"),
        ("offering_memorandum", "late_chunking"),
        ("pricing_supplement", "page_level"),
        ("base_indenture", "late_chunking"),
        ("supplemental_indenture", "late_chunking"),
    ])
    def test_capital_markets_docs(self, agent, doc_type, expected_level):
        inp = _make_input(document_type=doc_type, page_count=100)
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.processing_level == expected_level
        assert result.decision_method == "deterministic"


# ── Tier 1: Fund documents ──────────────────────────────────────────

class TestFundDocuments:
    @pytest.mark.parametrize("doc_type,expected_level", [
        ("fund_prospectus", "late_chunking"),
        ("private_placement_memorandum", "late_chunking"),
        ("subscription_agreement", "late_chunking"),
        ("side_letter", "page_level"),
        ("limited_partnership_agreement", "late_chunking"),
    ])
    def test_fund_docs(self, agent, doc_type, expected_level):
        inp = _make_input(document_type=doc_type, page_count=50)
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.processing_level == expected_level
        assert result.decision_method == "deterministic"


# ── Tier 1: Existing document types (backward compatibility) ─────────

class TestExistingDocumentTypes:
    """Ensure backward compatibility with previously supported types."""

    @pytest.mark.parametrize("doc_type,expected_strategy", [
        ("10-K", "parent_child"),
        ("10-Q", "parent_child"),
        ("20-F", "parent_child"),
        ("annual_report", "semantic"),
        ("pillar3_disclosure", "table_aware"),
        ("contract", "clause_aware"),
        ("loan_agreement", "clause_aware"),
        ("esg_report", "semantic"),
    ])
    def test_known_document_types(self, agent, doc_type, expected_strategy):
        inp = _make_input(document_type=doc_type)
        result = agent.determine_strategy(inp)
        assert result.requires_chunking is True
        assert result.chunking_strategy.strategy_name == expected_strategy
        assert result.chunking_strategy.processing_level == "late_chunking"
        assert result.decision_method == "deterministic"
        assert result.confidence == 0.95

    @pytest.mark.parametrize("label,expected_strategy", [
        ("sec_filing", "parent_child"),
        ("financial_report", "semantic"),
        ("basel_regulatory", "table_aware"),
        ("legal_document", "clause_aware"),
        ("sustainability", "semantic"),
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
        # 10-K maps to parent_child, not semantic
        assert result.chunking_strategy.strategy_name == "parent_child"

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


# ── Tier 1: Compliance & correspondence ──────────────────────────────

class TestComplianceAndCorrespondence:
    @pytest.mark.parametrize("doc_type,expected_level", [
        ("officer_certificate", "single_chunk"),
        ("compliance_certificate", "single_chunk"),
        ("legal_opinion", "late_chunking"),
        ("auditor_report", "late_chunking"),
        ("notice", "single_chunk"),
        ("waiver_letter", "single_chunk"),
        ("consent_letter", "single_chunk"),
        ("nda", "single_chunk"),
        ("engagement_letter", "single_chunk"),
        ("fee_letter", "single_chunk"),
        ("amendment", "late_chunking"),
    ])
    def test_compliance_correspondence_docs(self, agent, doc_type, expected_level):
        inp = _make_input(document_type=doc_type, page_count=5)
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.processing_level == expected_level
        assert result.decision_method == "deterministic"


# ── Tier 1: Financial statements & insurance ─────────────────────────

class TestFinancialAndInsurance:
    @pytest.mark.parametrize("doc_type,expected_level", [
        ("bank_statement", "page_level"),
        ("tax_return", "page_level"),
        ("insurance_certificate", "single_chunk"),
        ("insurance_policy", "late_chunking"),
        ("valuation_report", "late_chunking"),
        ("fairness_opinion", "late_chunking"),
    ])
    def test_financial_insurance_docs(self, agent, doc_type, expected_level):
        inp = _make_input(document_type=doc_type, page_count=10)
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.processing_level == expected_level
        assert result.decision_method == "deterministic"


# ── Tier 1: requires_chunking derived from processing_level ─────────

class TestRequiresChunking:
    """metadata_only and skip should set requires_chunking=False."""

    def test_metadata_only_does_not_require_chunking(self, agent):
        inp = _make_input(document_type="passport", page_count=1)
        result = agent.determine_strategy(inp)
        assert result.requires_chunking is False

    def test_single_chunk_requires_chunking(self, agent):
        inp = _make_input(document_type="board_resolution", page_count=2)
        result = agent.determine_strategy(inp)
        assert result.requires_chunking is True

    def test_page_level_requires_chunking(self, agent):
        inp = _make_input(document_type="bank_statement", page_count=10)
        result = agent.determine_strategy(inp)
        assert result.requires_chunking is True

    def test_late_chunking_requires_chunking(self, agent):
        inp = _make_input(document_type="10-K", page_count=100)
        result = agent.determine_strategy(inp)
        assert result.requires_chunking is True


# ── Tier 2: Learned outcomes ────────────────────────────────────────────

class TestLearnedStrategy:
    def test_insufficient_outcomes_falls_to_heuristic(self, agent):
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
        assert result.decision_method == "heuristic"

    def test_sufficient_outcomes_uses_learned(self, agent):
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
        assert result.decision_method == "heuristic"

    def test_best_strategy_selected_by_quality(self, agent):
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


# ── Tier 3: Heuristic fallback ────────────────────────────────────────

class TestHeuristicFallback:
    def test_unknown_type_uses_heuristic(self, agent):
        inp = _make_input(
            document_type="unknown_type",
            classification_label="unknown_label",
        )
        result = agent.determine_strategy(inp)
        assert result.decision_method == "heuristic"
        assert result.confidence == 0.5

    def test_no_type_uses_heuristic(self, agent):
        inp = _make_input()
        result = agent.determine_strategy(inp)
        assert result.decision_method == "heuristic"

    def test_single_page_minimal_text_metadata_only(self, agent):
        inp = _make_input(
            page_count=1,
            triage_summary={"avg_text_length": 100},
        )
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.processing_level == "metadata_only"
        assert result.requires_chunking is False

    def test_short_doc_single_chunk(self, agent):
        inp = _make_input(page_count=2)
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.processing_level == "single_chunk"
        assert result.requires_chunking is True

    def test_three_page_doc_single_chunk(self, agent):
        inp = _make_input(page_count=_SINGLE_CHUNK_MAX_PAGES)
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.processing_level == "single_chunk"

    def test_moderate_doc_page_level(self, agent):
        inp = _make_input(page_count=10)
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.processing_level == "page_level"
        assert result.requires_chunking is True

    def test_fifteen_page_doc_page_level(self, agent):
        inp = _make_input(page_count=_PAGE_LEVEL_MAX_PAGES)
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.processing_level == "page_level"

    def test_long_doc_late_chunking(self, agent):
        inp = _make_input(page_count=50)
        result = agent.determine_strategy(inp)
        assert result.chunking_strategy.processing_level == "late_chunking"
        assert result.chunking_strategy.macro_max_tokens == 8192
        assert result.requires_chunking is True


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

    def test_di_ratio_does_not_upgrade_none_extraction(self, agent):
        """Documents with table_extraction=none should not be upgraded."""
        inp = _make_input(
            document_type="contract",
            triage_summary={"di_page_ratio": 0.7, "avg_image_coverage": 0.3},
        )
        result = agent.determine_strategy(inp)
        # contract has table_extraction="none", DI adjustment only applies
        # when table_extraction is "span"
        assert result.chunking_strategy.table_extraction == "none"


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
        assert result.chunking_strategy.strategy_name == "semantic"
        assert result.chunking_strategy.processing_level == "late_chunking"

    def test_handle_message_identity_doc(self, agent):
        from agents.contracts import AgentMessage
        msg = AgentMessage(
            message_id=new_id(),
            query_id=new_id(),
            from_agent="orchestrator",
            to_agent="preprocessor",
            message_type="preprocess_request",
            payload={
                "doc_id": new_id(),
                "filename": "passport_john_doe.pdf",
                "page_count": 1,
                "document_type": "passport",
            },
            timestamp="2026-01-01T00:00:00Z",
        )
        result = agent.handle_message(msg)
        assert isinstance(result, PreprocessorResult)
        assert result.requires_chunking is False
        assert result.chunking_strategy.processing_level == "metadata_only"


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

    def test_all_rules_have_valid_processing_level(self):
        for key, strategy in _STRATEGY_RULES.items():
            assert strategy.processing_level in PROCESSING_LEVELS, (
                f"Strategy for {key} has invalid processing_level: "
                f"{strategy.processing_level}"
            )

    def test_default_strategy_is_late_chunking(self):
        assert _DEFAULT_STRATEGY.strategy_name == "late_chunking"
        assert _DEFAULT_STRATEGY.processing_level == "late_chunking"
        assert _DEFAULT_STRATEGY.macro_max_tokens == 8192

    def test_skip_strategy_has_zero_tokens(self):
        assert _SKIP_STRATEGY.macro_max_tokens == 0
        assert _SKIP_STRATEGY.child_target_tokens == 0
        assert _SKIP_STRATEGY.processing_level == "skip"

    def test_metadata_only_strategy(self):
        assert _METADATA_ONLY_STRATEGY.processing_level == "metadata_only"
        assert _METADATA_ONLY_STRATEGY.macro_max_tokens == 0

    def test_single_chunk_strategy(self):
        assert _SINGLE_CHUNK_STRATEGY.processing_level == "single_chunk"

    def test_page_level_strategy(self):
        assert _PAGE_LEVEL_STRATEGY.processing_level == "page_level"

    def test_strategy_count_covers_investment_banking(self):
        """Ensure we have strategies for a broad range of IB doc types."""
        assert len(_STRATEGY_RULES) >= 60, (
            f"Expected ≥60 document type strategies, got {len(_STRATEGY_RULES)}"
        )


# ── Processing level constants ───────────────────────────────────────────

class TestProcessingLevels:
    def test_processing_levels_constant(self):
        assert PROCESSING_LEVELS == (
            "skip", "metadata_only", "single_chunk", "page_level", "late_chunking"
        )

    def test_metadata_only_docs_exist(self):
        """At least some doc types should be metadata_only."""
        metadata_only_types = [
            k for k, v in _STRATEGY_RULES.items()
            if v.processing_level == "metadata_only"
        ]
        assert len(metadata_only_types) >= 5

    def test_single_chunk_docs_exist(self):
        single_chunk_types = [
            k for k, v in _STRATEGY_RULES.items()
            if v.processing_level == "single_chunk"
        ]
        assert len(single_chunk_types) >= 5

    def test_page_level_docs_exist(self):
        page_level_types = [
            k for k, v in _STRATEGY_RULES.items()
            if v.processing_level == "page_level"
        ]
        assert len(page_level_types) >= 3

    def test_late_chunking_docs_exist(self):
        late_chunking_types = [
            k for k, v in _STRATEGY_RULES.items()
            if v.processing_level == "late_chunking"
        ]
        assert len(late_chunking_types) >= 20
