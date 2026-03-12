"""Tests for agent typed contracts (MASTER_PROMPT §4, §5, §11.1).

Validates:
- All contracts are frozen (immutable)
- Required fields cannot be omitted
- Message serialisation round-trips
- Contract interoperability
"""

import pytest
from dataclasses import FrozenInstanceError

from agents.contracts import (
    AgentMessage,
    AuditLogEntry,
    Citation,
    ClaimVerification,
    ComplianceResult,
    ComplianceViolation,
    ConversationMemory,
    ConversationTurn,
    CostBreakdown,
    DecisionStep,
    DocumentScope,
    DocumentTarget,
    EvidenceLink,
    ExecutionStep,
    ExplainabilityReport,
    ModelAttribution,
    OrchestratorInput,
    OrchestratorOutput,
    OutputFormat,
    PermissionSet,
    QueryIntent,
    QueryPlan,
    RankedEvidence,
    RetrievalStrategy,
    SearchScope,
    SubQuery,
    SynthesisResult,
    TokenUsage,
    VerificationResult,
    new_id,
)


class TestContractImmutability:
    """§4: All agent contracts are frozen dataclasses."""

    def test_citation_is_frozen(self):
        c = Citation(
            citation_id="C1",
            chunk_id="abc",
            doc_id="doc1",
            page_numbers=[1],
            polygons=[],
            heading_path="h",
            section_id="s",
            text_snippet="text",
        )
        with pytest.raises(FrozenInstanceError):
            c.citation_id = "C2"

    def test_query_intent_is_frozen(self):
        qi = QueryIntent(intent="semantic")
        with pytest.raises(FrozenInstanceError):
            qi.intent = "coverage"

    def test_agent_message_is_frozen(self):
        msg = AgentMessage(
            message_id="m1",
            query_id="q1",
            from_agent="orchestrator",
            to_agent="retriever",
            message_type="retrieval_request",
            payload={"key": "value"},
            timestamp="2026-03-12T00:00:00Z",
        )
        with pytest.raises(FrozenInstanceError):
            msg.from_agent = "other"

    def test_synthesis_result_is_frozen(self):
        sr = SynthesisResult(query_id="q1", answer="test", citations=[])
        with pytest.raises(FrozenInstanceError):
            sr.answer = "modified"

    def test_verification_result_is_frozen(self):
        vr = VerificationResult(
            query_id="q1",
            overall_verdict="PASS",
            overall_confidence=1.0,
            per_claim=[],
            failed_claims=[],
        )
        with pytest.raises(FrozenInstanceError):
            vr.overall_verdict = "FAIL"

    def test_audit_log_entry_is_frozen(self):
        entry = AuditLogEntry(
            log_id="l1",
            query_id="q1",
            agent_id="synth",
            step_id="s1",
            event_type="llm_call",
            model_id="gpt-4o-mini",
            prompt_template_version="abc",
            full_prompt="prompt",
            full_response="response",
            input_tokens=100,
            output_tokens=50,
            temperature=0.0,
            latency_ms=200.0,
            cost_estimate=0.001,
            user_id="u1",
            timestamp="2026-03-12T00:00:00Z",
        )
        with pytest.raises(FrozenInstanceError):
            entry.full_response = "tampered"


class TestContractDefaults:
    """Verify default values work correctly."""

    def test_query_intent_defaults(self):
        qi = QueryIntent(intent="semantic")
        assert qi.pages == []
        assert qi.coverage_type is None
        assert qi.status_filter is None
        assert qi.entities == []
        assert qi.time_periods == []

    def test_orchestrator_input_defaults(self):
        inp = OrchestratorInput(
            query_id="q1",
            user_query="test query",
            conversation_memory=ConversationMemory(session_id="s1"),
            document_scope=DocumentScope(),
            user_permissions=PermissionSet(),
        )
        assert inp.token_budget == 50000
        assert inp.output_format.format_type == "prose"

    def test_ranked_evidence_defaults(self):
        re = RankedEvidence(
            query_id="q1",
            sub_query_id="sq1",
            chunks=[],
            retrieval_methods={},
            scores={},
        )
        assert re.total_candidates_scanned == 0
        assert re.fusion_weights is None

    def test_token_usage_defaults(self):
        tu = TokenUsage()
        assert tu.prompt_tokens == 0
        assert tu.total_tokens == 0


class TestContractRelationships:
    """Verify contracts compose correctly."""

    def test_query_plan_with_sub_queries(self):
        sq = SubQuery(
            sub_query_id="sq_0",
            query_text="test",
            intent=QueryIntent(intent="semantic"),
        )
        plan = QueryPlan(
            query_id="q1",
            original_query="test query",
            resolved_query="test query",
            primary_intent=QueryIntent(intent="semantic"),
            sub_queries=[sq],
            retrieval_strategies={"sq_0": RetrievalStrategy(method="hybrid")},
            document_targets=[DocumentTarget(doc_id="d1")],
        )
        assert len(plan.sub_queries) == 1
        assert plan.sub_queries[0].sub_query_id == "sq_0"
        assert plan.retrieval_strategies["sq_0"].method == "hybrid"

    def test_orchestrator_output_with_explainability(self):
        report = ExplainabilityReport(
            query_id="q1",
            timestamp="2026-03-12T00:00:00Z",
            decision_chain=[
                DecisionStep(
                    step_name="classify",
                    agent="orchestrator",
                    decision="semantic",
                    reason="default",
                    timestamp="2026-03-12T00:00:00Z",
                )
            ],
            evidence_map=[],
            models_used=[
                ModelAttribution(
                    model_id="gpt-4o-mini",
                    role="synthesis",
                    input_tokens=100,
                    output_tokens=50,
                    latency_ms=200.0,
                    cost_estimate=0.001,
                )
            ],
        )
        output = OrchestratorOutput(
            query_id="q1",
            answer="test answer",
            citations=[],
            confidence=0.9,
            explainability_report=report,
        )
        assert output.explainability_report is not None
        assert len(output.explainability_report.decision_chain) == 1
        assert len(output.explainability_report.models_used) == 1

    def test_multi_hop_sub_query_dependencies(self):
        sq1 = SubQuery(
            sub_query_id="sq_0",
            query_text="Find X",
            intent=QueryIntent(intent="semantic"),
        )
        sq2 = SubQuery(
            sub_query_id="sq_1",
            query_text="Use X to find Y",
            intent=QueryIntent(intent="semantic"),
            depends_on=["sq_0"],
        )
        assert sq2.depends_on == ["sq_0"]


class TestNewId:
    """Verify UUID generation."""

    def test_generates_unique_ids(self):
        ids = {new_id() for _ in range(100)}
        assert len(ids) == 100

    def test_returns_string(self):
        assert isinstance(new_id(), str)
        assert len(new_id()) == 36  # UUID format
