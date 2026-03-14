"""Tests for the Feedback Loop Subsystem.

Covers:
- Attribution rules (all 6 rules + multi-layer)
- Training row builders (all 5 layers)
- Boundary policy guard
- Feedback normalizer
- End-to-end pipeline
- Model promotion lifecycle
"""

import pytest
from datetime import datetime, timezone

from feedback_loop.models import (
    BoundaryKey,
    CalibrationTrainingRow,
    ChunkingDecision,
    ChunkingTrainingRow,
    ClassifierDecision,
    ClassifierTrainingRow,
    DecisionLayer,
    ExtractionDecision,
    ExtractionTrainingRow,
    FeedbackAttribution,
    FeedbackEvent,
    FeedbackRating,
    ImpactedLayer,
    ModelCandidate,
    ModelStage,
    NormalizedFeedback,
    PlannerDecision,
    PlannerTrainingRow,
    PredictionTrace,
    ReasonCode,
    TransformationDecision,
)
from feedback_loop.attribution import RuleBasedAttributionEngine
from feedback_loop.training_rows import DefaultTrainingRowBuilder
from feedback_loop.boundary import DefaultBoundaryPolicyGuard
from feedback_loop.normalizer import DefaultFeedbackNormalizer
from feedback_loop.services import (
    InMemoryFeedbackIngestionService,
    InMemoryTraceJoinService,
    InMemoryRetrainingOrchestrator,
    InMemoryModelPromotionController,
    DefaultModelEvaluator,
)
from feedback_loop.pipeline import FeedbackLoopPipeline


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def boundary():
    return BoundaryKey(client="acme_corp", division="investment_banking", jurisdiction="US")


@pytest.fixture
def boundary_other():
    return BoundaryKey(client="other_corp", division="retail", jurisdiction="UK")


@pytest.fixture
def trace(boundary):
    """A realistic prediction trace for a 10-K document processing run."""
    return PredictionTrace(
        trace_id="trace-001",
        query_id="q-001",
        doc_id="doc-001",
        user_id="user-001",
        boundary_key=boundary,
        planner=PlannerDecision(
            action="retrieve_then_synthesize",
            processing_path="full_pipeline",
            routing_confidence=0.85,
            query_decomposition=["find revenue", "find net income"],
        ),
        classifier=ClassifierDecision(
            document_type="10-K",
            classification_label="annual_report_sec",
            confidence=0.92,
            classification_method="deterministic",
            evidence_signals={"filename_match": True},
        ),
        chunking=ChunkingDecision(
            strategy_name="semantic",
            processing_level="late_chunking",
            chunk_count=45,
            chunk_ids=["c1", "c2", "c3"],
            chunk_texts=[
                "Total revenue for fiscal year 2024 was $52.3 billion.",
                "Net income attributable to shareholders was $12.1 billion.",
                "Operating expenses increased by 8% year-over-year.",
            ],
            decision_method="deterministic",
        ),
        extraction=ExtractionDecision(
            extracted_fields={
                "total_revenue": "$52.3 billion",
                "net_income": "$12.1 billion",
                "entity_name": "Acme Corp",
            },
            extraction_method="regex",
            field_confidences={
                "total_revenue": 0.95,
                "net_income": 0.90,
                "entity_name": 0.85,
            },
            source_chunk_ids={
                "total_revenue": ["c1"],
                "net_income": ["c2"],
                "entity_name": ["c1"],
            },
        ),
        transformation=TransformationDecision(
            normalized_fields={"entity_name": "Acme Corporation"},
            mcp_lookups_performed=1,
            unresolved_references=[],
        ),
        final_answer="Revenue was $52.3B, net income was $12.1B.",
        final_confidence=0.92,
        citations=["c1", "c2"],
        total_latency_ms=1500.0,
    )


@pytest.fixture
def engine():
    return RuleBasedAttributionEngine()


@pytest.fixture
def builder():
    return DefaultTrainingRowBuilder()


@pytest.fixture
def normalizer():
    return DefaultFeedbackNormalizer()


@pytest.fixture
def guard():
    return DefaultBoundaryPolicyGuard()


# ── Attribution Rule Tests ───────────────────────────────────────────


class TestRule1Classification:
    """Rule 1: If reviewer corrects document class or schema → classification impacted."""

    def test_document_type_correction(self, engine, trace, boundary):
        event = FeedbackEvent(
            trace_id="trace-001",
            boundary_key=boundary,
            rating=FeedbackRating.CORRECTION,
            correct_document_type="10-Q",
        )
        normalized = NormalizedFeedback(
            feedback_id=event.feedback_id,
            reason_codes=[ReasonCode.WRONG_DOCUMENT_CLASS],
        )
        result = engine.attribute(event, trace, normalized)
        layers = {il.layer for il in result.impacted_layers}
        assert DecisionLayer.CLASSIFICATION in layers

        cls_layer = next(il for il in result.impacted_layers if il.layer == DecisionLayer.CLASSIFICATION)
        assert cls_layer.confidence >= 0.9
        assert ReasonCode.WRONG_DOCUMENT_CLASS in cls_layer.reason_codes
        assert "rule_1" in cls_layer.rule_id

    def test_classification_label_correction(self, engine, trace, boundary):
        event = FeedbackEvent(
            trace_id="trace-001",
            boundary_key=boundary,
            rating=FeedbackRating.CORRECTION,
            correct_classification_label="quarterly_report_sec",
        )
        normalized = NormalizedFeedback(
            feedback_id=event.feedback_id,
            reason_codes=[ReasonCode.WRONG_SCHEMA_APPLIED],
        )
        result = engine.attribute(event, trace, normalized)
        layers = {il.layer for il in result.impacted_layers}
        assert DecisionLayer.CLASSIFICATION in layers


class TestRule2Chunking:
    """Rule 2: If correct evidence span absent from all chunks → chunking impacted."""

    def test_missing_evidence_span(self, engine, trace, boundary):
        event = FeedbackEvent(
            trace_id="trace-001",
            boundary_key=boundary,
            rating=FeedbackRating.CORRECTION,
            correct_evidence_spans=["The company reported EBITDA of $15.2 billion"],
        )
        normalized = NormalizedFeedback(
            feedback_id=event.feedback_id,
            reason_codes=[ReasonCode.MISSING_EVIDENCE_SPAN],
        )
        result = engine.attribute(event, trace, normalized)
        layers = {il.layer for il in result.impacted_layers}
        assert DecisionLayer.CHUNKING in layers

        chunk_layer = next(il for il in result.impacted_layers if il.layer == DecisionLayer.CHUNKING)
        assert chunk_layer.confidence >= 0.8
        assert "absent from all" in chunk_layer.explanation

    def test_evidence_span_present_no_chunking_impact(self, engine, trace, boundary):
        """If evidence IS in chunks, chunking is NOT impacted."""
        event = FeedbackEvent(
            trace_id="trace-001",
            boundary_key=boundary,
            rating=FeedbackRating.CORRECTION,
            correct_evidence_spans=["Total revenue for fiscal year 2024"],
        )
        normalized = NormalizedFeedback(
            feedback_id=event.feedback_id,
            reason_codes=[],
        )
        result = engine.attribute(event, trace, normalized)
        chunking_layers = [il for il in result.impacted_layers if il.layer == DecisionLayer.CHUNKING]
        assert len(chunking_layers) == 0


class TestRule3Extraction:
    """Rule 3: Correct evidence in chunks but wrong value extracted → extraction impacted."""

    def test_wrong_value_extracted(self, engine, trace, boundary):
        # Correct value "$53.0 billion" differs from predicted "$52.3 billion"
        # AND the correct value IS present in the chunk text (we add it)
        trace_with_evidence = trace.model_copy(update={
            "chunking": trace.chunking.model_copy(update={
                "chunk_texts": trace.chunking.chunk_texts + [
                    "Restated total revenue for fiscal year 2024 was $53.0 billion.",
                ],
            }),
        })
        event = FeedbackEvent(
            trace_id="trace-001",
            boundary_key=boundary,
            rating=FeedbackRating.CORRECTION,
            correct_field_values={"total_revenue": "$53.0 billion"},
        )
        normalized = NormalizedFeedback(
            feedback_id=event.feedback_id,
            reason_codes=[ReasonCode.WRONG_VALUE_EXTRACTED],
        )
        result = engine.attribute(event, trace_with_evidence, normalized)
        layers = {il.layer for il in result.impacted_layers}
        assert DecisionLayer.EXTRACTION in layers

        ext_layer = next(il for il in result.impacted_layers if il.layer == DecisionLayer.EXTRACTION)
        assert ext_layer.confidence >= 0.8
        assert "total_revenue" in ext_layer.explanation


class TestRule4UnresolvedReference:
    """Rule 4: Unresolved 'same as above' → extraction + calibration impacted."""

    def test_same_as_above_auto_accepted(self, engine, boundary):
        trace = PredictionTrace(
            trace_id="trace-002",
            boundary_key=boundary,
            planner=PlannerDecision(action="extract"),
            transformation=TransformationDecision(
                unresolved_references=["same as above"],
            ),
            final_confidence=0.7,
        )
        event = FeedbackEvent(
            trace_id="trace-002",
            boundary_key=boundary,
            rating=FeedbackRating.NEGATIVE,
            comment="The value 'same as above' was incorrectly accepted",
        )
        normalized = NormalizedFeedback(
            feedback_id=event.feedback_id,
            reason_codes=[ReasonCode.UNRESOLVED_REFERENCE],
        )
        result = engine.attribute(event, trace, normalized)
        layers = {il.layer for il in result.impacted_layers}
        assert DecisionLayer.EXTRACTION in layers
        assert DecisionLayer.CALIBRATION in layers


class TestRule5Planning:
    """Rule 5: Processing path override or correction rate exceeded → planning impacted."""

    def test_processing_path_override(self, engine, trace, boundary):
        event = FeedbackEvent(
            trace_id="trace-001",
            boundary_key=boundary,
            rating=FeedbackRating.CORRECTION,
            processing_path_override="metadata_only",
        )
        normalized = NormalizedFeedback(
            feedback_id=event.feedback_id,
            reason_codes=[ReasonCode.WRONG_PROCESSING_PATH],
        )
        result = engine.attribute(event, trace, normalized)
        layers = {il.layer for il in result.impacted_layers}
        assert DecisionLayer.PLANNING in layers

        plan_layer = next(il for il in result.impacted_layers if il.layer == DecisionLayer.PLANNING)
        assert "metadata_only" in plan_layer.explanation

    def test_correction_rate_exceeded(self, engine, trace, boundary):
        event = FeedbackEvent(
            trace_id="trace-001",
            boundary_key=boundary,
            rating=FeedbackRating.NEGATIVE,
        )
        normalized = NormalizedFeedback(
            feedback_id=event.feedback_id,
            reason_codes=[ReasonCode.CORRECTION_RATE_EXCEEDED],
        )
        result = engine.attribute(event, trace, normalized)
        layers = {il.layer for il in result.impacted_layers}
        assert DecisionLayer.PLANNING in layers


class TestRule6Calibration:
    """Rule 6: Wrong output with high confidence → calibration impacted."""

    def test_overconfident_wrong_answer(self, engine, trace, boundary):
        # trace.final_confidence = 0.92 (above default 0.8 threshold)
        event = FeedbackEvent(
            trace_id="trace-001",
            boundary_key=boundary,
            rating=FeedbackRating.NEGATIVE,
        )
        normalized = NormalizedFeedback(
            feedback_id=event.feedback_id,
            reason_codes=[],
        )
        result = engine.attribute(event, trace, normalized)
        layers = {il.layer for il in result.impacted_layers}
        assert DecisionLayer.CALIBRATION in layers

        cal_layer = next(il for il in result.impacted_layers if il.layer == DecisionLayer.CALIBRATION)
        assert "0.92" in cal_layer.explanation
        assert cal_layer.confidence >= 0.8

    def test_low_confidence_no_calibration_impact(self, engine, boundary):
        """If confidence is low and answer is wrong, calibration is NOT flagged by rule 6."""
        trace = PredictionTrace(
            trace_id="trace-003",
            boundary_key=boundary,
            planner=PlannerDecision(action="extract"),
            final_confidence=0.3,
        )
        event = FeedbackEvent(
            trace_id="trace-003",
            boundary_key=boundary,
            rating=FeedbackRating.NEGATIVE,
        )
        normalized = NormalizedFeedback(
            feedback_id=event.feedback_id,
            reason_codes=[],
        )
        result = engine.attribute(event, trace, normalized)
        calibration_layers = [il for il in result.impacted_layers if il.layer == DecisionLayer.CALIBRATION]
        assert len(calibration_layers) == 0


class TestMultiLayerAttribution:
    """A single feedback event can impact multiple layers."""

    def test_classification_plus_extraction_plus_calibration(self, engine, trace, boundary):
        """Correct doc type + correct field value (present in chunks) + high confidence → 3 layers."""
        # Add chunk containing the correct value so Rule 3 fires
        trace_ext = trace.model_copy(update={
            "chunking": trace.chunking.model_copy(update={
                "chunk_texts": trace.chunking.chunk_texts + [
                    "Restated total revenue was $53.0 billion after adjustments.",
                ],
            }),
        })
        event = FeedbackEvent(
            trace_id="trace-001",
            boundary_key=boundary,
            rating=FeedbackRating.CORRECTION,
            correct_document_type="10-Q",
            correct_field_values={"total_revenue": "$53.0 billion"},
        )
        normalized = NormalizedFeedback(
            feedback_id=event.feedback_id,
            reason_codes=[ReasonCode.WRONG_DOCUMENT_CLASS, ReasonCode.WRONG_VALUE_EXTRACTED],
        )
        result = engine.attribute(event, trace_ext, normalized)
        layers = {il.layer for il in result.impacted_layers}
        assert DecisionLayer.CLASSIFICATION in layers
        assert DecisionLayer.EXTRACTION in layers
        assert DecisionLayer.CALIBRATION in layers
        assert len(result.impacted_layers) >= 3

    def test_positive_feedback_no_attribution(self, engine, trace, boundary):
        """Positive feedback produces no impacted layers."""
        event = FeedbackEvent(
            trace_id="trace-001",
            boundary_key=boundary,
            rating=FeedbackRating.POSITIVE,
        )
        normalized = NormalizedFeedback(feedback_id=event.feedback_id, reason_codes=[])
        result = engine.attribute(event, trace, normalized)
        assert len(result.impacted_layers) == 0


# ── Normalizer Tests ─────────────────────────────────────────────────


class TestNormalizer:
    def test_comment_wrong_type(self, normalizer, boundary):
        event = FeedbackEvent(
            boundary_key=boundary,
            rating=FeedbackRating.NEGATIVE,
            comment="This is the wrong document type, it should be a 10-Q",
        )
        result = normalizer.normalize(event)
        assert ReasonCode.WRONG_DOCUMENT_CLASS in result.reason_codes

    def test_comment_same_as_above(self, normalizer, boundary):
        event = FeedbackEvent(
            boundary_key=boundary,
            rating=FeedbackRating.NEGATIVE,
            comment="The value 'same as above' was not resolved properly",
        )
        result = normalizer.normalize(event)
        assert ReasonCode.UNRESOLVED_REFERENCE in result.reason_codes

    def test_comment_hallucination(self, normalizer, boundary):
        event = FeedbackEvent(
            boundary_key=boundary,
            rating=FeedbackRating.NEGATIVE,
            comment="This answer was hallucinated, not in the document",
        )
        result = normalizer.normalize(event)
        assert ReasonCode.HALLUCINATED_CONTENT in result.reason_codes

    def test_explicit_corrections_generate_reason_codes(self, normalizer, boundary):
        event = FeedbackEvent(
            boundary_key=boundary,
            rating=FeedbackRating.CORRECTION,
            correct_document_type="10-Q",
            correct_field_values={"revenue": "100M"},
        )
        result = normalizer.normalize(event)
        assert ReasonCode.WRONG_DOCUMENT_CLASS in result.reason_codes
        assert ReasonCode.WRONG_VALUE_EXTRACTED in result.reason_codes
        assert result.severity >= 0.8

    def test_overconfidence_detected_from_trace(self, normalizer, boundary, trace):
        event = FeedbackEvent(
            boundary_key=boundary,
            rating=FeedbackRating.NEGATIVE,
        )
        result = normalizer.normalize(event, trace)
        assert ReasonCode.OVERCONFIDENT_WRONG_ANSWER in result.reason_codes

    def test_positive_no_reason_codes(self, normalizer, boundary):
        event = FeedbackEvent(
            boundary_key=boundary,
            rating=FeedbackRating.POSITIVE,
        )
        result = normalizer.normalize(event)
        assert result.reason_codes == []
        assert result.severity == 0.0


# ── Training Row Builder Tests ───────────────────────────────────────


class TestTrainingRowBuilder:
    def test_planner_rows_on_planning_impact(self, builder, trace, boundary):
        event = FeedbackEvent(
            boundary_key=boundary,
            rating=FeedbackRating.CORRECTION,
            processing_path_override="metadata_only",
        )
        attribution = FeedbackAttribution(
            feedback_id=event.feedback_id,
            trace_id=trace.trace_id,
            boundary_key=boundary,
            impacted_layers=[ImpactedLayer(
                layer=DecisionLayer.PLANNING,
                confidence=0.9,
                explanation="test",
                rule_id="rule_5",
            )],
        )
        rows = builder.build(event, trace, attribution)
        assert DecisionLayer.PLANNING in rows
        planner_rows = rows[DecisionLayer.PLANNING]
        assert len(planner_rows) == 1
        row = planner_rows[0]
        assert isinstance(row, PlannerTrainingRow)
        assert row.source_feedback_ids == [event.feedback_id]
        assert row.boundary_key.key == boundary.key
        assert row.chosen_processing_path == "full_pipeline"
        assert row.correct_processing_path == "metadata_only"
        assert row.accuracy == 0.0
        assert row.error_penalty == 1.0

    def test_classifier_rows_on_classification_impact(self, builder, trace, boundary):
        event = FeedbackEvent(
            boundary_key=boundary,
            rating=FeedbackRating.CORRECTION,
            correct_document_type="10-Q",
        )
        attribution = FeedbackAttribution(
            feedback_id=event.feedback_id,
            trace_id=trace.trace_id,
            boundary_key=boundary,
            impacted_layers=[ImpactedLayer(
                layer=DecisionLayer.CLASSIFICATION,
                confidence=0.95,
                explanation="test",
                rule_id="rule_1",
            )],
        )
        rows = builder.build(event, trace, attribution)
        assert DecisionLayer.CLASSIFICATION in rows
        cls_rows = rows[DecisionLayer.CLASSIFICATION]
        assert len(cls_rows) == 1
        row = cls_rows[0]
        assert isinstance(row, ClassifierTrainingRow)
        assert row.predicted_document_type == "10-K"
        assert row.correct_document_type == "10-Q"
        assert row.is_correct is False
        assert row.boundary_key.key == boundary.key

    def test_chunking_rows_with_quality_score(self, builder, trace, boundary):
        event = FeedbackEvent(
            boundary_key=boundary,
            rating=FeedbackRating.CORRECTION,
            correct_evidence_spans=["EBITDA of $15.2 billion"],
            correct_field_values={"total_revenue": "$52.3 billion"},
        )
        attribution = FeedbackAttribution(
            feedback_id=event.feedback_id,
            trace_id=trace.trace_id,
            boundary_key=boundary,
            impacted_layers=[ImpactedLayer(
                layer=DecisionLayer.CHUNKING,
                confidence=0.9,
                explanation="test",
                rule_id="rule_2",
            )],
        )
        rows = builder.build(event, trace, attribution)
        assert DecisionLayer.CHUNKING in rows
        chunk_rows = rows[DecisionLayer.CHUNKING]
        assert len(chunk_rows) == 1
        row = chunk_rows[0]
        assert isinstance(row, ChunkingTrainingRow)
        assert row.chosen_strategy == "semantic"
        assert len(row.missing_evidence_spans) == 1
        assert row.context_loss > 0
        # Verify Q(k,x) formula is computable
        q = row.quality_score
        assert isinstance(q, float)

    def test_extraction_rows_per_field(self, builder, trace, boundary):
        event = FeedbackEvent(
            boundary_key=boundary,
            rating=FeedbackRating.CORRECTION,
            correct_field_values={
                "total_revenue": "$52.3 billion USD",
                "net_income": "$13.0 billion",
            },
        )
        attribution = FeedbackAttribution(
            feedback_id=event.feedback_id,
            trace_id=trace.trace_id,
            boundary_key=boundary,
            impacted_layers=[ImpactedLayer(
                layer=DecisionLayer.EXTRACTION,
                confidence=0.9,
                explanation="test",
                rule_id="rule_3",
            )],
        )
        rows = builder.build(event, trace, attribution)
        assert DecisionLayer.EXTRACTION in rows
        ext_rows = rows[DecisionLayer.EXTRACTION]
        # One row per corrected field
        assert len(ext_rows) == 2
        for row in ext_rows:
            assert isinstance(row, ExtractionTrainingRow)
            assert row.source_feedback_ids == [event.feedback_id]
            assert row.boundary_key.key == boundary.key

    def test_calibration_rows(self, builder, trace, boundary):
        event = FeedbackEvent(
            boundary_key=boundary,
            rating=FeedbackRating.NEGATIVE,
        )
        attribution = FeedbackAttribution(
            feedback_id=event.feedback_id,
            trace_id=trace.trace_id,
            boundary_key=boundary,
            impacted_layers=[ImpactedLayer(
                layer=DecisionLayer.CALIBRATION,
                confidence=0.9,
                explanation="test",
                rule_id="rule_6",
            )],
        )
        rows = builder.build(event, trace, attribution)
        assert DecisionLayer.CALIBRATION in rows
        cal_rows = rows[DecisionLayer.CALIBRATION]
        assert len(cal_rows) == 1
        row = cal_rows[0]
        assert isinstance(row, CalibrationTrainingRow)
        assert row.is_correct is False
        assert row.is_overconfident is True
        assert row.final_confidence == 0.92
        assert row.confidence_bucket == "0.9-1.0"

    def test_only_impacted_layers_produce_rows(self, builder, trace, boundary):
        """Layers NOT in attribution produce no rows."""
        event = FeedbackEvent(
            boundary_key=boundary,
            rating=FeedbackRating.CORRECTION,
            correct_document_type="10-Q",
        )
        attribution = FeedbackAttribution(
            feedback_id=event.feedback_id,
            trace_id=trace.trace_id,
            boundary_key=boundary,
            impacted_layers=[ImpactedLayer(
                layer=DecisionLayer.CLASSIFICATION,
                confidence=0.95,
                explanation="test",
                rule_id="rule_1",
            )],
        )
        rows = builder.build(event, trace, attribution)
        assert DecisionLayer.PLANNING not in rows
        assert DecisionLayer.CHUNKING not in rows
        assert DecisionLayer.EXTRACTION not in rows
        assert DecisionLayer.CALIBRATION not in rows

    def test_reward_formula(self, boundary):
        """Verify R = Accuracy - λ₁·ReviewCost - λ₂·Latency - λ₃·ErrorPenalty."""
        row = PlannerTrainingRow(
            source_feedback_ids=["f1"],
            boundary_key=boundary,
            accuracy=1.0,
            review_cost=0.0,
            latency_ms=1000.0,
            error_penalty=0.0,
            lambda_review_cost=0.1,
            lambda_latency=0.01,
            lambda_error_penalty=1.0,
        )
        # R = 1.0 - 0.1*0.0 - 0.01*(1000/1000) - 1.0*0.0 = 1.0 - 0.01 = 0.99
        assert abs(row.reward - 0.99) < 0.001


# ── Boundary Policy Guard Tests ──────────────────────────────────────


class TestBoundaryPolicyGuard:
    def test_same_boundary_always_valid(self, guard, boundary):
        row = PlannerTrainingRow(
            source_feedback_ids=["f1"],
            boundary_key=boundary,
        )
        assert guard.validate_row(row, boundary) is True

    def test_different_boundary_invalid(self, guard, boundary, boundary_other):
        row = PlannerTrainingRow(
            source_feedback_ids=["f1"],
            boundary_key=boundary_other,
        )
        assert guard.validate_row(row, boundary) is False

    def test_sanitize_strips_text_fields(self, guard, boundary):
        row = ExtractionTrainingRow(
            source_feedback_ids=["f1"],
            boundary_key=boundary,
            field_name="revenue",
            predicted_value="$52.3 billion",
            correct_value="$52.3 billion USD",
            source_text_snippet="The total revenue for the fiscal year...",
        )
        sanitized = guard.sanitize_for_sharing(row)
        assert sanitized.source_text_snippet == ""
        assert sanitized.predicted_value == ""
        assert sanitized.correct_value == ""
        # Structural fields preserved
        assert sanitized.field_name == "revenue"
        assert sanitized.boundary_key.key == boundary.key

    def test_sharing_not_approved_by_default(self, guard, boundary, boundary_other):
        assert guard.is_sharing_approved(boundary, boundary_other) is False

    def test_explicit_sharing_approval(self, guard, boundary, boundary_other):
        guard.approve_sharing(boundary, boundary_other)
        assert guard.is_sharing_approved(boundary, boundary_other) is True

    def test_client_level_sharing(self, guard):
        b1 = BoundaryKey(client="acme", division="div_a", jurisdiction="US")
        b2 = BoundaryKey(client="acme", division="div_b", jurisdiction="US")
        assert guard.is_sharing_approved(b1, b2) is False
        guard.enable_client_sharing("acme")
        assert guard.is_sharing_approved(b1, b2) is True

    def test_filter_rows_for_boundary(self, guard, boundary, boundary_other):
        own_row = PlannerTrainingRow(
            source_feedback_ids=["f1"],
            boundary_key=boundary,
            query_text="sensitive query text",
        )
        other_row = PlannerTrainingRow(
            source_feedback_ids=["f2"],
            boundary_key=boundary_other,
            query_text="other client query",
        )
        # Without approval, other_row is dropped
        result = guard.filter_rows_for_boundary([own_row, other_row], boundary)
        assert len(result) == 1
        assert result[0].boundary_key.key == boundary.key

        # With approval, other_row is included but sanitized
        guard.approve_sharing(boundary_other, boundary)
        result = guard.filter_rows_for_boundary([own_row, other_row], boundary)
        assert len(result) == 2
        sanitized = result[1]
        assert sanitized.query_text == ""  # text stripped


# ── Model Promotion Tests ────────────────────────────────────────────


class TestModelPromotion:
    def test_valid_lifecycle(self, boundary):
        ctrl = InMemoryModelPromotionController()
        candidate = ModelCandidate(
            layer=DecisionLayer.CLASSIFICATION,
            boundary_key=boundary,
            model_version="v2",
            parent_model_version="v1",
            stage=ModelStage.SHADOW,
        )
        ctrl.register(candidate)

        # shadow → canary
        c = ctrl.promote(candidate.candidate_id, ModelStage.CANARY)
        assert c.stage == ModelStage.CANARY

        # canary → approved
        c = ctrl.promote(candidate.candidate_id, ModelStage.APPROVED)
        assert c.stage == ModelStage.APPROVED

        # Should be active
        active = ctrl.get_active_model(DecisionLayer.CLASSIFICATION, boundary)
        assert active is not None
        assert active.candidate_id == candidate.candidate_id

    def test_invalid_transition_rejected(self, boundary):
        ctrl = InMemoryModelPromotionController()
        candidate = ModelCandidate(
            layer=DecisionLayer.EXTRACTION,
            boundary_key=boundary,
            stage=ModelStage.SHADOW,
        )
        ctrl.register(candidate)

        # shadow → approved is not allowed (must go through canary)
        with pytest.raises(ValueError, match="Invalid transition"):
            ctrl.promote(candidate.candidate_id, ModelStage.APPROVED)

    def test_rollback(self, boundary):
        ctrl = InMemoryModelPromotionController()
        candidate = ModelCandidate(
            layer=DecisionLayer.CLASSIFICATION,
            boundary_key=boundary,
            stage=ModelStage.SHADOW,
        )
        ctrl.register(candidate)
        ctrl.promote(candidate.candidate_id, ModelStage.CANARY)
        ctrl.promote(candidate.candidate_id, ModelStage.APPROVED)

        # Rollback
        c = ctrl.rollback(candidate.candidate_id)
        assert c.stage == ModelStage.ROLLBACK_READY

        # No longer active
        active = ctrl.get_active_model(DecisionLayer.CLASSIFICATION, boundary)
        assert active is None


# ── End-to-End Pipeline Test ─────────────────────────────────────────


class TestEndToEndPipeline:
    """Full pipeline: trace → feedback → attribution → training rows."""

    def test_full_pipeline_correction(self, trace, boundary):
        """Realistic e2e: user corrects document type + field value on high-confidence result."""
        pipeline = FeedbackLoopPipeline()

        # Extend trace with a chunk containing the correct value so Rule 3 fires
        trace_ext = trace.model_copy(update={
            "chunking": trace.chunking.model_copy(update={
                "chunk_texts": trace.chunking.chunk_texts + [
                    "Restated total revenue was $53.0 billion after audit adjustments.",
                ],
            }),
        })

        # 1. Store the prediction trace
        pipeline.trace_join.store_trace(trace_ext)

        # 2. User submits correction feedback
        event = FeedbackEvent(
            trace_id=trace_ext.trace_id,
            query_id="q-001",
            doc_id="doc-001",
            boundary_key=boundary,
            rating=FeedbackRating.CORRECTION,
            correct_document_type="10-Q",
            correct_field_values={"total_revenue": "$53.0 billion"},
            comment="This is actually a quarterly report, not an annual report.",
        )

        # 3. Process through pipeline
        result = pipeline.process(event)

        # Verify attribution
        layers = {il.layer for il in result.attribution.impacted_layers}
        assert DecisionLayer.CLASSIFICATION in layers, "Classification should be impacted (doc type corrected)"
        assert DecisionLayer.EXTRACTION in layers, "Extraction should be impacted (field value corrected)"
        assert DecisionLayer.CALIBRATION in layers, "Calibration should be impacted (high confidence + wrong)"

        # Verify training rows generated for each impacted layer
        assert DecisionLayer.CLASSIFICATION in result.training_rows
        assert DecisionLayer.EXTRACTION in result.training_rows
        assert DecisionLayer.CALIBRATION in result.training_rows

        # Verify lineage
        for layer, rows in result.training_rows.items():
            for row in rows:
                assert event.feedback_id in row.source_feedback_ids
                assert row.boundary_key.key == boundary.key

        # Verify rows submitted
        assert result.rows_submitted >= 3

        # Verify classifier training row content
        cls_rows = result.training_rows[DecisionLayer.CLASSIFICATION]
        assert cls_rows[0].predicted_document_type == "10-K"
        assert cls_rows[0].correct_document_type == "10-Q"

        # Verify calibration training row content
        cal_rows = result.training_rows[DecisionLayer.CALIBRATION]
        assert cal_rows[0].is_overconfident is True
        assert cal_rows[0].final_confidence == 0.92

    def test_pipeline_positive_feedback(self, trace, boundary):
        """Positive feedback produces no training rows (nothing to correct)."""
        pipeline = FeedbackLoopPipeline()
        pipeline.trace_join.store_trace(trace)

        event = FeedbackEvent(
            trace_id="trace-001",
            boundary_key=boundary,
            rating=FeedbackRating.POSITIVE,
            comment="Great answer!",
        )
        result = pipeline.process(event)
        assert len(result.attribution.impacted_layers) == 0
        assert result.rows_submitted == 0

    def test_pipeline_unresolved_reference(self, boundary):
        """Unresolved 'same as above' triggers extraction + calibration."""
        pipeline = FeedbackLoopPipeline()

        trace = PredictionTrace(
            trace_id="trace-unresolved",
            query_id="q-unresolved",
            boundary_key=boundary,
            planner=PlannerDecision(action="extract"),
            transformation=TransformationDecision(
                unresolved_references=["same as above"],
            ),
            final_confidence=0.75,
        )
        pipeline.trace_join.store_trace(trace)

        event = FeedbackEvent(
            trace_id="trace-unresolved",
            boundary_key=boundary,
            rating=FeedbackRating.NEGATIVE,
            comment="'Same as above' was not resolved",
        )
        result = pipeline.process(event)
        layers = {il.layer for il in result.attribution.impacted_layers}
        assert DecisionLayer.EXTRACTION in layers
        assert DecisionLayer.CALIBRATION in layers

    def test_pipeline_no_trace_warning(self, boundary):
        """If no trace found, pipeline still completes with warning."""
        pipeline = FeedbackLoopPipeline()
        event = FeedbackEvent(
            trace_id="nonexistent",
            boundary_key=boundary,
            rating=FeedbackRating.NEGATIVE,
        )
        result = pipeline.process(event)
        assert len(result.warnings) > 0
        assert "No prediction trace" in result.warnings[0]

    def test_pipeline_retraining_trigger(self, trace, boundary):
        """After enough feedback, retraining can be triggered."""
        pipeline = FeedbackLoopPipeline()
        pipeline.trace_join.store_trace(trace)

        # Submit 12 correction events
        for i in range(12):
            event = FeedbackEvent(
                trace_id="trace-001",
                boundary_key=boundary,
                rating=FeedbackRating.CORRECTION,
                correct_document_type="10-Q",
            )
            pipeline.process(event)

        # Check dataset sizes
        cls_size = pipeline.orchestrator.get_dataset_size(
            DecisionLayer.CLASSIFICATION, boundary,
        )
        assert cls_size >= 12

        # Trigger retraining
        job_id = pipeline.orchestrator.trigger_retraining(
            layer=DecisionLayer.CLASSIFICATION,
            boundary_key=boundary,
            min_rows=10,
        )
        assert job_id is not None
        status = pipeline.orchestrator.get_job_status(job_id)
        assert status["status"] == "pending"


# ── Boundary Key Tests ───────────────────────────────────────────────


class TestBoundaryKey:
    def test_key_format(self):
        b = BoundaryKey(client="acme", division="ib", jurisdiction="US")
        assert b.key == "acme|ib|US"

    def test_key_client_only(self):
        b = BoundaryKey(client="acme")
        assert b.key == "acme"

    def test_same_boundary(self):
        b1 = BoundaryKey(client="acme", division="ib")
        b2 = BoundaryKey(client="acme", division="ib")
        assert b1.is_same_boundary(b2)

    def test_shares_client(self):
        b1 = BoundaryKey(client="acme", division="ib")
        b2 = BoundaryKey(client="acme", division="retail")
        assert b1.shares_client(b2)
        assert not b1.is_same_boundary(b2)


# ── Model Evaluator Tests ────────────────────────────────────────────


class TestModelEvaluator:
    def test_evaluate_improvement(self, boundary):
        evaluator = DefaultModelEvaluator(min_improvement=0.01)
        candidate = ModelCandidate(
            layer=DecisionLayer.CLASSIFICATION,
            boundary_key=boundary,
        )
        report = evaluator.evaluate(
            candidate,
            baseline_metrics={"accuracy": 0.85},
            validation_data=[1, 2, 3],
        )
        assert report.candidate_id == candidate.candidate_id
        assert report.baseline_metrics.accuracy == 0.85
