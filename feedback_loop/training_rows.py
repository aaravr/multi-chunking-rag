"""Training Row Builders — Layer-Specific Training Data Generation.

Converts attributed feedback events into training rows for each impacted layer.
Every row preserves source_feedback_ids and boundary_key for lineage and isolation.

Training row types and mathematical objectives:
    PlannerTrainingRow:     R = Accuracy - λ₁·ReviewCost - λ₂·Latency - λ₃·ErrorPenalty
    ClassifierTrainingRow:  P(c|z) = softmax(Wz + b)
    ChunkingTrainingRow:    Q(k,x) = α·EvidenceRecall + β·FieldAccuracy - γ·ContextLoss
    ExtractionTrainingRow:  v̂ = E(x, c, S)
    CalibrationTrainingRow: P(correct|z) = σ(wᵀz)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from feedback_loop.interfaces import TrainingRowBuilder
from feedback_loop.models import (
    CalibrationTrainingRow,
    ChunkingTrainingRow,
    ClassifierTrainingRow,
    DecisionLayer,
    ExtractionTrainingRow,
    FeedbackAttribution,
    FeedbackEvent,
    FeedbackRating,
    ImpactedLayer,
    PlannerTrainingRow,
    PredictionTrace,
    ReasonCode,
)

logger = logging.getLogger(__name__)


class DefaultTrainingRowBuilder(TrainingRowBuilder):
    """Builds layer-specific training rows from attributed feedback.

    Only generates rows for layers that were attributed as impacted.
    Supports one-to-many mapping: a single feedback event can produce
    training rows across multiple layers.
    """

    def build(
        self,
        event: FeedbackEvent,
        trace: PredictionTrace,
        attribution: FeedbackAttribution,
    ) -> Dict[DecisionLayer, List[Any]]:
        """Build training rows for all impacted layers."""
        result: Dict[DecisionLayer, List[Any]] = {}

        for impacted in attribution.impacted_layers:
            layer = impacted.layer
            rows = self._build_for_layer(event, trace, impacted)
            if rows:
                result.setdefault(layer, []).extend(rows)

        total = sum(len(v) for v in result.values())
        logger.info(
            "Built %d training row(s) across %d layer(s) for feedback %s",
            total, len(result), event.feedback_id,
        )
        return result

    def _build_for_layer(
        self,
        event: FeedbackEvent,
        trace: PredictionTrace,
        impacted: ImpactedLayer,
    ) -> List[Any]:
        """Dispatch to the correct layer builder."""
        builders = {
            DecisionLayer.PLANNING: self._build_planner_rows,
            DecisionLayer.CLASSIFICATION: self._build_classifier_rows,
            DecisionLayer.CHUNKING: self._build_chunking_rows,
            DecisionLayer.EXTRACTION: self._build_extraction_rows,
            DecisionLayer.CALIBRATION: self._build_calibration_rows,
        }
        builder = builders.get(impacted.layer)
        if builder is None:
            logger.warning("No builder for layer %s", impacted.layer)
            return []
        return builder(event, trace, impacted)

    # ── Planner Rows ─────────────────────────────────────────────────

    def _build_planner_rows(
        self,
        event: FeedbackEvent,
        trace: PredictionTrace,
        impacted: ImpactedLayer,
    ) -> List[PlannerTrainingRow]:
        """Build planner training rows.

        Objective: π*(z) = argmax_a E[R(a, z)]
        R = Accuracy - λ₁·ReviewCost - λ₂·Latency - λ₃·ErrorPenalty
        """
        is_correct = event.rating == FeedbackRating.POSITIVE
        return [PlannerTrainingRow(
            source_feedback_ids=[event.feedback_id],
            boundary_key=event.boundary_key,
            query_text=trace.final_answer[:500] if trace.final_answer else "",
            document_type=trace.classifier.document_type,
            classification_label=trace.classifier.classification_label,
            page_count=0,
            chosen_action=trace.planner.action,
            chosen_processing_path=trace.planner.processing_path,
            query_decomposition=trace.planner.query_decomposition,
            correct_action=None,
            correct_processing_path=event.processing_path_override,
            accuracy=1.0 if is_correct else 0.0,
            review_cost=0.0 if is_correct else 1.0,
            latency_ms=trace.total_latency_ms,
            error_penalty=0.0 if is_correct else 1.0,
        )]

    # ── Classifier Rows ──────────────────────────────────────────────

    def _build_classifier_rows(
        self,
        event: FeedbackEvent,
        trace: PredictionTrace,
        impacted: ImpactedLayer,
    ) -> List[ClassifierTrainingRow]:
        """Build classifier training rows.

        Objective: P(c|z) = softmax(Wz + b)
        """
        correct_type = (
            event.correct_document_type
            or trace.classifier.document_type
        )
        correct_label = (
            event.correct_classification_label
            or trace.classifier.classification_label
        )
        is_correct = (
            correct_type == trace.classifier.document_type
            and correct_label == trace.classifier.classification_label
        )
        return [ClassifierTrainingRow(
            source_feedback_ids=[event.feedback_id],
            boundary_key=event.boundary_key,
            filename="",
            front_matter_text="",
            structural_signals=trace.classifier.evidence_signals,
            predicted_document_type=trace.classifier.document_type,
            predicted_classification_label=trace.classifier.classification_label,
            predicted_confidence=trace.classifier.confidence,
            correct_document_type=correct_type,
            correct_classification_label=correct_label,
            is_correct=is_correct,
        )]

    # ── Chunking Rows ────────────────────────────────────────────────

    def _build_chunking_rows(
        self,
        event: FeedbackEvent,
        trace: PredictionTrace,
        impacted: ImpactedLayer,
    ) -> List[ChunkingTrainingRow]:
        """Build chunking training rows.

        Objective: Q(k,x) = α·EvidenceRecall + β·FieldAccuracy - γ·ContextLoss
        """
        # Compute evidence_recall: fraction of correct spans found in chunks
        total_spans = len(event.correct_evidence_spans) if event.correct_evidence_spans else 1
        found_spans = 0
        missing_spans: List[str] = []

        if event.correct_evidence_spans:
            produced_texts_lower = " ".join(
                ct.lower() for ct in trace.chunking.chunk_texts
            )
            for span in event.correct_evidence_spans:
                if span.lower().strip() in produced_texts_lower:
                    found_spans += 1
                else:
                    missing_spans.append(span)

        evidence_recall = found_spans / total_spans if total_spans > 0 else 0.0

        # field_accuracy: how many extracted fields were correct
        total_fields = len(trace.extraction.extracted_fields) or 1
        correct_fields = 0
        if event.correct_field_values:
            for fname, correct_val in event.correct_field_values.items():
                predicted = trace.extraction.extracted_fields.get(fname, "")
                if predicted == correct_val:
                    correct_fields += 1
        field_accuracy = correct_fields / total_fields

        # context_loss: 1.0 if missing spans, scaled by proportion
        context_loss = len(missing_spans) / total_spans if total_spans > 0 else 0.0

        return [ChunkingTrainingRow(
            source_feedback_ids=[event.feedback_id],
            boundary_key=event.boundary_key,
            document_type=trace.classifier.document_type,
            classification_label=trace.classifier.classification_label,
            page_count=0,
            chosen_strategy=trace.chunking.strategy_name,
            processing_level=trace.chunking.processing_level,
            chunk_count=trace.chunking.chunk_count,
            evidence_recall=evidence_recall,
            field_accuracy=field_accuracy,
            context_loss=context_loss,
            missing_evidence_spans=missing_spans,
        )]

    # ── Extraction Rows ──────────────────────────────────────────────

    def _build_extraction_rows(
        self,
        event: FeedbackEvent,
        trace: PredictionTrace,
        impacted: ImpactedLayer,
    ) -> List[ExtractionTrainingRow]:
        """Build extraction training rows.

        Objective: v̂ = E(x, c, S)
        One row per corrected field.
        """
        rows: List[ExtractionTrainingRow] = []

        if event.correct_field_values:
            for field_name, correct_value in event.correct_field_values.items():
                predicted = trace.extraction.extracted_fields.get(field_name, "")
                confidence = trace.extraction.field_confidences.get(field_name, 0.0)
                source_ids = trace.extraction.source_chunk_ids.get(field_name, [])
                is_unresolved = (
                    ReasonCode.UNRESOLVED_REFERENCE in [
                        rc for il in [impacted] for rc in il.reason_codes
                    ]
                )

                rows.append(ExtractionTrainingRow(
                    source_feedback_ids=[event.feedback_id],
                    boundary_key=event.boundary_key,
                    document_type=trace.classifier.document_type,
                    classification_label=trace.classifier.classification_label,
                    field_name=field_name,
                    predicted_value=predicted,
                    predicted_confidence=confidence,
                    extraction_method=trace.extraction.extraction_method,
                    source_chunk_ids=source_ids,
                    source_text_snippet="",
                    correct_value=correct_value,
                    is_correct=(predicted == correct_value),
                    is_unresolved_reference=is_unresolved,
                ))
        elif event.correct_answer is not None:
            # General answer correction without per-field detail
            rows.append(ExtractionTrainingRow(
                source_feedback_ids=[event.feedback_id],
                boundary_key=event.boundary_key,
                document_type=trace.classifier.document_type,
                classification_label=trace.classifier.classification_label,
                field_name="_answer",
                predicted_value=trace.final_answer[:500],
                predicted_confidence=trace.final_confidence,
                extraction_method=trace.extraction.extraction_method,
                correct_value=event.correct_answer,
                is_correct=False,
            ))

        return rows

    # ── Calibration Rows ─────────────────────────────────────────────

    def _build_calibration_rows(
        self,
        event: FeedbackEvent,
        trace: PredictionTrace,
        impacted: ImpactedLayer,
    ) -> List[CalibrationTrainingRow]:
        """Build calibration training rows.

        Objective: P(correct|z) = σ(wᵀz)
        """
        is_correct = event.rating == FeedbackRating.POSITIVE
        conf = trace.final_confidence

        # Determine calibration bucket (0.1-wide buckets)
        bucket_lower = int(conf * 10) / 10.0
        bucket_upper = bucket_lower + 0.1
        bucket = f"{bucket_lower:.1f}-{bucket_upper:.1f}"

        return [CalibrationTrainingRow(
            source_feedback_ids=[event.feedback_id],
            boundary_key=event.boundary_key,
            planner_confidence=trace.planner.routing_confidence,
            classifier_confidence=trace.classifier.confidence,
            chunking_quality_score=0.0,
            extraction_confidence=max(
                trace.extraction.field_confidences.values(), default=0.0
            ),
            final_confidence=conf,
            document_type=trace.classifier.document_type,
            query_intent=trace.planner.action,
            chunk_count=trace.chunking.chunk_count,
            model_id=trace.extraction.model_id,
            is_correct=is_correct,
            confidence_bucket=bucket,
            is_overconfident=(not is_correct and conf >= 0.8),
            is_underconfident=(is_correct and conf < 0.5),
        )]
