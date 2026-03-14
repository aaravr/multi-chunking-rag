"""Attribution Engine — Rule-Based Implementation.

Attributes a feedback event to one or more impacted decision layers using
deterministic rules.  Supports multiple impacted layers per event.

Deterministic Attribution Rules:

Rule 1 — Classification:
    If reviewer corrects document class or schema → classification impacted.

Rule 2 — Chunking:
    If correct evidence span is absent from all produced chunks → chunking impacted.

Rule 3 — Extraction:
    If correct evidence exists in available chunk but wrong value extracted → extraction impacted.

Rule 4 — Extraction + Calibration:
    If unresolved reference ("same as above") was auto-accepted → extraction and calibration.

Rule 5 — Planning:
    If manual processing path override or correction rate exceeds threshold → planning impacted.

Rule 6 — Calibration:
    If wrong output had high confidence above threshold → calibration impacted.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from feedback_loop.interfaces import AttributionEngine
from feedback_loop.models import (
    BoundaryKey,
    DecisionLayer,
    FeedbackAttribution,
    FeedbackEvent,
    FeedbackRating,
    ImpactedLayer,
    NormalizedFeedback,
    PredictionTrace,
    ReasonCode,
    _new_id,
)

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────

# Rule 5: correction rate threshold for planning attribution
CORRECTION_RATE_THRESHOLD = 0.3

# Rule 6: confidence threshold for overconfidence attribution
OVERCONFIDENCE_THRESHOLD = 0.8


# ── Rule-Based Attribution Engine ────────────────────────────────────


class RuleBasedAttributionEngine(AttributionEngine):
    """Deterministic rule-based attribution engine.

    Applies six rules in order.  Each rule can independently fire,
    producing multi-layer attribution for a single feedback event.

    Design priorities:
    - Deterministic rules first (no ML inference in attribution path)
    - Extensible for future learned attribution
    - Every attribution includes confidence and textual explanation
    """

    def __init__(
        self,
        overconfidence_threshold: float = OVERCONFIDENCE_THRESHOLD,
        correction_rate_threshold: float = CORRECTION_RATE_THRESHOLD,
        correction_rate_provider: Optional[object] = None,
    ) -> None:
        self._overconfidence_threshold = overconfidence_threshold
        self._correction_rate_threshold = correction_rate_threshold
        self._correction_rate_provider = correction_rate_provider

    def attribute(
        self,
        event: FeedbackEvent,
        trace: PredictionTrace,
        normalized: NormalizedFeedback,
    ) -> FeedbackAttribution:
        """Apply all deterministic rules and collect impacted layers."""
        impacted: List[ImpactedLayer] = []

        impacted.extend(self._rule_1_classification(event, trace, normalized))
        impacted.extend(self._rule_2_chunking(event, trace, normalized))
        impacted.extend(self._rule_3_extraction(event, trace, normalized))
        impacted.extend(self._rule_4_unresolved_reference(event, trace, normalized))
        impacted.extend(self._rule_5_planning(event, trace, normalized))
        impacted.extend(self._rule_6_calibration(event, trace, normalized))

        # Deduplicate by layer (keep highest confidence per layer)
        impacted = self._deduplicate(impacted)

        if not impacted:
            # Fallback: if negative/correction but no rule matched, attribute to extraction
            if event.rating in (FeedbackRating.NEGATIVE, FeedbackRating.CORRECTION):
                impacted.append(ImpactedLayer(
                    layer=DecisionLayer.EXTRACTION,
                    confidence=0.3,
                    explanation="No specific rule matched; defaulting to extraction layer.",
                    reason_codes=[ReasonCode.OTHER],
                    rule_id="fallback",
                ))

        logger.info(
            "Attribution for feedback %s: %d layer(s) impacted [%s]",
            event.feedback_id,
            len(impacted),
            ", ".join(il.layer.value for il in impacted),
        )

        return FeedbackAttribution(
            feedback_id=event.feedback_id,
            trace_id=trace.trace_id,
            boundary_key=event.boundary_key,
            impacted_layers=impacted,
            attribution_method="rule_based",
        )

    # ── Rule 1: Classification ───────────────────────────────────────

    def _rule_1_classification(
        self,
        event: FeedbackEvent,
        trace: PredictionTrace,
        normalized: NormalizedFeedback,
    ) -> List[ImpactedLayer]:
        """If reviewer corrects document class or schema → classification impacted."""
        results: List[ImpactedLayer] = []

        # Check explicit corrections
        type_corrected = (
            event.correct_document_type is not None
            and event.correct_document_type != trace.classifier.document_type
        )
        label_corrected = (
            event.correct_classification_label is not None
            and event.correct_classification_label != trace.classifier.classification_label
        )

        # Check reason codes from normalizer
        has_class_reason = ReasonCode.WRONG_DOCUMENT_CLASS in normalized.reason_codes
        has_schema_reason = ReasonCode.WRONG_SCHEMA_APPLIED in normalized.reason_codes

        if type_corrected or label_corrected or has_class_reason or has_schema_reason:
            explanations = []
            reason_codes = []
            if type_corrected:
                explanations.append(
                    f"Document type corrected: '{trace.classifier.document_type}' → "
                    f"'{event.correct_document_type}'"
                )
                reason_codes.append(ReasonCode.WRONG_DOCUMENT_CLASS)
            if label_corrected:
                explanations.append(
                    f"Classification label corrected: '{trace.classifier.classification_label}' → "
                    f"'{event.correct_classification_label}'"
                )
                reason_codes.append(ReasonCode.WRONG_SCHEMA_APPLIED)
            if has_class_reason and not type_corrected:
                explanations.append("Normalizer detected wrong document class signal.")
                reason_codes.append(ReasonCode.WRONG_DOCUMENT_CLASS)
            if has_schema_reason and not label_corrected:
                explanations.append("Normalizer detected wrong schema signal.")
                reason_codes.append(ReasonCode.WRONG_SCHEMA_APPLIED)

            results.append(ImpactedLayer(
                layer=DecisionLayer.CLASSIFICATION,
                confidence=0.95 if (type_corrected or label_corrected) else 0.7,
                explanation="; ".join(explanations),
                reason_codes=reason_codes,
                rule_id="rule_1_classification",
            ))

        return results

    # ── Rule 2: Chunking ─────────────────────────────────────────────

    def _rule_2_chunking(
        self,
        event: FeedbackEvent,
        trace: PredictionTrace,
        normalized: NormalizedFeedback,
    ) -> List[ImpactedLayer]:
        """If correct evidence span is absent from all produced chunks → chunking impacted."""
        results: List[ImpactedLayer] = []

        if not event.correct_evidence_spans:
            # Check reason code
            if ReasonCode.MISSING_EVIDENCE_SPAN in normalized.reason_codes:
                results.append(ImpactedLayer(
                    layer=DecisionLayer.CHUNKING,
                    confidence=0.7,
                    explanation="Normalizer detected missing evidence span signal.",
                    reason_codes=[ReasonCode.MISSING_EVIDENCE_SPAN],
                    rule_id="rule_2_chunking",
                ))
            return results

        # Check if any correct evidence span is absent from produced chunks
        produced_texts = set()
        for ct in trace.chunking.chunk_texts:
            produced_texts.add(ct.lower().strip())

        missing_spans = []
        for span in event.correct_evidence_spans:
            span_lower = span.lower().strip()
            # Check if span is a substring of any chunk text
            found = any(span_lower in ct for ct in produced_texts)
            if not found:
                missing_spans.append(span)

        if missing_spans:
            results.append(ImpactedLayer(
                layer=DecisionLayer.CHUNKING,
                confidence=0.9,
                explanation=(
                    f"{len(missing_spans)} correct evidence span(s) absent from all "
                    f"{len(trace.chunking.chunk_ids)} produced chunks."
                ),
                reason_codes=[ReasonCode.MISSING_EVIDENCE_SPAN],
                rule_id="rule_2_chunking",
            ))

        return results

    # ── Rule 3: Extraction ───────────────────────────────────────────

    def _rule_3_extraction(
        self,
        event: FeedbackEvent,
        trace: PredictionTrace,
        normalized: NormalizedFeedback,
    ) -> List[ImpactedLayer]:
        """If correct evidence exists in chunks but wrong value extracted → extraction impacted."""
        results: List[ImpactedLayer] = []

        if not event.correct_field_values:
            # Check reason codes
            if ReasonCode.WRONG_VALUE_EXTRACTED in normalized.reason_codes:
                results.append(ImpactedLayer(
                    layer=DecisionLayer.EXTRACTION,
                    confidence=0.7,
                    explanation="Normalizer detected wrong value extraction signal.",
                    reason_codes=[ReasonCode.WRONG_VALUE_EXTRACTED],
                    rule_id="rule_3_extraction",
                ))
            return results

        # For each corrected field, check if correct value exists in chunk texts
        produced_texts_lower = " ".join(
            ct.lower() for ct in trace.chunking.chunk_texts
        )

        wrong_fields = []
        for field_name, correct_value in event.correct_field_values.items():
            predicted = trace.extraction.extracted_fields.get(field_name, "")
            if predicted and predicted != correct_value:
                # Correct value IS in chunks but extraction got it wrong
                if correct_value.lower() in produced_texts_lower:
                    wrong_fields.append(field_name)

        if wrong_fields:
            results.append(ImpactedLayer(
                layer=DecisionLayer.EXTRACTION,
                confidence=0.9,
                explanation=(
                    f"Correct value(s) present in chunks but extracted incorrectly "
                    f"for field(s): {', '.join(wrong_fields)}."
                ),
                reason_codes=[ReasonCode.WRONG_VALUE_EXTRACTED],
                rule_id="rule_3_extraction",
            ))

        return results

    # ── Rule 4: Unresolved Reference ─────────────────────────────────

    def _rule_4_unresolved_reference(
        self,
        event: FeedbackEvent,
        trace: PredictionTrace,
        normalized: NormalizedFeedback,
    ) -> List[ImpactedLayer]:
        """If unresolved reference ('same as above') was auto-accepted → extraction + calibration."""
        results: List[ImpactedLayer] = []

        has_unresolved = (
            ReasonCode.UNRESOLVED_REFERENCE in normalized.reason_codes
            or len(trace.transformation.unresolved_references) > 0
        )

        if has_unresolved:
            unresolved_refs = trace.transformation.unresolved_references
            results.append(ImpactedLayer(
                layer=DecisionLayer.EXTRACTION,
                confidence=0.85,
                explanation=(
                    f"Unresolved reference(s) auto-accepted: {unresolved_refs or ['detected by normalizer']}. "
                    f"Extraction should have flagged or resolved these."
                ),
                reason_codes=[ReasonCode.UNRESOLVED_REFERENCE],
                rule_id="rule_4_unresolved_reference",
            ))
            results.append(ImpactedLayer(
                layer=DecisionLayer.CALIBRATION,
                confidence=0.75,
                explanation=(
                    "Unresolved reference was auto-accepted with confidence; "
                    "calibration should have signaled uncertainty."
                ),
                reason_codes=[ReasonCode.UNRESOLVED_REFERENCE],
                rule_id="rule_4_unresolved_reference",
            ))

        return results

    # ── Rule 5: Planning ─────────────────────────────────────────────

    def _rule_5_planning(
        self,
        event: FeedbackEvent,
        trace: PredictionTrace,
        normalized: NormalizedFeedback,
    ) -> List[ImpactedLayer]:
        """If manual processing path override or correction rate exceeds threshold → planning."""
        results: List[ImpactedLayer] = []

        # Check for explicit processing path override
        if event.processing_path_override is not None:
            path_differs = (
                event.processing_path_override != trace.planner.processing_path
            )
            if path_differs:
                results.append(ImpactedLayer(
                    layer=DecisionLayer.PLANNING,
                    confidence=0.9,
                    explanation=(
                        f"Processing path overridden: '{trace.planner.processing_path}' → "
                        f"'{event.processing_path_override}'."
                    ),
                    reason_codes=[ReasonCode.WRONG_PROCESSING_PATH],
                    rule_id="rule_5_planning",
                ))

        # Check reason codes
        if ReasonCode.WRONG_PROCESSING_PATH in normalized.reason_codes:
            if not results:  # avoid duplicate
                results.append(ImpactedLayer(
                    layer=DecisionLayer.PLANNING,
                    confidence=0.7,
                    explanation="Normalizer detected wrong processing path signal.",
                    reason_codes=[ReasonCode.WRONG_PROCESSING_PATH],
                    rule_id="rule_5_planning",
                ))

        if ReasonCode.CORRECTION_RATE_EXCEEDED in normalized.reason_codes:
            results.append(ImpactedLayer(
                layer=DecisionLayer.PLANNING,
                confidence=0.8,
                explanation=(
                    f"Correction rate for chosen processing path exceeds threshold "
                    f"({self._correction_rate_threshold:.0%})."
                ),
                reason_codes=[ReasonCode.CORRECTION_RATE_EXCEEDED],
                rule_id="rule_5_planning",
            ))

        return results

    # ── Rule 6: Calibration ──────────────────────────────────────────

    def _rule_6_calibration(
        self,
        event: FeedbackEvent,
        trace: PredictionTrace,
        normalized: NormalizedFeedback,
    ) -> List[ImpactedLayer]:
        """If wrong output had high confidence above threshold → calibration impacted."""
        results: List[ImpactedLayer] = []

        is_wrong = event.rating in (FeedbackRating.NEGATIVE, FeedbackRating.CORRECTION)
        is_overconfident = trace.final_confidence >= self._overconfidence_threshold

        if is_wrong and is_overconfident:
            results.append(ImpactedLayer(
                layer=DecisionLayer.CALIBRATION,
                confidence=0.9,
                explanation=(
                    f"Output was incorrect but had high confidence "
                    f"({trace.final_confidence:.2f} >= {self._overconfidence_threshold:.2f}). "
                    f"Confidence model is miscalibrated."
                ),
                reason_codes=[ReasonCode.OVERCONFIDENT_WRONG_ANSWER],
                rule_id="rule_6_calibration",
            ))

        # Check reason code from normalizer
        if ReasonCode.OVERCONFIDENT_WRONG_ANSWER in normalized.reason_codes:
            if not results:
                results.append(ImpactedLayer(
                    layer=DecisionLayer.CALIBRATION,
                    confidence=0.7,
                    explanation="Normalizer detected overconfident wrong answer signal.",
                    reason_codes=[ReasonCode.OVERCONFIDENT_WRONG_ANSWER],
                    rule_id="rule_6_calibration",
                ))

        return results

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(layers: List[ImpactedLayer]) -> List[ImpactedLayer]:
        """Keep highest-confidence entry per layer, merge reason codes."""
        by_layer: dict[DecisionLayer, ImpactedLayer] = {}
        for il in layers:
            existing = by_layer.get(il.layer)
            if existing is None:
                by_layer[il.layer] = il
            elif il.confidence > existing.confidence:
                merged_reasons = list(set(existing.reason_codes + il.reason_codes))
                merged_explanation = existing.explanation + " | " + il.explanation
                by_layer[il.layer] = ImpactedLayer(
                    layer=il.layer,
                    confidence=il.confidence,
                    explanation=merged_explanation,
                    reason_codes=merged_reasons,
                    rule_id=il.rule_id,
                )
            else:
                merged_reasons = list(set(existing.reason_codes + il.reason_codes))
                merged_explanation = existing.explanation + " | " + il.explanation
                by_layer[il.layer] = ImpactedLayer(
                    layer=existing.layer,
                    confidence=existing.confidence,
                    explanation=merged_explanation,
                    reason_codes=merged_reasons,
                    rule_id=existing.rule_id,
                )
        return list(by_layer.values())
