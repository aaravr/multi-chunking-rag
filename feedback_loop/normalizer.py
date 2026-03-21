"""Feedback Normalizer — Maps raw feedback signals into structured types and reason codes.

Deterministic rule-based normalization.  Parses free-text comments and
structured correction fields to derive reason codes and severity.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from feedback_loop.interfaces import FeedbackNormalizer as FeedbackNormalizerInterface
from feedback_loop.models import (
    FeedbackEvent,
    FeedbackRating,
    NormalizedFeedback,
    PredictionTrace,
    ReasonCode,
)

logger = logging.getLogger(__name__)

# Keyword patterns mapped to reason codes (case-insensitive)
_COMMENT_PATTERNS: List[tuple[str, ReasonCode]] = [
    (r"wrong\s*(document\s*)?type", ReasonCode.WRONG_DOCUMENT_CLASS),
    (r"wrong\s*(document\s*)?class", ReasonCode.WRONG_DOCUMENT_CLASS),
    (r"misclassif", ReasonCode.WRONG_DOCUMENT_CLASS),
    (r"wrong\s*schema", ReasonCode.WRONG_SCHEMA_APPLIED),
    (r"missing\s*(evidence|chunk|span|text)", ReasonCode.MISSING_EVIDENCE_SPAN),
    (r"not\s*found\s*in\s*chunk", ReasonCode.MISSING_EVIDENCE_SPAN),
    (r"wrong\s*(value|extraction|answer|field)", ReasonCode.WRONG_VALUE_EXTRACTED),
    (r"incorrect\s*(value|extraction|answer|field)", ReasonCode.WRONG_VALUE_EXTRACTED),
    (r"incomplete\s*(extract|field|value)", ReasonCode.INCOMPLETE_EXTRACTION),
    (r"same\s*as\s*above", ReasonCode.UNRESOLVED_REFERENCE),
    (r"see\s*(above|previous|prior)", ReasonCode.UNRESOLVED_REFERENCE),
    (r"unresolved\s*ref", ReasonCode.UNRESOLVED_REFERENCE),
    (r"overconfident", ReasonCode.OVERCONFIDENT_WRONG_ANSWER),
    (r"too\s*confident", ReasonCode.OVERCONFIDENT_WRONG_ANSWER),
    (r"should\s*not\s*be\s*(so\s*)?confident", ReasonCode.OVERCONFIDENT_WRONG_ANSWER),
    (r"hallucin", ReasonCode.HALLUCINATED_CONTENT),
    (r"made\s*up", ReasonCode.HALLUCINATED_CONTENT),
    (r"fabricat", ReasonCode.HALLUCINATED_CONTENT),
    (r"wrong\s*path", ReasonCode.WRONG_PROCESSING_PATH),
    (r"wrong\s*pipeline", ReasonCode.WRONG_PROCESSING_PATH),
    (r"should\s*have\s*(used|been|processed)", ReasonCode.WRONG_PROCESSING_PATH),
]

_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), rc) for p, rc in _COMMENT_PATTERNS]


class DefaultFeedbackNormalizer(FeedbackNormalizerInterface):
    """Rule-based feedback normalizer.

    Derives structured reason codes from:
    1. Explicit correction fields on FeedbackEvent
    2. Keyword matching on free-text comments
    3. Trace comparison (predicted vs corrected values)
    """

    def normalize(
        self,
        event: FeedbackEvent,
        trace: Optional[PredictionTrace] = None,
    ) -> NormalizedFeedback:
        """Normalize a raw feedback event into structured form."""
        reason_codes: List[ReasonCode] = []
        corrections: Dict[str, Any] = {}

        # 1. Structural signals from explicit correction fields
        if event.correct_document_type is not None:
            reason_codes.append(ReasonCode.WRONG_DOCUMENT_CLASS)
            corrections["document_type"] = event.correct_document_type

        if event.correct_classification_label is not None:
            reason_codes.append(ReasonCode.WRONG_SCHEMA_APPLIED)
            corrections["classification_label"] = event.correct_classification_label

        if event.correct_evidence_spans:
            reason_codes.append(ReasonCode.MISSING_EVIDENCE_SPAN)
            corrections["evidence_spans"] = event.correct_evidence_spans

        if event.correct_field_values:
            reason_codes.append(ReasonCode.WRONG_VALUE_EXTRACTED)
            corrections["field_values"] = event.correct_field_values

        if event.processing_path_override is not None:
            reason_codes.append(ReasonCode.WRONG_PROCESSING_PATH)
            corrections["processing_path"] = event.processing_path_override

        # 2. Keyword matching on comment
        if event.comment:
            for pattern, rc in _COMPILED_PATTERNS:
                if pattern.search(event.comment) and rc not in reason_codes:
                    reason_codes.append(rc)

        # 3. Trace-based signals
        if trace is not None:
            # Check for unresolved references in transformation
            if trace.transformation.unresolved_references:
                if ReasonCode.UNRESOLVED_REFERENCE not in reason_codes:
                    reason_codes.append(ReasonCode.UNRESOLVED_REFERENCE)

            # Check overconfidence
            if (
                event.rating in (FeedbackRating.NEGATIVE, FeedbackRating.CORRECTION)
                and trace.final_confidence >= 0.8
            ):
                if ReasonCode.OVERCONFIDENT_WRONG_ANSWER not in reason_codes:
                    reason_codes.append(ReasonCode.OVERCONFIDENT_WRONG_ANSWER)

        # Fallback: if negative/correction but no specific reason found
        if not reason_codes and event.rating in (FeedbackRating.NEGATIVE, FeedbackRating.CORRECTION):
            reason_codes.append(ReasonCode.OTHER)

        # Severity: higher for corrections than simple negatives
        severity = 0.0
        if event.rating == FeedbackRating.CORRECTION:
            severity = 0.9
        elif event.rating == FeedbackRating.NEGATIVE:
            severity = 0.6
        elif event.rating == FeedbackRating.POSITIVE:
            severity = 0.0

        # Deduplicate
        reason_codes = list(dict.fromkeys(reason_codes))

        return NormalizedFeedback(
            feedback_id=event.feedback_id,
            reason_codes=reason_codes,
            structured_corrections=corrections,
            severity=severity,
        )
