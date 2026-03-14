"""Feedback Loop Pipeline — End-to-End Orchestration.

Ties all services together into the complete feedback-to-retraining pipeline:
    Ingest → Join → Normalize → Attribute → Build → Guard → Submit

This is the main entry point for processing feedback events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from feedback_loop.attribution import RuleBasedAttributionEngine
from feedback_loop.boundary import DefaultBoundaryPolicyGuard
from feedback_loop.models import (
    DecisionLayer,
    FeedbackAttribution,
    FeedbackEvent,
    NormalizedFeedback,
    PredictionTrace,
)
from feedback_loop.normalizer import DefaultFeedbackNormalizer
from feedback_loop.services import (
    InMemoryFeedbackIngestionService,
    InMemoryRetrainingOrchestrator,
    InMemoryTraceJoinService,
)
from feedback_loop.training_rows import DefaultTrainingRowBuilder

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of processing a single feedback event through the full pipeline."""
    feedback_id: str
    trace_id: str
    normalized: NormalizedFeedback
    attribution: FeedbackAttribution
    training_rows: Dict[DecisionLayer, List[Any]]
    rows_submitted: int
    warnings: List[str]


class FeedbackLoopPipeline:
    """End-to-end feedback loop pipeline.

    Processes a feedback event through all stages:
    1. Ingest (validate + store raw event)
    2. Join (link with runtime prediction trace)
    3. Normalize (derive structured reason codes)
    4. Attribute (determine impacted decision layers)
    5. Build (generate layer-specific training rows)
    6. Guard (enforce boundary policies)
    7. Submit (queue rows for retraining)
    """

    def __init__(
        self,
        ingestion: Optional[InMemoryFeedbackIngestionService] = None,
        trace_join: Optional[InMemoryTraceJoinService] = None,
        normalizer: Optional[DefaultFeedbackNormalizer] = None,
        attribution: Optional[RuleBasedAttributionEngine] = None,
        row_builder: Optional[DefaultTrainingRowBuilder] = None,
        boundary_guard: Optional[DefaultBoundaryPolicyGuard] = None,
        orchestrator: Optional[InMemoryRetrainingOrchestrator] = None,
    ) -> None:
        self.ingestion = ingestion or InMemoryFeedbackIngestionService()
        self.trace_join = trace_join or InMemoryTraceJoinService()
        self.normalizer = normalizer or DefaultFeedbackNormalizer()
        self.attribution = attribution or RuleBasedAttributionEngine()
        self.row_builder = row_builder or DefaultTrainingRowBuilder()
        self.boundary_guard = boundary_guard or DefaultBoundaryPolicyGuard()
        self.orchestrator = orchestrator or InMemoryRetrainingOrchestrator(
            boundary_guard=self.boundary_guard,
        )

    def process(self, event: FeedbackEvent) -> PipelineResult:
        """Process a single feedback event through the full pipeline."""
        warnings: List[str] = []

        # 1. Ingest
        feedback_id = self.ingestion.ingest(event)

        # 2. Join with trace
        trace = self.trace_join.join(event)
        if trace is None:
            # Create a minimal trace if none found
            warnings.append(f"No prediction trace found for feedback {feedback_id}")
            trace = PredictionTrace(
                trace_id="",
                query_id=event.query_id,
                doc_id=event.doc_id,
                boundary_key=event.boundary_key,
            )

        # 3. Normalize
        normalized = self.normalizer.normalize(event, trace)

        # 4. Attribute
        attribution = self.attribution.attribute(event, trace, normalized)

        # 5. Build training rows
        training_rows = self.row_builder.build(event, trace, attribution)

        # 6. Guard + Submit
        rows_submitted = 0
        for layer, rows in training_rows.items():
            # Validate boundary on each row
            valid_rows = [
                r for r in rows
                if self.boundary_guard.validate_row(r, event.boundary_key)
            ]
            if valid_rows:
                self.orchestrator.submit_rows(layer, valid_rows, event.boundary_key)
                rows_submitted += len(valid_rows)

        logger.info(
            "Pipeline complete for feedback %s: %d layers impacted, %d rows submitted",
            feedback_id, len(attribution.impacted_layers), rows_submitted,
        )

        return PipelineResult(
            feedback_id=feedback_id,
            trace_id=trace.trace_id,
            normalized=normalized,
            attribution=attribution,
            training_rows=training_rows,
            rows_submitted=rows_submitted,
            warnings=warnings,
        )
