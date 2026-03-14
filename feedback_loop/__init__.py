"""Feedback Loop Subsystem — Production-Grade Feedback-to-Retraining Pipeline.

This package implements the enterprise feedback loop for the agentic IDP platform.
It maps the full decision chain:

    x -> p -> c -> k -> e -> t -> y

Where:
    x = document and case context
    p = planning / orchestration decision
    c = classification decision
    k = chunking / linking decision
    e = extraction decision
    t = transformation / resolution
    y = final output

The subsystem:
1. Ingests feedback events from reviewers, validators, downstream systems
2. Joins feedback with runtime prediction traces
3. Normalizes raw feedback into structured types and reason codes
4. Attributes feedback to one or more impacted decision layers
5. Generates layer-specific training rows (planner, classifier, chunking, extraction, calibration)
6. Enforces boundary-safe training policies B = (client, division, jurisdiction)
7. Orchestrates retraining with model evaluation and controlled promotion

Mathematical objectives:
    Planning:     π*(z) = argmax_a E[R(a, z)]
    Classification: P(c|z) = softmax(Wz + b)
    Chunking:     k*(x) = argmax_k E[Q(k, x)]
    Extraction:   v̂ = E(x, c, S)
    Calibration:  P(correct|z) = σ(wᵀz)

Hard constraints:
    - Never train globally on raw cross-client document text
    - Never collapse all feedback into extraction labels
    - Always preserve source feedback lineage
    - Every training row carries source_feedback_ids and boundary_key
"""

from feedback_loop.models import (
    BoundaryGranularity,
    BoundaryKey,
    PredictionTrace,
    FeedbackEvent,
    FeedbackAttribution,
    ImpactedLayer,
    PlannerTrainingRow,
    ClassifierTrainingRow,
    ChunkingTrainingRow,
    ExtractionTrainingRow,
    CalibrationTrainingRow,
    ModelCandidate,
    EvaluationReport,
)
from feedback_loop.pipeline import FeedbackLoopPipeline, PipelineResult
from feedback_loop.services import (
    InMemoryFeedbackIngestionService,
    InMemoryTraceJoinService,
    InMemoryRetrainingOrchestrator,
    PostgresFeedbackIngestionService,
    PostgresTraceJoinService,
    PostgresRetrainingOrchestrator,
)

__all__ = [
    # Domain models
    "BoundaryGranularity",
    "BoundaryKey",
    "PredictionTrace",
    "FeedbackEvent",
    "FeedbackAttribution",
    "ImpactedLayer",
    "PlannerTrainingRow",
    "ClassifierTrainingRow",
    "ChunkingTrainingRow",
    "ExtractionTrainingRow",
    "CalibrationTrainingRow",
    "ModelCandidate",
    "EvaluationReport",
    # Pipeline
    "FeedbackLoopPipeline",
    "PipelineResult",
    # In-memory services (tests + local dev)
    "InMemoryFeedbackIngestionService",
    "InMemoryTraceJoinService",
    "InMemoryRetrainingOrchestrator",
    # DB-backed services (production)
    "PostgresFeedbackIngestionService",
    "PostgresTraceJoinService",
    "PostgresRetrainingOrchestrator",
]
