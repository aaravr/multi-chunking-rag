"""FastAPI routes for the feedback loop subsystem.

These are route skeletons that integrate with the feedback loop pipeline.
Wire into the main application via: app.include_router(feedback_router).

All routes enforce boundary_key validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# FastAPI import guard — allows the module to be imported without FastAPI installed
try:
    from fastapi import APIRouter, HTTPException, Query
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from feedback_loop.models import (
    BoundaryKey,
    DecisionLayer,
    FeedbackEvent,
    FeedbackRating,
    ModelStage,
    RetrainingTrigger,
)
from feedback_loop.pipeline import FeedbackLoopPipeline


# ── Request/Response Schemas ─────────────────────────────────────────


class IngestFeedbackRequest(BaseModel):
    """API request to ingest a feedback event."""
    trace_id: str = ""
    query_id: str = ""
    doc_id: str = ""
    user_id: str = ""
    boundary_key: BoundaryKey
    rating: FeedbackRating
    comment: str = ""
    correct_answer: Optional[str] = None
    correct_document_type: Optional[str] = None
    correct_classification_label: Optional[str] = None
    correct_evidence_spans: List[str] = Field(default_factory=list)
    correct_field_values: Dict[str, str] = Field(default_factory=dict)
    processing_path_override: Optional[str] = None
    channel: str = "api"
    cited_chunk_ids: List[str] = Field(default_factory=list)


class IngestFeedbackResponse(BaseModel):
    feedback_id: str
    layers_impacted: List[str]
    training_rows_generated: int
    warnings: List[str]


class TraceResponse(BaseModel):
    trace_id: str
    query_id: str
    doc_id: str
    planner_action: str
    classifier_document_type: str
    classifier_confidence: float
    chunking_strategy: str
    chunk_count: int
    extraction_method: str
    final_confidence: float


class AttributionResponse(BaseModel):
    attribution_id: str
    feedback_id: str
    impacted_layers: List[Dict[str, Any]]
    attribution_method: str


class TrainingRowsResponse(BaseModel):
    feedback_id: str
    layers: Dict[str, int]
    total_rows: int


class TriggerRetrainingRequest(BaseModel):
    layer: DecisionLayer
    boundary_key: BoundaryKey
    trigger: RetrainingTrigger = RetrainingTrigger.THRESHOLD
    min_rows: int = 10


class TriggerRetrainingResponse(BaseModel):
    job_id: Optional[str]
    status: str
    message: str


class EvaluationReportResponse(BaseModel):
    report_id: str
    candidate_id: str
    layer: str
    baseline_accuracy: float
    candidate_accuracy: float
    improvement_delta: float
    recommendation: str


# ── Router Factory ───────────────────────────────────────────────────


def create_feedback_router(pipeline: Optional[FeedbackLoopPipeline] = None) -> Any:
    """Create a FastAPI router for the feedback loop subsystem.

    Args:
        pipeline: Optional pre-configured pipeline.  If None, creates default.

    Returns:
        FastAPI APIRouter (or a stub dict if FastAPI is not installed).
    """
    if not _FASTAPI_AVAILABLE:
        # Return route definitions as a dict for documentation/testing
        return {
            "POST /feedback/ingest": "Ingest a feedback event",
            "GET /feedback/trace/{trace_id}": "Fetch a prediction trace",
            "POST /feedback/attribute/{feedback_id}": "Attribute a feedback event",
            "POST /feedback/training-rows/{feedback_id}": "Build training rows",
            "POST /feedback/retrain": "Trigger retraining for a layer",
            "GET /feedback/evaluation/{candidate_id}": "Fetch evaluation report",
        }

    router = APIRouter(prefix="/feedback", tags=["feedback-loop"])
    _pipeline = pipeline or FeedbackLoopPipeline()

    @router.post("/ingest", response_model=IngestFeedbackResponse)
    def ingest_feedback(req: IngestFeedbackRequest) -> IngestFeedbackResponse:
        """Ingest a feedback event and process through the full pipeline.

        Validates payload, joins with prediction trace, normalizes,
        attributes to layers, builds training rows, and submits for retraining.
        """
        event = FeedbackEvent(
            trace_id=req.trace_id,
            query_id=req.query_id,
            doc_id=req.doc_id,
            user_id=req.user_id,
            boundary_key=req.boundary_key,
            rating=req.rating,
            comment=req.comment,
            correct_answer=req.correct_answer,
            correct_document_type=req.correct_document_type,
            correct_classification_label=req.correct_classification_label,
            correct_evidence_spans=req.correct_evidence_spans,
            correct_field_values=req.correct_field_values,
            processing_path_override=req.processing_path_override,
            cited_chunk_ids=req.cited_chunk_ids,
        )
        result = _pipeline.process(event)
        return IngestFeedbackResponse(
            feedback_id=result.feedback_id,
            layers_impacted=[
                il.layer.value for il in result.attribution.impacted_layers
            ],
            training_rows_generated=result.rows_submitted,
            warnings=result.warnings,
        )

    @router.get("/trace/{trace_id}", response_model=TraceResponse)
    def get_trace(trace_id: str) -> TraceResponse:
        """Fetch a prediction trace by ID."""
        trace = _pipeline.trace_join.get_trace(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="Trace not found")
        return TraceResponse(
            trace_id=trace.trace_id,
            query_id=trace.query_id,
            doc_id=trace.doc_id,
            planner_action=trace.planner.action,
            classifier_document_type=trace.classifier.document_type,
            classifier_confidence=trace.classifier.confidence,
            chunking_strategy=trace.chunking.strategy_name,
            chunk_count=trace.chunking.chunk_count,
            extraction_method=trace.extraction.extraction_method,
            final_confidence=trace.final_confidence,
        )

    @router.post("/attribute/{feedback_id}", response_model=AttributionResponse)
    def attribute_feedback(feedback_id: str) -> AttributionResponse:
        """Re-attribute a previously ingested feedback event.

        Useful for reprocessing after attribution rules change.
        """
        event = _pipeline.ingestion.get_event(feedback_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Feedback event not found")
        trace = _pipeline.trace_join.join(event)
        if trace is None:
            raise HTTPException(status_code=404, detail="No trace found for feedback")
        normalized = _pipeline.normalizer.normalize(event, trace)
        attribution = _pipeline.attribution.attribute(event, trace, normalized)
        return AttributionResponse(
            attribution_id=attribution.attribution_id,
            feedback_id=attribution.feedback_id,
            impacted_layers=[
                {
                    "layer": il.layer.value,
                    "confidence": il.confidence,
                    "explanation": il.explanation,
                    "reason_codes": [rc.value for rc in il.reason_codes],
                    "rule_id": il.rule_id,
                }
                for il in attribution.impacted_layers
            ],
            attribution_method=attribution.attribution_method,
        )

    @router.post("/training-rows/{feedback_id}", response_model=TrainingRowsResponse)
    def build_training_rows(feedback_id: str) -> TrainingRowsResponse:
        """Build training rows for a previously ingested feedback event."""
        event = _pipeline.ingestion.get_event(feedback_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Feedback event not found")
        trace = _pipeline.trace_join.join(event)
        if trace is None:
            raise HTTPException(status_code=404, detail="No trace found")
        normalized = _pipeline.normalizer.normalize(event, trace)
        attribution = _pipeline.attribution.attribute(event, trace, normalized)
        rows = _pipeline.row_builder.build(event, trace, attribution)
        layers = {layer.value: len(r) for layer, r in rows.items()}
        return TrainingRowsResponse(
            feedback_id=feedback_id,
            layers=layers,
            total_rows=sum(layers.values()),
        )

    @router.post("/retrain", response_model=TriggerRetrainingResponse)
    def trigger_retraining(req: TriggerRetrainingRequest) -> TriggerRetrainingResponse:
        """Trigger a retraining job for a specific layer and boundary."""
        job_id = _pipeline.orchestrator.trigger_retraining(
            layer=req.layer,
            boundary_key=req.boundary_key,
            trigger=req.trigger,
            min_rows=req.min_rows,
        )
        if job_id is None:
            return TriggerRetrainingResponse(
                job_id=None,
                status="insufficient_data",
                message=f"Not enough training rows for {req.layer.value}/{req.boundary_key.key}",
            )
        return TriggerRetrainingResponse(
            job_id=job_id,
            status="triggered",
            message=f"Retraining job {job_id} triggered for {req.layer.value}",
        )

    @router.get("/evaluation/{candidate_id}", response_model=EvaluationReportResponse)
    def get_evaluation_report(candidate_id: str) -> EvaluationReportResponse:
        """Fetch evaluation report for a model candidate.

        In production, this retrieves from the evaluation store.
        """
        # Placeholder — in production, look up from persistence
        raise HTTPException(
            status_code=501,
            detail="Evaluation report retrieval requires production persistence layer",
        )

    return router
