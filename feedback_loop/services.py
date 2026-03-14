"""Concrete service implementations — In-memory for tests, SQL-friendly for production.

These implementations provide the core feedback loop pipeline:
    Ingest → Join → Normalize → Attribute → Build → Guard → Orchestrate → Evaluate → Promote
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from feedback_loop.attribution import RuleBasedAttributionEngine
from feedback_loop.boundary import DefaultBoundaryPolicyGuard
from feedback_loop.interfaces import (
    BoundaryPolicyGuard,
    FeedbackIngestionService,
    ModelEvaluator,
    ModelPromotionController,
    RetrainingOrchestrator,
    TraceJoinService,
)
from feedback_loop.models import (
    BoundaryKey,
    CalibrationTrainingRow,
    DecisionLayer,
    EvaluationReport,
    FeedbackAttribution,
    FeedbackEvent,
    FeedbackRating,
    LayerMetrics,
    ModelCandidate,
    ModelStage,
    RetrainingTrigger,
    PredictionTrace,
    _new_id,
    _utcnow,
)

logger = logging.getLogger(__name__)


# ── Feedback Ingestion ───────────────────────────────────────────────


class InMemoryFeedbackIngestionService(FeedbackIngestionService):
    """In-memory feedback store.  Replace with DB-backed implementation in production."""

    def __init__(self) -> None:
        self._events: Dict[str, FeedbackEvent] = {}

    def ingest(self, event: FeedbackEvent) -> str:
        """Validate and store a feedback event."""
        if not event.feedback_id:
            event = event.model_copy(update={"feedback_id": _new_id()})
        if not event.boundary_key:
            raise ValueError("FeedbackEvent must carry a boundary_key")
        self._events[event.feedback_id] = event
        logger.info("Ingested feedback %s (rating=%s)", event.feedback_id, event.rating.value)
        return event.feedback_id

    def get_event(self, feedback_id: str) -> Optional[FeedbackEvent]:
        return self._events.get(feedback_id)

    def list_events(
        self,
        boundary_key: Optional[BoundaryKey] = None,
        doc_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[FeedbackEvent]:
        events = list(self._events.values())
        if boundary_key:
            events = [e for e in events if e.boundary_key.key == boundary_key.key]
        if doc_id:
            events = [e for e in events if e.doc_id == doc_id]
        return sorted(events, key=lambda e: e.created_at, reverse=True)[:limit]


# ── Trace Join ───────────────────────────────────────────────────────


class InMemoryTraceJoinService(TraceJoinService):
    """In-memory trace store.  Replace with DB-backed implementation in production."""

    def __init__(self) -> None:
        self._traces: Dict[str, PredictionTrace] = {}
        self._by_query: Dict[str, str] = {}  # query_id → trace_id

    def store_trace(self, trace: PredictionTrace) -> str:
        self._traces[trace.trace_id] = trace
        if trace.query_id:
            self._by_query[trace.query_id] = trace.trace_id
        return trace.trace_id

    def get_trace(self, trace_id: str) -> Optional[PredictionTrace]:
        return self._traces.get(trace_id)

    def get_trace_by_query(self, query_id: str) -> Optional[PredictionTrace]:
        trace_id = self._by_query.get(query_id)
        if trace_id:
            return self._traces.get(trace_id)
        return None

    def join(self, event: FeedbackEvent) -> Optional[PredictionTrace]:
        """Join feedback with its trace by trace_id, then query_id."""
        if event.trace_id:
            trace = self.get_trace(event.trace_id)
            if trace:
                return trace
        if event.query_id:
            return self.get_trace_by_query(event.query_id)
        return None


# ── Retraining Orchestrator ──────────────────────────────────────────


class InMemoryRetrainingOrchestrator(RetrainingOrchestrator):
    """In-memory retraining orchestrator.

    Groups rows into datasets per (layer, boundary_key).
    Triggers retraining when row count exceeds min_rows.
    """

    def __init__(
        self,
        boundary_guard: Optional[BoundaryPolicyGuard] = None,
    ) -> None:
        # (layer, boundary_key_str) → list of rows
        self._datasets: Dict[tuple[str, str], List[Any]] = defaultdict(list)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._boundary_guard = boundary_guard or DefaultBoundaryPolicyGuard()

    def submit_rows(
        self,
        layer: DecisionLayer,
        rows: List[Any],
        boundary_key: BoundaryKey,
    ) -> str:
        """Submit training rows for a layer.  Returns dataset_id."""
        dataset_key = (layer.value, boundary_key.key)
        # Validate boundary on each row
        valid_rows = [
            r for r in rows
            if self._boundary_guard.validate_row(r, boundary_key)
        ]
        self._datasets[dataset_key].extend(valid_rows)
        dataset_id = f"ds-{layer.value}-{boundary_key.key}-{len(self._datasets[dataset_key])}"
        logger.info(
            "Submitted %d/%d rows for %s/%s (dataset: %s)",
            len(valid_rows), len(rows), layer.value, boundary_key.key, dataset_id,
        )
        return dataset_id

    def trigger_retraining(
        self,
        layer: DecisionLayer,
        boundary_key: BoundaryKey,
        trigger: RetrainingTrigger = RetrainingTrigger.THRESHOLD,
        min_rows: int = 10,
    ) -> Optional[str]:
        """Trigger a retraining job if sufficient data exists."""
        dataset_key = (layer.value, boundary_key.key)
        rows = self._datasets.get(dataset_key, [])
        if len(rows) < min_rows:
            logger.info(
                "Insufficient data for %s/%s: %d < %d",
                layer.value, boundary_key.key, len(rows), min_rows,
            )
            return None

        job_id = _new_id()
        self._jobs[job_id] = {
            "job_id": job_id,
            "layer": layer.value,
            "boundary_key": boundary_key.key,
            "trigger": trigger.value,
            "row_count": len(rows),
            "status": "pending",
            "created_at": _utcnow().isoformat(),
        }
        logger.info(
            "Triggered retraining job %s for %s/%s (%d rows)",
            job_id, layer.value, boundary_key.key, len(rows),
        )
        return job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        return self._jobs.get(job_id, {"status": "not_found"})

    def get_dataset_size(self, layer: DecisionLayer, boundary_key: BoundaryKey) -> int:
        """Get number of accumulated rows for a layer+boundary."""
        return len(self._datasets.get((layer.value, boundary_key.key), []))


# ── Model Evaluator ──────────────────────────────────────────────────


class DefaultModelEvaluator(ModelEvaluator):
    """Compares candidate model metrics against baseline.

    Uses primary metric (accuracy by default) for comparison.
    Supports configurable significance threshold.
    """

    def __init__(self, min_improvement: float = 0.01) -> None:
        self._min_improvement = min_improvement

    def evaluate(
        self,
        candidate: ModelCandidate,
        baseline_metrics: Dict[str, float],
        validation_data: List[Any],
    ) -> EvaluationReport:
        """Evaluate candidate against baseline."""
        # Build baseline LayerMetrics
        baseline = LayerMetrics(
            accuracy=baseline_metrics.get("accuracy", 0.0),
            precision=baseline_metrics.get("precision", 0.0),
            recall=baseline_metrics.get("recall", 0.0),
            f1=baseline_metrics.get("f1", 0.0),
            ece=baseline_metrics.get("ece", 0.0),
            sample_count=baseline_metrics.get("sample_count", 0),
        )

        # Candidate metrics (in production, run model on validation_data)
        # For now, scaffold with placeholder
        candidate_metrics = LayerMetrics(
            accuracy=baseline_metrics.get("accuracy", 0.0),
            sample_count=len(validation_data),
        )

        delta = candidate_metrics.accuracy - baseline.accuracy
        is_sig = abs(delta) >= self._min_improvement
        rec = "promote" if delta >= self._min_improvement else "reject"

        return EvaluationReport(
            candidate_id=candidate.candidate_id,
            layer=candidate.layer,
            boundary_key=candidate.boundary_key,
            baseline_metrics=baseline,
            candidate_metrics=candidate_metrics,
            improvement_delta=delta,
            is_statistically_significant=is_sig,
            recommendation=rec,
            explanation=f"Accuracy delta: {delta:+.4f} (min: {self._min_improvement})",
        )


# ── Model Promotion Controller ───────────────────────────────────────


class InMemoryModelPromotionController(ModelPromotionController):
    """In-memory model lifecycle controller.

    Stages: shadow → canary → approved / rejected → rollback_ready.
    """

    def __init__(self) -> None:
        self._candidates: Dict[str, ModelCandidate] = {}
        # (layer, boundary_key) → active candidate_id
        self._active: Dict[tuple[str, str], str] = {}

    def register(self, candidate: ModelCandidate) -> None:
        """Register a new model candidate."""
        self._candidates[candidate.candidate_id] = candidate

    def promote(self, candidate_id: str, target_stage: ModelStage) -> ModelCandidate:
        """Promote a candidate to the target lifecycle stage."""
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Candidate {candidate_id} not found")

        _VALID_TRANSITIONS = {
            ModelStage.SHADOW: {ModelStage.CANARY, ModelStage.REJECTED},
            ModelStage.CANARY: {ModelStage.APPROVED, ModelStage.REJECTED},
            ModelStage.APPROVED: {ModelStage.ROLLBACK_READY},
            ModelStage.REJECTED: set(),
            ModelStage.ROLLBACK_READY: {ModelStage.APPROVED},
        }

        allowed = _VALID_TRANSITIONS.get(candidate.stage, set())
        if target_stage not in allowed:
            raise ValueError(
                f"Invalid transition: {candidate.stage.value} → {target_stage.value}. "
                f"Allowed: {', '.join(s.value for s in allowed)}"
            )

        updated = candidate.model_copy(update={"stage": target_stage})
        self._candidates[candidate_id] = updated

        if target_stage == ModelStage.APPROVED:
            key = (updated.layer.value, updated.boundary_key.key)
            self._active[key] = candidate_id

        logger.info(
            "Model %s promoted: %s → %s",
            candidate_id, candidate.stage.value, target_stage.value,
        )
        return updated

    def rollback(self, candidate_id: str) -> ModelCandidate:
        """Rollback: set to ROLLBACK_READY and deactivate."""
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Candidate {candidate_id} not found")

        updated = candidate.model_copy(update={"stage": ModelStage.ROLLBACK_READY})
        self._candidates[candidate_id] = updated

        # Remove from active if it was active
        key = (updated.layer.value, updated.boundary_key.key)
        if self._active.get(key) == candidate_id:
            del self._active[key]

        return updated

    def get_active_model(
        self,
        layer: DecisionLayer,
        boundary_key: BoundaryKey,
    ) -> Optional[ModelCandidate]:
        key = (layer.value, boundary_key.key)
        cid = self._active.get(key)
        if cid:
            return self._candidates.get(cid)
        return None
