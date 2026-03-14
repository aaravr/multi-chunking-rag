"""Concrete service implementations — In-memory for tests, DB-backed for production.

These implementations provide the core feedback loop pipeline:
    Ingest → Join → Normalize → Attribute → Build → Guard → Orchestrate → Evaluate → Promote

In-memory implementations are suitable for unit tests and local development.
DB-backed (Postgres*) implementations use migration 009 tables for production persistence.
"""

from __future__ import annotations

import json
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
    """In-memory retraining orchestrator (test/local-dev only).

    Groups rows into datasets per (layer, boundary_key).
    Triggers retraining when row count exceeds min_rows.

    Boundary validation is NOT performed here — the pipeline's Guard step
    (FeedbackLoopPipeline.process, step 6) is the sole boundary gate.
    """

    def __init__(
        self,
        boundary_guard: Optional[BoundaryPolicyGuard] = None,
    ) -> None:
        # (layer, boundary_key_str) → list of rows
        self._datasets: Dict[tuple[str, str], List[Any]] = defaultdict(list)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        # Retained for interface compatibility but not used for validation
        self._boundary_guard = boundary_guard or DefaultBoundaryPolicyGuard()

    def submit_rows(
        self,
        layer: DecisionLayer,
        rows: List[Any],
        boundary_key: BoundaryKey,
    ) -> str:
        """Submit training rows for a layer.  Returns dataset_id.

        Boundary validation is NOT performed here — the pipeline's Guard
        step (FeedbackLoopPipeline.process, step 6) is the sole boundary
        gate.  Rows arriving here have already been validated.
        """
        dataset_key = (layer.value, boundary_key.key)
        self._datasets[dataset_key].extend(rows)
        dataset_id = f"ds-{layer.value}-{boundary_key.key}-{len(self._datasets[dataset_key])}"
        logger.info(
            "Submitted %d rows for %s/%s (dataset: %s)",
            len(rows), layer.value, boundary_key.key, dataset_id,
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


# ══════════════════════════════════════════════════════════════════════
# DB-Backed Implementations (migration 009 tables)
# ══════════════════════════════════════════════════════════════════════
#
# These implementations use PostgreSQL for durable persistence.
# They accept a connection-factory callable that returns a context-managed
# connection (matching the pattern in storage/db_pool.py).
# ══════════════════════════════════════════════════════════════════════


class PostgresFeedbackIngestionService(FeedbackIngestionService):
    """DB-backed feedback ingestion service using feedback_events table (migration 009)."""

    def __init__(self, get_conn: Callable) -> None:
        self._get_conn = get_conn

    def ingest(self, event: FeedbackEvent) -> str:
        feedback_id = event.feedback_id or _new_id()
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO feedback_events (
                        feedback_id, trace_id, query_id, doc_id, user_id,
                        boundary_client, boundary_division, boundary_jurisdiction,
                        rating, comment, correct_answer,
                        correct_document_type, correct_classification_label,
                        correct_evidence_spans, correct_field_values,
                        processing_path_override, channel, cited_chunk_ids
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s, %s,
                        %s, %s, %s
                    )
                    ON CONFLICT (feedback_id) DO NOTHING
                    """,
                    (
                        feedback_id,
                        event.trace_id or None,
                        event.query_id or None,
                        event.doc_id or None,
                        event.user_id or None,
                        event.boundary_key.client,
                        event.boundary_key.division,
                        event.boundary_key.jurisdiction,
                        event.rating.value,
                        event.comment or None,
                        event.correct_answer or None,
                        event.correct_document_type or None,
                        event.correct_classification_label or None,
                        event.correct_evidence_spans or None,
                        json.dumps(event.correct_field_values) if event.correct_field_values else None,
                        event.processing_path_override or None,
                        event.channel.value if event.channel else "api",
                        event.cited_chunk_ids or None,
                    ),
                )
            conn.commit()
        logger.info("Persisted feedback %s (rating=%s)", feedback_id, event.rating.value)
        return feedback_id

    def get_event(self, feedback_id: str) -> Optional[FeedbackEvent]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT feedback_id, trace_id, query_id, doc_id, user_id,
                           boundary_client, boundary_division, boundary_jurisdiction,
                           rating, comment, correct_answer,
                           correct_document_type, correct_classification_label,
                           correct_evidence_spans, correct_field_values,
                           processing_path_override, channel, cited_chunk_ids, created_at
                    FROM feedback_events WHERE feedback_id = %s
                    """,
                    (feedback_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return self._row_to_event(row)

    def list_events(
        self,
        boundary_key: Optional[BoundaryKey] = None,
        doc_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[FeedbackEvent]:
        clauses = []
        params: list = []
        if boundary_key:
            clauses.append("boundary_client = %s AND boundary_division = %s AND boundary_jurisdiction = %s")
            params.extend([boundary_key.client, boundary_key.division, boundary_key.jurisdiction])
        if doc_id:
            clauses.append("doc_id = %s")
            params.append(doc_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT feedback_id, trace_id, query_id, doc_id, user_id,
                           boundary_client, boundary_division, boundary_jurisdiction,
                           rating, comment, correct_answer,
                           correct_document_type, correct_classification_label,
                           correct_evidence_spans, correct_field_values,
                           processing_path_override, channel, cited_chunk_ids, created_at
                    FROM feedback_events{where}
                    ORDER BY created_at DESC LIMIT %s
                    """,
                    params,
                )
                return [self._row_to_event(r) for r in cur.fetchall()]

    @staticmethod
    def _row_to_event(row: tuple) -> FeedbackEvent:
        return FeedbackEvent(
            feedback_id=str(row[0]),
            trace_id=str(row[1]) if row[1] else "",
            query_id=row[2] or "",
            doc_id=str(row[3]) if row[3] else "",
            user_id=row[4] or "",
            boundary_key=BoundaryKey(client=row[5], division=row[6], jurisdiction=row[7]),
            rating=FeedbackRating(row[8]),
            comment=row[9] or "",
            correct_answer=row[10] or "",
            correct_document_type=row[11] or "",
            correct_classification_label=row[12] or "",
            correct_evidence_spans=row[13] or [],
            correct_field_values=row[14] or {},
            processing_path_override=row[15] or "",
            cited_chunk_ids=row[17] or [],
        )


class PostgresTraceJoinService(TraceJoinService):
    """DB-backed trace join service using prediction_traces table (migration 009)."""

    def __init__(self, get_conn: Callable) -> None:
        self._get_conn = get_conn

    def store_trace(self, trace: PredictionTrace) -> str:
        trace_id = trace.trace_id or _new_id()
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prediction_traces (
                        trace_id, query_id, doc_id, user_id,
                        boundary_client, boundary_division, boundary_jurisdiction,
                        planner_decision, classifier_decision,
                        chunking_decision, extraction_decision, transformation_decision,
                        final_answer, final_confidence, citations,
                        model_versions, total_latency_ms
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s
                    )
                    ON CONFLICT (trace_id) DO NOTHING
                    """,
                    (
                        trace_id,
                        trace.query_id or None,
                        trace.doc_id or None,
                        trace.user_id or None,
                        trace.boundary_key.client if trace.boundary_key else "",
                        trace.boundary_key.division if trace.boundary_key else "",
                        trace.boundary_key.jurisdiction if trace.boundary_key else "",
                        json.dumps(trace.planner_decision.model_dump() if trace.planner_decision else {}),
                        json.dumps(trace.classifier_decision.model_dump() if trace.classifier_decision else {}),
                        json.dumps(trace.chunking_decision.model_dump() if trace.chunking_decision else {}),
                        json.dumps(trace.extraction_decision.model_dump() if trace.extraction_decision else {}),
                        json.dumps(trace.transformation_decision.model_dump() if trace.transformation_decision else {}),
                        trace.final_answer or None,
                        trace.final_confidence,
                        trace.citations or None,
                        json.dumps(trace.model_versions or {}),
                        trace.total_latency_ms,
                    ),
                )
            conn.commit()
        return trace_id

    def get_trace(self, trace_id: str) -> Optional[PredictionTrace]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT trace_id, query_id, doc_id, user_id,
                           boundary_client, boundary_division, boundary_jurisdiction,
                           final_answer, final_confidence, citations,
                           model_versions, total_latency_ms
                    FROM prediction_traces WHERE trace_id = %s
                    """,
                    (trace_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return self._row_to_trace(row)

    def get_trace_by_query(self, query_id: str) -> Optional[PredictionTrace]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT trace_id, query_id, doc_id, user_id,
                           boundary_client, boundary_division, boundary_jurisdiction,
                           final_answer, final_confidence, citations,
                           model_versions, total_latency_ms
                    FROM prediction_traces WHERE query_id = %s
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (query_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return self._row_to_trace(row)

    def join(self, event: FeedbackEvent) -> Optional[PredictionTrace]:
        if event.trace_id:
            trace = self.get_trace(event.trace_id)
            if trace:
                return trace
        if event.query_id:
            return self.get_trace_by_query(event.query_id)
        return None

    @staticmethod
    def _row_to_trace(row: tuple) -> PredictionTrace:
        return PredictionTrace(
            trace_id=str(row[0]),
            query_id=row[1] or "",
            doc_id=str(row[2]) if row[2] else "",
            user_id=row[3] or "",
            boundary_key=BoundaryKey(client=row[4], division=row[5], jurisdiction=row[6]),
            final_answer=row[7] or "",
            final_confidence=row[8] or 0.0,
            citations=row[9] or [],
            model_versions=row[10] or {},
            total_latency_ms=row[11] or 0.0,
        )


class PostgresRetrainingOrchestrator(RetrainingOrchestrator):
    """DB-backed retraining orchestrator using retraining_jobs + training_rows_* tables.

    Persists both training rows (to layer-specific tables) and retraining job
    metadata to the database for durability and auditability.

    Boundary validation is NOT performed here — the pipeline's Guard step
    (FeedbackLoopPipeline.process, step 6) is the sole boundary gate.
    """

    # Maps DecisionLayer to the corresponding training row table name
    _TABLE_MAP: Dict[DecisionLayer, str] = {
        DecisionLayer.PLANNING: "training_rows_planner",
        DecisionLayer.CLASSIFICATION: "training_rows_classifier",
        DecisionLayer.CHUNKING: "training_rows_chunking",
        DecisionLayer.EXTRACTION: "training_rows_extraction",
        DecisionLayer.CALIBRATION: "training_rows_calibration",
    }

    def __init__(
        self,
        get_conn: Callable,
        boundary_guard: Optional[BoundaryPolicyGuard] = None,
    ) -> None:
        self._get_conn = get_conn
        self._boundary_guard = boundary_guard or DefaultBoundaryPolicyGuard()

    def submit_rows(
        self,
        layer: DecisionLayer,
        rows: List[Any],
        boundary_key: BoundaryKey,
    ) -> str:
        """Persist training rows to the layer-specific DB table.

        Boundary validation is NOT performed here — the pipeline's Guard
        step is the sole boundary gate.
        """
        table = self._TABLE_MAP.get(layer)
        if not table:
            raise ValueError(f"No training row table for layer {layer.value}")

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                for row in rows:
                    self._persist_row(cur, table, layer, row, boundary_key)
            conn.commit()

        dataset_id = f"ds-{layer.value}-{boundary_key.key}-{_new_id()[:8]}"
        logger.info(
            "Persisted %d rows to %s for %s/%s (dataset: %s)",
            len(rows), table, layer.value, boundary_key.key, dataset_id,
        )
        return dataset_id

    def _persist_row(
        self,
        cur: Any,
        table: str,
        layer: DecisionLayer,
        row: Any,
        boundary_key: BoundaryKey,
    ) -> None:
        """Insert a single training row into its layer-specific table."""
        row_id = _new_id()
        source_ids = row.source_feedback_ids if hasattr(row, "source_feedback_ids") else []

        if layer == DecisionLayer.PLANNING:
            cur.execute(
                f"""INSERT INTO {table} (
                    row_id, source_feedback_ids, boundary_client,
                    boundary_division, boundary_jurisdiction, accuracy
                ) VALUES (%s, %s, %s, %s, %s, %s)""",
                (row_id, source_ids, boundary_key.client,
                 boundary_key.division, boundary_key.jurisdiction,
                 getattr(row, "accuracy", 0.0)),
            )
        elif layer == DecisionLayer.CLASSIFICATION:
            cur.execute(
                f"""INSERT INTO {table} (
                    row_id, source_feedback_ids, boundary_client,
                    boundary_division, boundary_jurisdiction, is_correct
                ) VALUES (%s, %s, %s, %s, %s, %s)""",
                (row_id, source_ids, boundary_key.client,
                 boundary_key.division, boundary_key.jurisdiction,
                 getattr(row, "is_correct", False)),
            )
        elif layer == DecisionLayer.CHUNKING:
            cur.execute(
                f"""INSERT INTO {table} (
                    row_id, source_feedback_ids, boundary_client,
                    boundary_division, boundary_jurisdiction, evidence_recall
                ) VALUES (%s, %s, %s, %s, %s, %s)""",
                (row_id, source_ids, boundary_key.client,
                 boundary_key.division, boundary_key.jurisdiction,
                 getattr(row, "evidence_recall", 0.0)),
            )
        elif layer == DecisionLayer.EXTRACTION:
            cur.execute(
                f"""INSERT INTO {table} (
                    row_id, source_feedback_ids, boundary_client,
                    boundary_division, boundary_jurisdiction,
                    field_name, is_correct
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (row_id, source_ids, boundary_key.client,
                 boundary_key.division, boundary_key.jurisdiction,
                 getattr(row, "field_name", ""), getattr(row, "is_correct", False)),
            )
        elif layer == DecisionLayer.CALIBRATION:
            cur.execute(
                f"""INSERT INTO {table} (
                    row_id, source_feedback_ids, boundary_client,
                    boundary_division, boundary_jurisdiction,
                    is_correct, confidence_bucket
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (row_id, source_ids, boundary_key.client,
                 boundary_key.division, boundary_key.jurisdiction,
                 getattr(row, "is_correct", False),
                 getattr(row, "confidence_bucket", "0.0-0.1")),
            )

    def trigger_retraining(
        self,
        layer: DecisionLayer,
        boundary_key: BoundaryKey,
        trigger: RetrainingTrigger = RetrainingTrigger.THRESHOLD,
        min_rows: int = 10,
    ) -> Optional[str]:
        table = self._TABLE_MAP.get(layer)
        if not table:
            return None

        # Count rows from DB, not in-memory
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE boundary_client = %s",
                    (boundary_key.client,),
                )
                row_count = cur.fetchone()[0]

        if row_count < min_rows:
            logger.info(
                "Insufficient data for %s/%s: %d < %d",
                layer.value, boundary_key.key, row_count, min_rows,
            )
            return None

        job_id = _new_id()
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO retraining_jobs (
                        job_id, layer, boundary_client, boundary_division,
                        boundary_jurisdiction, trigger_type, row_count, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending')
                    """,
                    (
                        job_id, layer.value,
                        boundary_key.client, boundary_key.division,
                        boundary_key.jurisdiction,
                        trigger.value, row_count,
                    ),
                )
            conn.commit()
        logger.info(
            "Persisted retraining job %s for %s/%s (%d rows)",
            job_id, layer.value, boundary_key.key, row_count,
        )
        return job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT job_id, layer, status, row_count, created_at FROM retraining_jobs WHERE job_id = %s",
                    (job_id,),
                )
                row = cur.fetchone()
                if not row:
                    return {"status": "not_found"}
                return {
                    "job_id": str(row[0]),
                    "layer": row[1],
                    "status": row[2],
                    "row_count": row[3],
                    "created_at": row[4].isoformat() if row[4] else None,
                }

    def get_dataset_size(self, layer: DecisionLayer, boundary_key: BoundaryKey) -> int:
        """Count persisted training rows from DB."""
        table = self._TABLE_MAP.get(layer)
        if not table:
            return 0
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE boundary_client = %s",
                    (boundary_key.client,),
                )
                return cur.fetchone()[0]
