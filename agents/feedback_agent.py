"""Feedback Agent — collects user feedback and routes it to learning systems (§4.10).

.. deprecated::
    This module is deprecated in favor of the ``feedback_loop/`` subsystem,
    which provides multi-layer attribution, boundary-safe training isolation,
    layer-specific training row generation, and model lifecycle management.
    See ``docs/ENGINEERING_REVIEW.md`` Section 4 for migration guidance.

    **Expected removal: Phase 8** (Multi-Document & Security).

    The canonical feedback/retraining architecture is:
        feedback_loop/pipeline.py  — end-to-end orchestration
        feedback_loop/models.py    — typed domain models
        feedback_loop/attribution.py — 6-rule attribution engine
        feedback_loop/training_rows.py — layer-specific row builders
        feedback_loop/boundary.py  — boundary-safe isolation

    No new code should import this module. If you need feedback functionality,
    use ``FeedbackLoopPipeline.create_production()`` or ``create_test()``.

Responsibilities (legacy):
- Receive user feedback (positive/negative/correction) on query answers
- Validate and store feedback entries persistently
- Route feedback to classifier (pattern accuracy), preprocessor (quality scores),
  and retriever (relevance signals) for online learning
- Track feedback statistics for retraining triggers

Message types handled:
- ``feedback_request`` → expects FeedbackEntry in payload → returns FeedbackResult
"""

from __future__ import annotations

import warnings

warnings.warn(
    "agents.feedback_agent is deprecated. Use feedback_loop/ subsystem instead. "
    "See docs/ENGINEERING_REVIEW.md Section 4 for migration guidance.",
    DeprecationWarning,
    stacklevel=2,
)

import logging
import threading
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from cachetools import TTLCache

from agents.base import BaseAgent
from agents.contracts import (
    AgentMessage,
    FeedbackEntry,
    FeedbackResult,
    new_id,
)
from agents.message_bus import MessageBus
from agents.model_gateway import ModelGateway

logger = logging.getLogger(__name__)

# Valid feedback ratings
_VALID_RATINGS = {"positive", "negative", "correction"}


class FeedbackStore:
    """Thread-safe, bounded store for feedback entries.

    Persists entries in memory with optional database backing.
    Provides aggregation for retraining triggers.
    """

    def __init__(self, max_entries: int = 50_000, ttl_s: float = 86_400.0) -> None:
        self._entries: TTLCache[str, FeedbackEntry] = TTLCache(
            maxsize=max_entries, ttl=ttl_s,
        )
        self._by_doc: Dict[str, List[str]] = defaultdict(list)
        self._stats: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def store(self, entry: FeedbackEntry) -> None:
        """Store a feedback entry."""
        with self._lock:
            self._entries[entry.feedback_id] = entry
            self._by_doc[entry.doc_id].append(entry.feedback_id)
            self._stats[entry.rating] += 1

    def get(self, feedback_id: str) -> Optional[FeedbackEntry]:
        with self._lock:
            return self._entries.get(feedback_id)

    def get_by_doc(self, doc_id: str) -> List[FeedbackEntry]:
        """Return all feedback entries for a document."""
        with self._lock:
            ids = self._by_doc.get(doc_id, [])
            return [self._entries[fid] for fid in ids if fid in self._entries]

    def get_recent(self, limit: int = 100) -> List[FeedbackEntry]:
        """Return most recent feedback entries."""
        with self._lock:
            items = list(self._entries.values())
            return items[-limit:]

    @property
    def total_count(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def negative_count_since(self, since_count: int) -> int:
        """Count negative + correction entries since a given total."""
        with self._lock:
            total = self._stats.get("negative", 0) + self._stats.get("correction", 0)
            return max(0, total - since_count)


class FeedbackAgent(BaseAgent):
    """Collects user feedback and routes to learning subsystems.

    Feedback routing:
    - **Classifier**: Updates pattern accuracy when classification was wrong
    - **Preprocessor**: Adjusts quality scores for chunking strategies
    - **Retriever**: Logs relevance signals for future retrieval tuning

    The agent is purely deterministic — no LLM calls.
    """

    agent_name = "feedback"

    def __init__(
        self,
        bus: MessageBus,
        gateway: Optional[ModelGateway] = None,
        store: Optional[FeedbackStore] = None,
    ) -> None:
        self._store = store or FeedbackStore()
        super().__init__(bus, gateway)

    @property
    def store(self) -> FeedbackStore:
        return self._store

    def handle_message(self, message: AgentMessage) -> FeedbackResult:
        """Handle a feedback_request message.

        Expected payload: FeedbackEntry fields as dict.
        """
        start = time.monotonic()
        payload = message.payload

        entry = FeedbackEntry(
            feedback_id=payload.get("feedback_id") or new_id(),
            query_id=payload.get("query_id", message.query_id),
            doc_id=payload.get("doc_id", ""),
            rating=payload.get("rating", ""),
            comment=payload.get("comment", ""),
            correct_answer=payload.get("correct_answer", ""),
            cited_chunk_ids=payload.get("cited_chunk_ids", []),
            timestamp=payload.get("timestamp") or self._now_iso(),
        )

        return self.process_feedback(entry)

    def process_feedback(self, entry: FeedbackEntry) -> FeedbackResult:
        """Validate, store, and route a feedback entry."""
        start = time.monotonic()

        if entry.rating not in _VALID_RATINGS:
            raise ValueError(
                f"Invalid feedback rating '{entry.rating}'. "
                f"Must be one of: {_VALID_RATINGS}"
            )

        # Store the entry
        self._store.store(entry)

        # Route to learning subsystems
        routed_to: List[str] = []
        actions: List[str] = []

        if entry.rating in ("negative", "correction"):
            # Route to classifier for pattern accuracy update
            classifier_routed = self._route_to_classifier(entry)
            if classifier_routed:
                routed_to.append("classifier")
                actions.append("updated_pattern_accuracy")

            # Route to preprocessor for quality score adjustment
            preprocessor_routed = self._route_to_preprocessor(entry)
            if preprocessor_routed:
                routed_to.append("preprocessor")
                actions.append("adjusted_quality_score")

        # Always log retrieval relevance signal
        retriever_routed = self._route_to_retriever(entry)
        if retriever_routed:
            routed_to.append("retriever")
            actions.append("logged_relevance_signal")

        latency_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Feedback[%s]: rating=%s routed_to=%s actions=%s (%.0fms)",
            entry.feedback_id[:8],
            entry.rating,
            routed_to,
            actions,
            latency_ms,
        )

        return FeedbackResult(
            feedback_id=entry.feedback_id,
            query_id=entry.query_id,
            routed_to=routed_to,
            actions_taken=actions,
        )

    def _route_to_classifier(self, entry: FeedbackEntry) -> bool:
        """Route negative feedback to classifier's pattern store."""
        try:
            if "classifier" not in self.bus._handlers:
                return False

            self.bus.send(AgentMessage(
                message_id=new_id(),
                query_id=entry.query_id,
                from_agent=self.agent_name,
                to_agent="classifier",
                message_type="feedback_signal",
                payload={
                    "feedback_type": "answer_quality",
                    "rating": entry.rating,
                    "doc_id": entry.doc_id,
                    "correct_answer": entry.correct_answer,
                },
                timestamp=self._now_iso(),
            ))
            return True
        except Exception:
            logger.debug("Classifier not available for feedback routing", exc_info=True)
            return False

    def _route_to_preprocessor(self, entry: FeedbackEntry) -> bool:
        """Route negative feedback to preprocessor's outcome store."""
        try:
            if "preprocessor" not in self.bus._handlers:
                return False

            self.bus.send(AgentMessage(
                message_id=new_id(),
                query_id=entry.query_id,
                from_agent=self.agent_name,
                to_agent="preprocessor",
                message_type="feedback_signal",
                payload={
                    "feedback_type": "chunking_quality",
                    "rating": entry.rating,
                    "doc_id": entry.doc_id,
                    "cited_chunk_ids": entry.cited_chunk_ids,
                },
                timestamp=self._now_iso(),
            ))
            return True
        except Exception:
            logger.debug("Preprocessor not available for feedback routing", exc_info=True)
            return False

    def _route_to_retriever(self, entry: FeedbackEntry) -> bool:
        """Log retrieval relevance signal from feedback."""
        try:
            from agents.agent_eval import EvalCase, get_evaluator

            relevance = 1.0 if entry.rating == "positive" else 0.0
            get_evaluator().record(EvalCase(
                query_id=entry.query_id,
                agent_name="retriever",
                recall_at_k=relevance,
                precision_at_k=relevance,
            ))
            return True
        except Exception:
            logger.debug("Evaluator not available for relevance signal", exc_info=True)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return feedback statistics."""
        return {
            "total_entries": self._store.total_count,
            "by_rating": self._store.stats,
        }


# ── Singleton ────────────────────────────────────────────────────────

_singleton: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    """Return the global FeedbackStore singleton."""
    global _singleton
    if _singleton is None:
        _singleton = FeedbackStore()
    return _singleton
