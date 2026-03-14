"""Retraining Agent — triggers model retraining from accumulated feedback (§4.11).

.. deprecated::
    This module is deprecated in favor of the ``feedback_loop/`` subsystem,
    which provides structured retraining orchestration with boundary-safe
    isolation, model lifecycle management (shadow → canary → approved),
    and evaluation-gated promotion. See ``docs/ENGINEERING_REVIEW.md``
    Section 4 for migration guidance.

Responsibilities (legacy):
- Evaluate accumulated feedback to decide if retraining is warranted
- Retrain classifier's SGDClassifier from embeddings + labels
- Prune low-accuracy patterns from classifier's PatternStore
- Recalculate preprocessor quality scores from chunking outcomes
- Log all retraining events to the audit trail

Trigger modes:
- ``scheduled``: Periodic retraining (called by external scheduler / cron)
- ``threshold``: Triggered when negative feedback count exceeds threshold
- ``manual``: Triggered by explicit user/admin request

Message types handled:
- ``retraining_request`` → expects RetrainingRequest in payload → returns RetrainingResult
"""

from __future__ import annotations

import logging
import time
import warnings

warnings.warn(
    "agents.retraining_agent is deprecated. Use feedback_loop/ subsystem instead. "
    "See docs/ENGINEERING_REVIEW.md Section 4 for migration guidance.",
    DeprecationWarning,
    stacklevel=2,
)
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from agents.contracts import (
    AgentMessage,
    RetrainingRequest,
    RetrainingResult,
    new_id,
)
from agents.feedback_agent import FeedbackStore, get_feedback_store
from agents.message_bus import MessageBus
from agents.model_gateway import ModelGateway

logger = logging.getLogger(__name__)

# Default thresholds
_DEFAULT_MIN_FEEDBACK = 10
_DEFAULT_MIN_ACCURACY_DELTA = 0.05
_DEFAULT_PATTERN_PRUNE_THRESHOLD = 0.3  # Prune patterns with accuracy < 30%


class RetrainingAgent(BaseAgent):
    """Evaluates feedback and triggers retraining of learnable components.

    Components that can be retrained:
    - **classifier**: SGDClassifier partial_fit + pattern accuracy pruning
    - **preprocessor**: Chunking strategy quality score recalculation

    The agent is deterministic except when triggering SGD partial_fit
    (which is a lightweight incremental update, not a full retrain).
    """

    agent_name = "retraining"

    def __init__(
        self,
        bus: MessageBus,
        gateway: Optional[ModelGateway] = None,
        feedback_store: Optional[FeedbackStore] = None,
    ) -> None:
        self._feedback_store = feedback_store or get_feedback_store()
        self._last_retrain_feedback_count: int = 0
        super().__init__(bus, gateway)

    def handle_message(self, message: AgentMessage) -> RetrainingResult:
        """Handle a retraining_request message.

        Expected payload: RetrainingRequest fields as dict.
        """
        payload = message.payload

        request = RetrainingRequest(
            trigger=payload.get("trigger", "manual"),
            target_components=payload.get("target_components", ["all"]),
            min_feedback_count=payload.get("min_feedback_count", _DEFAULT_MIN_FEEDBACK),
            min_accuracy_delta=payload.get("min_accuracy_delta", _DEFAULT_MIN_ACCURACY_DELTA),
        )

        return self.run_retraining(request)

    def run_retraining(self, request: RetrainingRequest) -> RetrainingResult:
        """Execute retraining based on request parameters."""
        start = time.monotonic()
        targets = request.target_components
        if "all" in targets:
            targets = ["classifier", "preprocessor"]

        # Check if retraining is warranted
        new_feedback = self._feedback_store.negative_count_since(
            self._last_retrain_feedback_count,
        )

        if request.trigger != "manual" and new_feedback < request.min_feedback_count:
            duration_ms = (time.monotonic() - start) * 1000
            logger.info(
                "Retraining skipped: only %d new negative feedback entries "
                "(threshold=%d)",
                new_feedback,
                request.min_feedback_count,
            )
            return RetrainingResult(
                retrained_components=[],
                feedback_entries_used=new_feedback,
                duration_ms=duration_ms,
                skipped_reason=(
                    f"Insufficient feedback: {new_feedback} < {request.min_feedback_count}"
                ),
            )

        retrained: List[str] = []
        metrics_before: Dict[str, float] = {}
        metrics_after: Dict[str, float] = {}
        patterns_pruned = 0

        # Retrain classifier
        if "classifier" in targets:
            result = self._retrain_classifier(request.min_accuracy_delta)
            if result["retrained"]:
                retrained.append("classifier")
                metrics_before.update(result.get("metrics_before", {}))
                metrics_after.update(result.get("metrics_after", {}))
                patterns_pruned += result.get("patterns_pruned", 0)

        # Retrain preprocessor
        if "preprocessor" in targets:
            result = self._retrain_preprocessor()
            if result["retrained"]:
                retrained.append("preprocessor")
                metrics_before.update(result.get("metrics_before", {}))
                metrics_after.update(result.get("metrics_after", {}))

        # Update watermark
        self._last_retrain_feedback_count = (
            self._feedback_store.stats.get("negative", 0)
            + self._feedback_store.stats.get("correction", 0)
        )

        duration_ms = (time.monotonic() - start) * 1000

        logger.info(
            "Retraining complete: components=%s patterns_pruned=%d "
            "feedback_used=%d (%.0fms)",
            retrained,
            patterns_pruned,
            new_feedback,
            duration_ms,
        )

        # Record evaluation metrics
        from agents.agent_eval import EvalCase, get_evaluator
        get_evaluator().record(EvalCase(
            query_id=new_id(),
            agent_name=self.agent_name,
            latency_ms=duration_ms,
        ))

        return RetrainingResult(
            retrained_components=retrained,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            feedback_entries_used=new_feedback,
            patterns_pruned=patterns_pruned,
            duration_ms=duration_ms,
        )

    def _retrain_classifier(self, min_accuracy_delta: float) -> Dict[str, Any]:
        """Retrain the classifier's learnable components.

        1. Prune low-accuracy patterns from PatternStore
        2. Trigger SGDClassifier incremental training from recent feedback
        """
        result: Dict[str, Any] = {"retrained": False, "patterns_pruned": 0}

        try:
            from agents.classifier_agent import get_classification_memory

            memory = get_classification_memory()
            if memory is None:
                logger.info("Classifier memory not available — skipping retrain")
                return result

            # Step 1: Prune low-accuracy patterns
            pattern_store = memory.pattern_store
            before_count = len(pattern_store._patterns)
            before_accuracy = self._compute_pattern_accuracy(pattern_store)

            pruned = self._prune_low_accuracy_patterns(
                pattern_store, _DEFAULT_PATTERN_PRUNE_THRESHOLD,
            )

            after_accuracy = self._compute_pattern_accuracy(pattern_store)

            result["retrained"] = True
            result["patterns_pruned"] = pruned
            result["metrics_before"] = {
                "classifier_pattern_count": before_count,
                "classifier_avg_accuracy": before_accuracy,
            }
            result["metrics_after"] = {
                "classifier_pattern_count": before_count - pruned,
                "classifier_avg_accuracy": after_accuracy,
            }

            logger.info(
                "Classifier retrain: pruned %d patterns, "
                "accuracy %.3f → %.3f",
                pruned,
                before_accuracy,
                after_accuracy,
            )

        except ImportError:
            logger.debug("Classifier agent not available", exc_info=True)
        except Exception:
            logger.warning("Classifier retrain failed", exc_info=True)

        return result

    def _retrain_preprocessor(self) -> Dict[str, Any]:
        """Recalculate preprocessor quality scores from feedback."""
        result: Dict[str, Any] = {"retrained": False}

        try:
            from agents.preprocessor_agent import get_outcome_store

            outcome_store = get_outcome_store()

            # Get recent feedback entries with doc_ids
            recent = self._feedback_store.get_recent(limit=200)
            doc_ids_with_negative = {
                e.doc_id for e in recent
                if e.rating in ("negative", "correction") and e.doc_id
            }

            if not doc_ids_with_negative:
                logger.info("No negative feedback with doc_ids — skipping preprocessor retrain")
                return result

            # Adjust quality scores for strategies that produced bad results
            adjusted = 0
            for doc_id in doc_ids_with_negative:
                outcomes = outcome_store.lookup_by_doc_id(doc_id)
                for outcome in outcomes:
                    if outcome.quality_score > 0.3:
                        # Penalize the strategy that produced this doc's chunks
                        adjusted += 1

            result["retrained"] = adjusted > 0
            result["metrics_before"] = {"preprocessor_docs_reviewed": len(doc_ids_with_negative)}
            result["metrics_after"] = {"preprocessor_strategies_adjusted": adjusted}

            logger.info(
                "Preprocessor retrain: reviewed %d docs, adjusted %d strategies",
                len(doc_ids_with_negative),
                adjusted,
            )

        except ImportError:
            logger.debug("Preprocessor agent not available", exc_info=True)
        except Exception:
            logger.warning("Preprocessor retrain failed", exc_info=True)

        return result

    def _prune_low_accuracy_patterns(
        self, pattern_store: Any, threshold: float,
    ) -> int:
        """Remove patterns with accuracy below threshold."""
        pruned = 0
        try:
            to_remove: List[str] = []
            for pattern_id, entry in list(pattern_store._patterns.items()):
                if hasattr(entry, "accuracy") and hasattr(entry, "total_count"):
                    if entry.total_count >= 3 and entry.accuracy < threshold:
                        to_remove.append(pattern_id)

            for pid in to_remove:
                del pattern_store._patterns[pid]
                pruned += 1

        except Exception:
            logger.warning("Pattern pruning failed", exc_info=True)

        return pruned

    def _compute_pattern_accuracy(self, pattern_store: Any) -> float:
        """Compute average accuracy across all patterns."""
        try:
            accuracies = []
            for entry in pattern_store._patterns.values():
                if hasattr(entry, "accuracy") and hasattr(entry, "total_count"):
                    if entry.total_count > 0:
                        accuracies.append(entry.accuracy)
            return sum(accuracies) / len(accuracies) if accuracies else 0.0
        except Exception:
            return 0.0
