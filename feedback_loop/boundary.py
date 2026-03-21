"""Boundary Policy Guard — Training Isolation Enforcement.

Enforces boundary-safe learning: B = (client, division, jurisdiction).

Default rules:
1. Every feedback event must carry or derive boundary_key
2. Every generated training row must carry boundary_key
3. No cross-boundary training data mixing by default
4. No global reusable raw cross-client document text
5. Only approved sanitized feature sharing is allowed
6. Optional hierarchical sharing: θ_B = θ_shared + δ_B

Boundary validation is two-tiered:
- **Ingestion-time validation:** Accepts events with partial boundary keys
  (empty division/jurisdiction). Logs warnings but stores the event.
- **Trainable-row validation:** Requires either full granularity OR an
  explicit ``reduced_granularity_approved`` flag on the BoundaryKey.
  Training rows with empty division/jurisdiction and no approval are REJECTED.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from feedback_loop.interfaces import BoundaryPolicyGuard
from feedback_loop.models import BoundaryGranularity, BoundaryKey

logger = logging.getLogger(__name__)

# Fields that contain raw document text and MUST be stripped for cross-boundary sharing
_TEXT_FIELDS = frozenset({
    "front_matter_text",
    "source_text_snippet",
    "query_text",
    "correct_answer",
    "correct_value",
    "predicted_value",
    "missing_evidence_spans",
})

# Fields that are safe for cross-boundary sharing (structural/numeric features only)
_SHAREABLE_FIELDS = frozenset({
    "document_type",
    "classification_label",
    "page_count",
    "chunk_count",
    "chosen_strategy",
    "processing_level",
    "evidence_recall",
    "field_accuracy",
    "context_loss",
    "predicted_confidence",
    "final_confidence",
    "is_correct",
    "is_overconfident",
    "is_underconfident",
    "confidence_bucket",
    "accuracy",
    "error_penalty",
    "latency_ms",
    "extraction_method",
    "query_intent",
    "model_id",
})


class DefaultBoundaryPolicyGuard(BoundaryPolicyGuard):
    """Default boundary policy guard.

    Maintains a registry of approved sharing pairs.
    By default, no cross-boundary sharing is allowed.

    Boundary validation distinguishes two contexts:

    **Ingestion-time** (``validate_event_boundary``):
        Accepts partial boundary keys. Events are stored with warnings.

    **Training-time** (``validate_row``):
        Requires full granularity OR ``reduced_granularity_approved=True``.
        Rows lacking both are rejected to prevent silent data contamination.
    """

    def __init__(self) -> None:
        # Set of approved sharing pairs: (source_key, target_key)
        self._approved_pairs: Set[Tuple[str, str]] = set()
        # Set of boundary keys that allow same-client sharing
        self._client_sharing_enabled: Set[str] = set()

    def approve_sharing(
        self,
        source_boundary: BoundaryKey,
        target_boundary: BoundaryKey,
    ) -> None:
        """Approve data sharing between two boundaries."""
        self._approved_pairs.add((source_boundary.key, target_boundary.key))
        logger.info(
            "Approved sharing: %s → %s",
            source_boundary.key, target_boundary.key,
        )

    def enable_client_sharing(self, client: str) -> None:
        """Enable same-client sharing across divisions/jurisdictions."""
        self._client_sharing_enabled.add(client)

    # ── Ingestion-time validation ────────────────────────────────────

    def validate_event_boundary(self, boundary: BoundaryKey) -> List[str]:
        """Validate a boundary key at ingestion time (permissive).

        Returns a list of warning strings (empty = clean).
        Events are always stored, but warnings are logged.
        """
        warnings: List[str] = []
        if not boundary.client:
            warnings.append("BoundaryKey has empty client — event has no tenant isolation")
        if not boundary.division:
            warnings.append(
                f"BoundaryKey has empty division for client={boundary.client}"
            )
        if not boundary.jurisdiction:
            warnings.append(
                f"BoundaryKey has empty jurisdiction for client={boundary.client}"
            )
        for w in warnings:
            logger.warning("Ingestion boundary warning: %s", w)
        return warnings

    # ── Training-time validation ─────────────────────────────────────

    def validate_row(self, row: Any, expected_boundary: BoundaryKey) -> bool:
        """Check if a training row is boundary-safe (strict).

        A row is valid if:
        1. It has a boundary_key attribute that is a BoundaryKey
        2. The client is non-empty
        3. Either full granularity OR reduced_granularity_approved=True
        4. The boundary_key matches the expected boundary

        This is the **training-time** gate. Rows that fail here are
        silently dropped from the training dataset. They are not deleted —
        the feedback event that produced them is still auditable.
        """
        if not hasattr(row, "boundary_key"):
            logger.warning("Training row missing boundary_key: %s", type(row).__name__)
            return False

        row_boundary = row.boundary_key
        if not isinstance(row_boundary, BoundaryKey):
            logger.warning("Training row boundary_key is not a BoundaryKey: %s", type(row_boundary))
            return False

        # Hard constraint: client must be non-empty
        if not row_boundary.client:
            logger.error(
                "BoundaryKey has empty client — training row REJECTED. "
                "Every training row must carry a non-empty client identifier."
            )
            return False

        # Strict granularity enforcement for trainable rows
        if not row_boundary.is_full_granularity:
            if not row_boundary.reduced_granularity_approved:
                logger.warning(
                    "Training row REJECTED: boundary %s has %s granularity "
                    "without reduced_granularity_approved=True. "
                    "Set reduced_granularity_approved on the BoundaryKey or "
                    "provide full (client, division, jurisdiction).",
                    row_boundary.key, row_boundary.granularity.value,
                )
                return False
            # Approved reduced granularity — log but allow
            logger.info(
                "Training row accepted at %s granularity (policy-approved) "
                "for client=%s",
                row_boundary.granularity.value, row_boundary.client,
            )

        if row_boundary.key != expected_boundary.key:
            logger.warning(
                "Boundary mismatch: row has %s, expected %s",
                row_boundary.key, expected_boundary.key,
            )
            return False

        return True

    def sanitize_for_sharing(self, row: Any) -> Any:
        """Strip client-specific text fields for approved cross-boundary sharing.

        Returns a sanitized copy with raw document text removed.
        Only structural/numeric features are retained.
        """
        if not hasattr(row, "model_copy"):
            # Not a Pydantic model — return as-is with warning
            logger.warning("Cannot sanitize non-Pydantic row: %s", type(row).__name__)
            return row

        updates: Dict[str, Any] = {}
        for field_name in _TEXT_FIELDS:
            if hasattr(row, field_name):
                val = getattr(row, field_name)
                if isinstance(val, str):
                    updates[field_name] = ""
                elif isinstance(val, list):
                    updates[field_name] = []
                elif isinstance(val, dict):
                    updates[field_name] = {}

        return row.model_copy(update=updates)

    def is_sharing_approved(
        self,
        source_boundary: BoundaryKey,
        target_boundary: BoundaryKey,
    ) -> bool:
        """Check if data sharing between two boundaries is approved.

        Returns True if:
        1. Same boundary (always allowed)
        2. Explicit pair approval exists
        3. Same client and client-level sharing is enabled
        """
        if source_boundary.key == target_boundary.key:
            return True

        if (source_boundary.key, target_boundary.key) in self._approved_pairs:
            return True

        if (
            source_boundary.shares_client(target_boundary)
            and source_boundary.client in self._client_sharing_enabled
        ):
            return True

        return False

    def filter_rows_for_boundary(
        self,
        rows: List[Any],
        target_boundary: BoundaryKey,
    ) -> List[Any]:
        """Filter and sanitize rows for use in a target boundary's training.

        - Same-boundary rows pass through unchanged
        - Approved cross-boundary rows are sanitized (text stripped)
        - Unapproved cross-boundary rows are dropped
        """
        result: List[Any] = []
        for row in rows:
            if not hasattr(row, "boundary_key"):
                continue
            source = row.boundary_key
            if source.key == target_boundary.key:
                result.append(row)
            elif self.is_sharing_approved(source, target_boundary):
                result.append(self.sanitize_for_sharing(row))
            else:
                logger.debug(
                    "Dropped row from boundary %s (target: %s)",
                    source.key, target_boundary.key,
                )
        return result
