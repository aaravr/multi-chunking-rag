"""Abstract service interfaces for the feedback loop subsystem.

Each interface defines a single responsibility in the feedback-to-retraining pipeline.
Implementations can be swapped (in-memory for tests, DB-backed for production).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from feedback_loop.models import (
    BoundaryKey,
    CalibrationTrainingRow,
    ChunkingTrainingRow,
    ClassifierTrainingRow,
    DecisionLayer,
    EvaluationReport,
    ExtractionTrainingRow,
    FeedbackAttribution,
    FeedbackEvent,
    ModelCandidate,
    ModelStage,
    NormalizedFeedback,
    PlannerTrainingRow,
    PredictionTrace,
    RetrainingTrigger,
)


class FeedbackIngestionService(ABC):
    """Accepts and validates feedback events from reviewers, validators, downstream systems.

    Responsibilities:
    - Validate payload completeness and types
    - Derive boundary_key if not provided
    - Persist raw event
    - Return feedback_id for tracking
    """

    @abstractmethod
    def ingest(self, event: FeedbackEvent) -> str:
        """Validate and store a feedback event.  Returns feedback_id."""

    @abstractmethod
    def get_event(self, feedback_id: str) -> Optional[FeedbackEvent]:
        """Retrieve a stored feedback event by ID."""

    @abstractmethod
    def list_events(
        self,
        boundary_key: Optional[BoundaryKey] = None,
        doc_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[FeedbackEvent]:
        """List feedback events, optionally filtered by boundary or document."""


class TraceJoinService(ABC):
    """Joins feedback events with runtime prediction traces.

    Responsibilities:
    - Store prediction traces at runtime
    - Retrieve traces by trace_id or query_id
    - Join a feedback event with its corresponding prediction trace
    """

    @abstractmethod
    def store_trace(self, trace: PredictionTrace) -> str:
        """Persist a prediction trace.  Returns trace_id."""

    @abstractmethod
    def get_trace(self, trace_id: str) -> Optional[PredictionTrace]:
        """Retrieve a prediction trace by ID."""

    @abstractmethod
    def get_trace_by_query(self, query_id: str) -> Optional[PredictionTrace]:
        """Retrieve a prediction trace by query_id."""

    @abstractmethod
    def join(self, event: FeedbackEvent) -> Optional[PredictionTrace]:
        """Join a feedback event with its prediction trace.

        Looks up by event.trace_id first, then by event.query_id.
        Returns None if no trace is found.
        """


class FeedbackNormalizer(ABC):
    """Maps raw comments and feedback signals into structured types and reason codes.

    Responsibilities:
    - Parse free-text comments into reason codes
    - Detect feedback patterns (correction type, severity)
    - Produce NormalizedFeedback with structured_corrections
    """

    @abstractmethod
    def normalize(
        self,
        event: FeedbackEvent,
        trace: Optional[PredictionTrace] = None,
    ) -> NormalizedFeedback:
        """Normalize a raw feedback event into structured form."""


class AttributionEngine(ABC):
    """Attributes a feedback event to one or more impacted decision layers.

    Responsibilities:
    - Apply deterministic attribution rules first
    - Support extensibility for learned attribution
    - Return impacted layers with confidence and explanation
    - Support multiple impacted layers per event
    """

    @abstractmethod
    def attribute(
        self,
        event: FeedbackEvent,
        trace: PredictionTrace,
        normalized: NormalizedFeedback,
    ) -> FeedbackAttribution:
        """Attribute a feedback event to impacted layers."""


class TrainingRowBuilder(ABC):
    """Converts attributed feedback into layer-specific training rows.

    Responsibilities:
    - Emit rows only for impacted layers
    - Preserve source_feedback_ids in every row
    - Include boundary_key in every row
    - Support one-to-many (one event → multiple rows across layers)
    """

    @abstractmethod
    def build(
        self,
        event: FeedbackEvent,
        trace: PredictionTrace,
        attribution: FeedbackAttribution,
    ) -> Dict[DecisionLayer, List[Any]]:
        """Build layer-specific training rows.

        Returns a dict mapping DecisionLayer to list of training rows.
        Only impacted layers produce rows.
        """


class BoundaryPolicyGuard(ABC):
    """Enforces boundary-safe training rules.

    Default rules:
    - Every training row must carry boundary_key
    - No cross-boundary training data mixing by default
    - No global reusable raw cross-client document text
    - Only approved sanitized feature sharing allowed
    - Optional hierarchical sharing: θ_B = θ_shared + δ_B

    Responsibilities:
    - Validate rows carry correct boundary_key
    - Strip or block disallowed fields
    - Approve or reject cross-boundary data usage
    """

    @abstractmethod
    def validate_row(self, row: Any, expected_boundary: BoundaryKey) -> bool:
        """Check if a training row is boundary-safe.  Returns True if allowed."""

    @abstractmethod
    def sanitize_for_sharing(self, row: Any) -> Any:
        """Strip client-specific text fields for approved cross-boundary sharing.

        Returns a sanitized copy with raw document text removed.
        """

    @abstractmethod
    def is_sharing_approved(
        self,
        source_boundary: BoundaryKey,
        target_boundary: BoundaryKey,
    ) -> bool:
        """Check if data sharing between two boundaries is approved."""


class RetrainingOrchestrator(ABC):
    """Groups training rows into datasets and triggers layer-specific retraining jobs.

    Supports scheduled and event-driven execution.
    """

    @abstractmethod
    def submit_rows(
        self,
        layer: DecisionLayer,
        rows: List[Any],
        boundary_key: BoundaryKey,
    ) -> str:
        """Submit training rows for a layer.  Returns dataset_id."""

    @abstractmethod
    def trigger_retraining(
        self,
        layer: DecisionLayer,
        boundary_key: BoundaryKey,
        trigger: RetrainingTrigger = RetrainingTrigger.THRESHOLD,
        min_rows: int = 10,
    ) -> Optional[str]:
        """Trigger a retraining job for a layer.  Returns job_id or None if insufficient data."""

    @abstractmethod
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a retraining job."""


class ModelEvaluator(ABC):
    """Compares candidate model metrics against the baseline.

    Supports per-layer evaluation with configurable metrics.
    """

    @abstractmethod
    def evaluate(
        self,
        candidate: ModelCandidate,
        baseline_metrics: Dict[str, float],
        validation_data: List[Any],
    ) -> EvaluationReport:
        """Evaluate a candidate model against baseline metrics."""


class ModelPromotionController(ABC):
    """Controls model lifecycle: shadow → canary → approve / reject → rollback_ready.

    Supports controlled promotion with governance audit trail.
    """

    @abstractmethod
    def promote(self, candidate_id: str, target_stage: ModelStage) -> ModelCandidate:
        """Promote a candidate to the target lifecycle stage."""

    @abstractmethod
    def rollback(self, candidate_id: str) -> ModelCandidate:
        """Rollback a candidate to its parent model version."""

    @abstractmethod
    def get_active_model(
        self,
        layer: DecisionLayer,
        boundary_key: BoundaryKey,
    ) -> Optional[ModelCandidate]:
        """Get the currently active (approved) model for a layer+boundary."""
