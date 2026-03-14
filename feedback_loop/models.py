"""Typed domain models for the feedback loop subsystem.

All models use Pydantic v2 for validation, serialization, and SQL-friendly persistence.
Every training row carries source_feedback_ids and boundary_key for lineage and isolation.

Mathematical mapping:
    PlannerTrainingRow   → π*(z) = argmax_a E[R(a, z)]
    ClassifierTrainingRow → P(c|z) = softmax(Wz + b)
    ChunkingTrainingRow  → k*(x) = argmax_k E[Q(k, x)]
    ExtractionTrainingRow → v̂ = E(x, c, S)
    CalibrationTrainingRow → P(correct|z) = σ(wᵀz)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


# ── Enums ────────────────────────────────────────────────────────────


class DecisionLayer(str, Enum):
    """The five decision layers in the IDP pipeline: x→p→c→k→e→t→y."""
    PLANNING = "planning"
    CLASSIFICATION = "classification"
    CHUNKING = "chunking"
    EXTRACTION = "extraction"
    CALIBRATION = "calibration"


class FeedbackRating(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CORRECTION = "correction"


class FeedbackChannel(str, Enum):
    UI = "ui"
    API = "api"
    BATCH_EVALUATION = "batch_evaluation"
    DOWNSTREAM_SYSTEM = "downstream_system"


class ReasonCode(str, Enum):
    """Structured reason codes derived from raw feedback normalization."""
    WRONG_DOCUMENT_CLASS = "wrong_document_class"
    WRONG_SCHEMA_APPLIED = "wrong_schema_applied"
    MISSING_EVIDENCE_SPAN = "missing_evidence_span"
    IRRELEVANT_CHUNKS_RETRIEVED = "irrelevant_chunks_retrieved"
    WRONG_VALUE_EXTRACTED = "wrong_value_extracted"
    INCOMPLETE_EXTRACTION = "incomplete_extraction"
    UNRESOLVED_REFERENCE = "unresolved_reference"
    OVERCONFIDENT_WRONG_ANSWER = "overconfident_wrong_answer"
    UNDERCONFIDENT_CORRECT_ANSWER = "underconfident_correct_answer"
    WRONG_PROCESSING_PATH = "wrong_processing_path"
    CORRECTION_RATE_EXCEEDED = "correction_rate_exceeded"
    HALLUCINATED_CONTENT = "hallucinated_content"
    OTHER = "other"


class ModelStage(str, Enum):
    """Model lifecycle stages for controlled promotion."""
    SHADOW = "shadow"
    CANARY = "canary"
    APPROVED = "approved"
    REJECTED = "rejected"
    ROLLBACK_READY = "rollback_ready"


class RetrainingTrigger(str, Enum):
    SCHEDULED = "scheduled"
    THRESHOLD = "threshold"
    MANUAL = "manual"
    EVENT_DRIVEN = "event_driven"


# ── Boundary Key ─────────────────────────────────────────────────────


class BoundaryKey(BaseModel):
    """Training isolation boundary: B = (client, division, jurisdiction).

    Every feedback event and training row must carry or derive a boundary_key.
    No cross-boundary training data mixing by default.

    The client field is required and must be non-empty. Empty division or
    jurisdiction fields are allowed but will generate a validation warning
    in boundary policy enforcement.
    """
    client: str = Field(..., min_length=1, description="Client identifier (required, non-empty)")
    division: str = ""
    jurisdiction: str = ""

    @property
    def key(self) -> str:
        """Deterministic string key for grouping and policy lookup."""
        parts = [self.client]
        if self.division:
            parts.append(self.division)
        if self.jurisdiction:
            parts.append(self.jurisdiction)
        return "|".join(parts)

    def is_same_boundary(self, other: BoundaryKey) -> bool:
        return self.key == other.key

    def shares_client(self, other: BoundaryKey) -> bool:
        return self.client == other.client


# ── Prediction Trace ─────────────────────────────────────────────────


class PlannerDecision(BaseModel):
    """Runtime record of the planner/orchestrator decision (layer p)."""
    action: str = ""                                 # e.g. "retrieve_then_synthesize"
    query_decomposition: List[str] = Field(default_factory=list)  # sub-queries
    processing_path: str = ""                        # e.g. "full_pipeline", "metadata_only"
    routing_confidence: float = 0.0
    model_id: str = ""
    latency_ms: float = 0.0


class ClassifierDecision(BaseModel):
    """Runtime record of the classification decision (layer c)."""
    document_type: str = ""
    classification_label: str = ""
    confidence: float = 0.0
    classification_method: str = ""                  # "deterministic"|"llm"|"memory_match"
    evidence_signals: Dict[str, Any] = Field(default_factory=dict)
    model_id: str = ""


class ChunkingDecision(BaseModel):
    """Runtime record of the chunking strategy decision (layer k)."""
    strategy_name: str = ""
    processing_level: str = ""                       # "skip"|"metadata_only"|"late_chunking"|...
    chunk_count: int = 0
    chunk_ids: List[str] = Field(default_factory=list)
    chunk_texts: List[str] = Field(default_factory=list)   # stripped for boundary safety
    evidence_spans: List[Dict[str, Any]] = Field(default_factory=list)
    decision_method: str = ""                        # "deterministic"|"learned"|"default"


class ExtractionDecision(BaseModel):
    """Runtime record of the extraction decision (layer e)."""
    extracted_fields: Dict[str, str] = Field(default_factory=dict)  # field_name → raw_value
    extraction_method: str = ""                      # "regex"|"llm"|"hybrid"
    field_confidences: Dict[str, float] = Field(default_factory=dict)
    source_chunk_ids: Dict[str, List[str]] = Field(default_factory=dict)
    model_id: str = ""


class TransformationDecision(BaseModel):
    """Runtime record of the transformation/resolution decision (layer t)."""
    normalized_fields: Dict[str, str] = Field(default_factory=dict)
    mcp_lookups_performed: int = 0
    unresolved_references: List[str] = Field(default_factory=list)


class PredictionTrace(BaseModel):
    """Complete runtime decision trace for a single query/document processing run.

    Joins all five decision layers: p → c → k → e → t.
    This is persisted at runtime and joined with feedback events post-hoc.
    """
    trace_id: str = Field(default_factory=_new_id)
    query_id: str = ""
    doc_id: str = ""
    user_id: str = ""
    boundary_key: BoundaryKey

    # Per-layer decisions
    planner: PlannerDecision = Field(default_factory=PlannerDecision)
    classifier: ClassifierDecision = Field(default_factory=ClassifierDecision)
    chunking: ChunkingDecision = Field(default_factory=ChunkingDecision)
    extraction: ExtractionDecision = Field(default_factory=ExtractionDecision)
    transformation: TransformationDecision = Field(default_factory=TransformationDecision)

    # Final output
    final_answer: str = ""
    final_confidence: float = 0.0
    citations: List[str] = Field(default_factory=list)

    # Model versions for reproducibility
    model_versions: Dict[str, str] = Field(default_factory=dict)

    # Timing
    total_latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=_utcnow)


# ── Feedback Event ───────────────────────────────────────────────────


class FeedbackEvent(BaseModel):
    """A feedback event from a reviewer, validator, or downstream system.

    Raw input to the feedback loop.  The attribution engine will map this
    to one or more impacted decision layers.
    """
    feedback_id: str = Field(default_factory=_new_id)
    trace_id: str = ""                               # links to PredictionTrace
    query_id: str = ""
    doc_id: str = ""
    user_id: str = ""
    boundary_key: BoundaryKey

    # Feedback content
    rating: FeedbackRating
    comment: str = ""
    correct_answer: Optional[str] = None
    correct_document_type: Optional[str] = None
    correct_classification_label: Optional[str] = None
    correct_evidence_spans: List[str] = Field(default_factory=list)
    correct_field_values: Dict[str, str] = Field(default_factory=dict)

    # Processing path override
    processing_path_override: Optional[str] = None

    # Metadata
    channel: FeedbackChannel = FeedbackChannel.UI
    cited_chunk_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


# ── Normalized Feedback ──────────────────────────────────────────────


class NormalizedFeedback(BaseModel):
    """Structured feedback after normalization.

    Maps raw comments and signals into structured reason codes.
    """
    feedback_id: str
    reason_codes: List[ReasonCode] = Field(default_factory=list)
    structured_corrections: Dict[str, Any] = Field(default_factory=dict)
    severity: float = 0.0                            # 0.0–1.0


# ── Attribution Result ───────────────────────────────────────────────


class ImpactedLayer(BaseModel):
    """One impacted decision layer from attribution.

    A single feedback event can impact multiple layers.
    """
    layer: DecisionLayer
    confidence: float                                # 0.0–1.0
    explanation: str
    reason_codes: List[ReasonCode] = Field(default_factory=list)
    rule_id: str = ""                                # which attribution rule fired


class FeedbackAttribution(BaseModel):
    """Result of attributing a feedback event to decision layers.

    Maps one feedback event → one or more impacted layers.
    Preserves lineage from source feedback.
    """
    attribution_id: str = Field(default_factory=_new_id)
    feedback_id: str
    trace_id: str
    boundary_key: BoundaryKey
    impacted_layers: List[ImpactedLayer]
    attribution_method: str = "rule_based"           # "rule_based"|"learned"
    created_at: datetime = Field(default_factory=_utcnow)


# ── Training Rows (Layer-Specific) ──────────────────────────────────
#
# Every training row carries:
#   - source_feedback_ids: lineage to originating feedback events
#   - boundary_key: training isolation boundary
#   - row_id: unique identifier


class PlannerTrainingRow(BaseModel):
    """Training row for the planning/orchestration layer.

    Objective: π*(z) = argmax_a E[R(a, z)]
    R = Accuracy - λ₁·ReviewCost - λ₂·Latency - λ₃·ErrorPenalty
    """
    row_id: str = Field(default_factory=_new_id)
    source_feedback_ids: List[str]
    boundary_key: BoundaryKey

    # Context z
    query_text: str = ""
    document_type: str = ""
    classification_label: str = ""
    page_count: int = 0

    # Action a (what was chosen)
    chosen_action: str = ""
    chosen_processing_path: str = ""
    query_decomposition: List[str] = Field(default_factory=list)

    # Correct action (from feedback)
    correct_action: Optional[str] = None
    correct_processing_path: Optional[str] = None

    # Reward signal
    accuracy: float = 0.0                            # 0 or 1 from feedback
    review_cost: float = 0.0                         # normalized human review cost
    latency_ms: float = 0.0
    error_penalty: float = 0.0                       # 1.0 if correction needed

    # Reward weights (configurable per deployment)
    lambda_review_cost: float = 0.1
    lambda_latency: float = 0.01
    lambda_error_penalty: float = 1.0

    @property
    def reward(self) -> float:
        """R = Accuracy - λ₁·ReviewCost - λ₂·Latency - λ₃·ErrorPenalty."""
        return (
            self.accuracy
            - self.lambda_review_cost * self.review_cost
            - self.lambda_latency * (self.latency_ms / 1000.0)
            - self.lambda_error_penalty * self.error_penalty
        )

    created_at: datetime = Field(default_factory=_utcnow)


class ClassifierTrainingRow(BaseModel):
    """Training row for the classification layer.

    Objective: P(c|z) = softmax(Wz + b)
    """
    row_id: str = Field(default_factory=_new_id)
    source_feedback_ids: List[str]
    boundary_key: BoundaryKey

    # Input features z
    filename: str = ""
    front_matter_text: str = ""                      # first N chars (boundary-safe)
    structural_signals: Dict[str, Any] = Field(default_factory=dict)

    # Model prediction
    predicted_document_type: str = ""
    predicted_classification_label: str = ""
    predicted_confidence: float = 0.0

    # Ground truth (from feedback correction)
    correct_document_type: str = ""
    correct_classification_label: str = ""

    # Training signal
    is_correct: bool = False

    created_at: datetime = Field(default_factory=_utcnow)


class ChunkingTrainingRow(BaseModel):
    """Training row for the chunking/linking layer.

    Objective: k*(x) = argmax_k E[Q(k, x)]
    Q(k, x) = α·EvidenceRecall + β·FieldAccuracy - γ·ContextLoss
    """
    row_id: str = Field(default_factory=_new_id)
    source_feedback_ids: List[str]
    boundary_key: BoundaryKey

    # Input x
    document_type: str = ""
    classification_label: str = ""
    page_count: int = 0

    # Strategy k (what was chosen)
    chosen_strategy: str = ""
    processing_level: str = ""
    chunk_count: int = 0

    # Quality signal Q(k, x)
    evidence_recall: float = 0.0                     # fraction of correct spans found
    field_accuracy: float = 0.0                      # extraction accuracy given chunks
    context_loss: float = 0.0                        # information lost by chunking

    # Weights (configurable)
    alpha_evidence_recall: float = 0.5
    beta_field_accuracy: float = 0.3
    gamma_context_loss: float = 0.2

    @property
    def quality_score(self) -> float:
        """Q(k,x) = α·EvidenceRecall + β·FieldAccuracy - γ·ContextLoss."""
        return (
            self.alpha_evidence_recall * self.evidence_recall
            + self.beta_field_accuracy * self.field_accuracy
            - self.gamma_context_loss * self.context_loss
        )

    # Missing spans (what should have been chunked but wasn't)
    missing_evidence_spans: List[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=_utcnow)


class ExtractionTrainingRow(BaseModel):
    """Training row for the extraction layer.

    Objective: v̂ = E(x, c, S)
    """
    row_id: str = Field(default_factory=_new_id)
    source_feedback_ids: List[str]
    boundary_key: BoundaryKey

    # Input context
    document_type: str = ""
    classification_label: str = ""
    field_name: str = ""

    # Model prediction
    predicted_value: str = ""
    predicted_confidence: float = 0.0
    extraction_method: str = ""                      # "regex"|"llm"|"hybrid"
    source_chunk_ids: List[str] = Field(default_factory=list)
    source_text_snippet: str = ""                    # boundary-safe snippet (no raw doc text)

    # Ground truth
    correct_value: str = ""

    # Signal
    is_correct: bool = False
    is_unresolved_reference: bool = False             # "same as above" etc.

    created_at: datetime = Field(default_factory=_utcnow)


class CalibrationTrainingRow(BaseModel):
    """Training row for confidence calibration.

    Objective: P(correct|z) = σ(wᵀz)
    """
    row_id: str = Field(default_factory=_new_id)
    source_feedback_ids: List[str]
    boundary_key: BoundaryKey

    # Feature vector z (per-layer confidence signals)
    planner_confidence: float = 0.0
    classifier_confidence: float = 0.0
    chunking_quality_score: float = 0.0
    extraction_confidence: float = 0.0
    final_confidence: float = 0.0

    # Additional features
    document_type: str = ""
    query_intent: str = ""
    chunk_count: int = 0
    model_id: str = ""

    # Ground truth label
    is_correct: bool = False

    # Calibration diagnostics
    confidence_bucket: str = ""                      # e.g. "0.8-0.9"
    is_overconfident: bool = False                    # high conf + wrong
    is_underconfident: bool = False                   # low conf + correct

    created_at: datetime = Field(default_factory=_utcnow)


# ── Model Lifecycle ──────────────────────────────────────────────────


class ModelCandidate(BaseModel):
    """A candidate model version produced by retraining."""
    candidate_id: str = Field(default_factory=_new_id)
    layer: DecisionLayer
    boundary_key: BoundaryKey
    model_version: str = ""
    parent_model_version: str = ""                   # what it was trained from

    # Training provenance
    training_row_count: int = 0
    source_feedback_ids: List[str] = Field(default_factory=list)
    training_started_at: Optional[datetime] = None
    training_completed_at: Optional[datetime] = None

    # Hierarchical sharing: θ_B = θ_shared + δ_B
    shared_base_version: Optional[str] = None        # θ_shared
    boundary_delta_version: Optional[str] = None     # δ_B

    stage: ModelStage = ModelStage.SHADOW
    created_at: datetime = Field(default_factory=_utcnow)


class LayerMetrics(BaseModel):
    """Metrics for a single decision layer."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    ece: float = 0.0                                 # expected calibration error
    sample_count: int = 0
    extra: Dict[str, float] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Comparison of candidate model metrics vs baseline."""
    report_id: str = Field(default_factory=_new_id)
    candidate_id: str
    layer: DecisionLayer
    boundary_key: BoundaryKey

    baseline_metrics: LayerMetrics
    candidate_metrics: LayerMetrics

    # Decision
    improvement_delta: float = 0.0                   # candidate - baseline (primary metric)
    is_statistically_significant: bool = False
    p_value: Optional[float] = None
    recommendation: str = ""                         # "promote"|"reject"|"needs_review"
    explanation: str = ""

    created_at: datetime = Field(default_factory=_utcnow)
