from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Canonical State Enums ────────────────────────────────────────────


class DocumentLifecycle(str, Enum):
    """Canonical document lifecycle states.

    Tracks a document from initial ingestion through classification,
    extraction, and verification to a fully processed state.
    """
    PENDING = "pending"              # Uploaded but not yet ingested
    INGESTING = "ingesting"          # Page triage + canonicalization in progress
    INGESTED = "ingested"            # Pages and chunks stored; awaiting classification
    CLASSIFYING = "classifying"      # Classifier agent running
    CLASSIFIED = "classified"        # document_type + classification_label assigned
    EXTRACTING = "extracting"        # Extractor agent running
    EXTRACTED = "extracted"          # Fields extracted; awaiting transformation
    TRANSFORMING = "transforming"    # Transformer agent normalizing values
    COMPLETE = "complete"            # Fully processed and queryable
    ERROR = "error"                  # Processing failed; requires remediation


class QueryState(str, Enum):
    """Canonical query lifecycle states.

    Tracks a query through the orchestrator ReAct loop from receipt
    through routing, retrieval, synthesis, and verification.
    """
    RECEIVED = "received"            # Query received; awaiting routing
    ROUTING = "routing"              # Router agent classifying intent
    RETRIEVING = "retrieving"        # Retriever agent searching evidence
    SYNTHESIZING = "synthesizing"    # Synthesiser agent generating answer
    VERIFYING = "verifying"          # Verifier agent checking claims
    COMPLETE = "complete"            # Answer delivered to user
    ERROR = "error"                  # Query processing failed


class FeedbackState(str, Enum):
    """Canonical feedback event states.

    Tracks feedback from ingestion through attribution, training row
    generation, and retraining orchestration.
    """
    INGESTED = "ingested"            # Raw feedback stored
    QUARANTINED = "quarantined"      # No prediction trace found; non-trainable
    ATTRIBUTED = "attributed"        # Impacted layers identified
    TRAINING_ROWS_BUILT = "training_rows_built"  # Training rows generated
    SUBMITTED = "submitted"          # Rows submitted to retraining orchestrator
    TRAINED = "trained"              # Retraining completed using this feedback


@dataclass(frozen=True)
class TriageMetrics:
    text_length: int
    text_density: float
    image_coverage_ratio: float
    layout_complexity_score: float


@dataclass(frozen=True)
class TriageDecision:
    metrics: TriageMetrics
    decision: str
    reason_codes: List[str]


@dataclass(frozen=True)
class DIResult:
    result: Dict[str, Any]


@dataclass(frozen=True)
class CanonicalSpan:
    text: str
    char_start: int
    char_end: int
    polygons: List[Dict[str, Any]]
    source_type: str
    page_number: int
    heading_path: str
    section_id: str
    is_table: bool


@dataclass(frozen=True)
class CanonicalPage:
    doc_id: str
    page_number: int
    text: str
    spans: List[CanonicalSpan]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    page_numbers: List[int]
    macro_id: int
    child_id: int
    chunk_type: str
    text_content: str
    char_start: int
    char_end: int
    polygons: List[Dict[str, Any]]
    source_type: str
    score: float
    heading_path: str
    section_id: str
    document_type: Optional[str] = None
    classification_label: Optional[str] = None


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    filename: str
    sha256: str
    page_count: int
    document_type: Optional[str] = None
    classification_label: Optional[str] = None


@dataclass(frozen=True)
class PageRecord:
    doc_id: str
    page_number: int
    triage_metrics: TriageMetrics
    triage_decision: str
    reason_codes: List[str]
    di_json_path: Optional[str]


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    page_numbers: List[int]
    macro_id: int
    child_id: int
    chunk_type: str
    text_content: str
    char_start: int
    char_end: int
    polygons: List[Dict[str, Any]]
    source_type: str
    embedding_model: str
    embedding_dim: int
    embedding: List[float]
    heading_path: str
    section_id: str
    document_type: Optional[str] = None
    classification_label: Optional[str] = None


@dataclass(frozen=True)
class DocumentFact:
    doc_id: str
    fact_name: str
    value: Optional[str]
    status: str
    confidence: float
    source_chunk_id: Optional[str]
    page_numbers: List[int]
    polygons: List[Dict[str, Any]]
    evidence_excerpt: Optional[str]
