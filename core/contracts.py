from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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
    chunk_type: str
    text_content: str
    char_start: int
    char_end: int
    polygons: List[Dict[str, Any]]
    source_type: str
    score: float
    heading_path: str
    section_id: str


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    filename: str
    sha256: str
    page_count: int


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
