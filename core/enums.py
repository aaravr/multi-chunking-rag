"""Canonical string enums for the IDP platform.

Centralises all domain constants that were previously scattered as hardcoded
string literals across agents, retrieval, ingestion, and synthesis modules.
Every module MUST import from here — never use raw string literals for these
domain values.
"""

from enum import Enum


class IntentType(str, Enum):
    """Query intent classification types (§4.2)."""
    LOCATION = "location"
    COVERAGE = "coverage"
    SEMANTIC = "semantic"
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    AGGREGATION = "aggregation"
    METADATA = "metadata"


class CoverageType(str, Enum):
    """Coverage query subtypes."""
    LIST = "list"
    NUMERIC_LIST = "numeric_list"
    ATTRIBUTE = "attribute"
    POINTER = "pointer"


class ChunkType(str, Enum):
    """Canonical chunk type labels assigned during ingestion."""
    NARRATIVE = "narrative"
    TABLE = "table"
    HEADING = "heading"
    BOILERPLATE = "boilerplate"


class SourceType(str, Enum):
    """Provenance source for a chunk or span."""
    NATIVE = "native"      # PyMUPDF extraction
    DI = "di"              # Azure Document Intelligence


class TriageDecisionType(str, Enum):
    """Page-level triage decisions."""
    NATIVE_ONLY = "native_only"
    DI_REQUIRED = "di_required"
    DI_SKIPPED = "di_skipped"


class ProgressPhase(str, Enum):
    """Progress callback phase identifiers."""
    TRIAGE = "triage"
    DI = "di"
    DI_SKIPPED = "di_skipped"
    PAGES_COMMITTED = "pages_committed"
    EMBED = "embed"
    CLASSIFY = "classify"
    CLASSIFY_DONE = "classify_done"
    PREPROCESS = "preprocess"
    PREPROCESS_DONE = "preprocess_done"
    METADATA_ONLY = "metadata_only"
    METADATA_ONLY_DONE = "metadata_only_done"


class ProcessingLevel(str, Enum):
    """Preprocessor chunking processing levels."""
    SKIP = "skip"
    METADATA_ONLY = "metadata_only"
    LATE_CHUNKING = "late_chunking"


class RetrievalMethod(str, Enum):
    """Retrieval strategy methods."""
    VECTOR = "vector"
    HYBRID = "hybrid"
    BM25 = "bm25"


class SynthesisMode(str, Enum):
    """How an answer was synthesised."""
    DETERMINISTIC = "deterministic"
    LLM = "llm"


class CoverageMode(str, Enum):
    """Coverage extraction modes."""
    DETERMINISTIC = "deterministic"
    LLM_FALLBACK = "llm_fallback"
    LLM_ALWAYS = "llm_always"


class VerificationVerdict(str, Enum):
    """Verifier agent verdicts."""
    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"


class ExtractionMethod(str, Enum):
    """How a field value was extracted."""
    REGEX = "regex"
    LLM = "llm"


class FactStatus(str, Enum):
    """Document fact statuses."""
    FOUND = "found"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"


class LLMProvider(str, Enum):
    """LLM provider backends."""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"


class ParserBackend(str, Enum):
    """Document parser backends."""
    PYMUPDF = "pymupdf"
    DOCLING = "docling"
