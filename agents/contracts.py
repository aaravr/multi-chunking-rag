"""Typed agent contracts per MASTER_PROMPT §4.

Every agent has a frozen input/output contract. These dataclasses are the
ONLY way agents communicate — no untyped dicts cross agent boundaries.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Shared value objects ──────────────────────────────────────────────

@dataclass(frozen=True)
class Citation:
    """Maps a [C#] tag in the answer to a specific chunk with page/polygon lineage."""
    citation_id: str
    chunk_id: str
    doc_id: str
    page_numbers: List[int]
    polygons: List[Dict[str, Any]]
    heading_path: str
    section_id: str
    text_snippet: str


@dataclass(frozen=True)
class TokenUsage:
    """Token consumption tracker across all model calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class CostBreakdown:
    """Cost tracking per model tier."""
    model_id: str = ""
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0


@dataclass(frozen=True)
class ModelAttribution:
    """Records which model was used, in which role, with usage stats."""
    model_id: str
    role: str           # e.g. "synthesis", "verification", "embedding"
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_estimate: float


@dataclass(frozen=True)
class DocumentScope:
    """Defines which documents/corpus to search."""
    doc_ids: List[str] = field(default_factory=list)
    corpus_id: Optional[str] = None


@dataclass(frozen=True)
class PermissionSet:
    """Placeholder for user permissions (RBAC + document-level)."""
    user_id: str = ""
    role: str = "reader"
    clearance_level: int = 0


@dataclass(frozen=True)
class OutputFormat:
    """Requested output style."""
    format_type: str = "prose"  # "prose" | "bullet_list" | "comparison_table" | "citation_list"


# ── Query Intent (extends PoC QueryIntent) ────────────────────────────

@dataclass(frozen=True)
class QueryIntent:
    """Classified query intent (MASTER_PROMPT §4.2)."""
    intent: str                                  # location | coverage | semantic | comparison | multi_hop | aggregation | metadata
    pages: List[int] = field(default_factory=list)
    coverage_type: Optional[str] = None          # list | attribute | numeric_list | pointer
    status_filter: Optional[str] = None          # closed
    entities: List[str] = field(default_factory=list)       # For comparison queries
    time_periods: List[str] = field(default_factory=list)   # For comparison queries
    sub_query_dependencies: List[str] = field(default_factory=list)  # For multi-hop


# ── Agent Message Envelope (§5.1) ────────────────────────────────────

@dataclass(frozen=True)
class AgentMessage:
    """Typed message envelope for inter-agent communication (§5.1)."""
    message_id: str
    query_id: str
    from_agent: str
    to_agent: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: str                 # ISO 8601
    token_budget_remaining: int = 0


# ── Router Agent Contracts (§4.2) ────────────────────────────────────

@dataclass(frozen=True)
class SubQuery:
    """A decomposed sub-query with dependency tracking."""
    sub_query_id: str
    query_text: str
    intent: QueryIntent
    depends_on: List[str] = field(default_factory=list)  # sub_query_ids this depends on


@dataclass(frozen=True)
class RetrievalStrategy:
    """How to retrieve evidence for a sub-query."""
    method: str          # "vector" | "bm25" | "hybrid" | "metadata" | "section_expansion"
    top_k: int = 10
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentTarget:
    """Identifies a specific document to search."""
    doc_id: str
    document_type: Optional[str] = None


@dataclass(frozen=True)
class QueryPlan:
    """Output of the Router Agent (§4.2)."""
    query_id: str
    original_query: str
    resolved_query: str
    primary_intent: QueryIntent
    sub_queries: List[SubQuery]
    retrieval_strategies: Dict[str, RetrievalStrategy]
    document_targets: List[DocumentTarget]
    estimated_token_cost: int = 0
    classification_confidence: float = 0.0
    classification_alternatives: List[QueryIntent] = field(default_factory=list)
    classification_method: str = "deterministic"


# ── Retriever Agent Contracts (§4.3) ────────────────────────────────

@dataclass(frozen=True)
class SearchScope:
    """Records what was actually searched."""
    doc_ids: List[str]
    sections_searched: List[str] = field(default_factory=list)
    pages_searched: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class RankedEvidence:
    """Output of the Retriever Agent (§4.3)."""
    query_id: str
    sub_query_id: str
    chunks: List[Any]                              # List[RetrievedChunk] from core.contracts
    retrieval_methods: Dict[str, str]              # chunk_id → method
    scores: Dict[str, float]                       # chunk_id → score
    fusion_weights: Optional[Dict[str, float]] = None
    total_candidates_scanned: int = 0
    total_candidates_filtered: int = 0
    search_scope: Optional[SearchScope] = None


# ── Synthesiser Agent Contracts (§4.4) ───────────────────────────────

@dataclass(frozen=True)
class SynthesisResult:
    """Output of the Synthesiser Agent (§4.4)."""
    query_id: str
    answer: str
    citations: List[Citation]
    prompt_template_id: str = ""
    prompt_template_version: str = ""
    model_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    synthesis_mode: str = "llm"  # "deterministic" | "llm" | "hybrid"


# ── Verifier Agent Contracts (§4.5) ──────────────────────────────────

@dataclass(frozen=True)
class ClaimVerification:
    """Per-claim verification result."""
    claim_text: str
    verdict: str         # SUPPORTED | UNSUPPORTED | PARTIALLY_SUPPORTED | UNVERIFIABLE
    cited_chunk_ids: List[str]
    evidence_overlap: float
    reason: str


@dataclass(frozen=True)
class VerificationResult:
    """Output of the Verifier Agent (§4.5)."""
    query_id: str
    overall_verdict: str          # PASS | FAIL | PARTIAL
    overall_confidence: float
    per_claim: List[ClaimVerification]
    failed_claims: List[str]
    verification_model: str = ""
    verification_method: str = "deterministic"


# ── Compliance Agent Contracts (§4.6) ────────────────────────────────

@dataclass(frozen=True)
class ComplianceViolation:
    """A single compliance rule violation."""
    rule_id: str
    severity: str      # BLOCK | WARN
    description: str
    action_taken: str  # "redacted" | "blocked" | "disclaimer_appended"


@dataclass(frozen=True)
class ComplianceResult:
    """Output of the Compliance Agent (§4.6)."""
    query_id: str
    passed: bool
    violations: List[ComplianceViolation]
    redactions_applied: int = 0
    disclaimers_added: List[str] = field(default_factory=list)


# ── Explainability Contracts (§4.7) ──────────────────────────────────

@dataclass(frozen=True)
class EvidenceLink:
    """Maps a claim in the answer to its evidence chain."""
    claim_text: str
    citation_id: str
    chunk_id: str
    page_numbers: List[int]
    polygons: List[Dict[str, Any]]


@dataclass(frozen=True)
class DecisionStep:
    """One step in the decision provenance chain."""
    step_name: str
    agent: str
    decision: str
    reason: str
    timestamp: str


@dataclass(frozen=True)
class ExplainabilityReport:
    """Output of the Explainability Agent (§4.7)."""
    query_id: str
    timestamp: str
    # Level 1 — Decision Provenance
    decision_chain: List[DecisionStep]
    evidence_map: List[EvidenceLink]
    # Level 2 — Model Attribution
    models_used: List[ModelAttribution]
    total_cost: float = 0.0
    # Levels 3-4 are computed on-demand
    sensitivity_analysis: Optional[Dict[str, Any]] = None
    calibration_report: Optional[Dict[str, Any]] = None


# ── Orchestrator Contracts (§4.1) ────────────────────────────────────

@dataclass(frozen=True)
class ConversationTurn:
    """A single turn in the conversation."""
    role: str            # "user" | "assistant"
    content: str
    query_id: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass(frozen=True)
class ConversationMemory:
    """Conversational context for coreference resolution (§6.2)."""
    session_id: str
    user_id: str = ""
    recent_turns: List[ConversationTurn] = field(default_factory=list)
    active_doc_ids: List[str] = field(default_factory=list)
    unresolved_references: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class OrchestratorInput:
    """Input to the Orchestrator Agent (§4.1)."""
    query_id: str
    user_query: str
    conversation_memory: ConversationMemory
    document_scope: DocumentScope
    user_permissions: PermissionSet
    output_format: OutputFormat = field(default_factory=OutputFormat)
    token_budget: int = 50000


@dataclass(frozen=True)
class ExecutionStep:
    """Record of a single step in the orchestrator's execution trace."""
    step_id: str
    agent: str
    action: str
    status: str       # "pending" | "running" | "completed" | "failed"
    result_summary: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    timestamp: str = ""


@dataclass(frozen=True)
class OrchestratorOutput:
    """Output of the Orchestrator Agent (§4.1)."""
    query_id: str
    answer: str
    citations: List[Citation]
    confidence: float
    explainability_report: Optional[ExplainabilityReport] = None
    execution_trace: List[ExecutionStep] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    warnings: List[str] = field(default_factory=list)


# ── Audit Log Entry (§2.4) ──────────────────────────────────────────

@dataclass(frozen=True)
class AuditLogEntry:
    """Immutable audit record for every LLM call (§2.4)."""
    log_id: str
    query_id: str
    agent_id: str
    step_id: str
    event_type: str            # "llm_call" | "retrieval" | "verification" | "compliance"
    model_id: str
    prompt_template_version: str
    full_prompt: str
    full_response: str
    input_tokens: int
    output_tokens: int
    temperature: float
    latency_ms: float
    cost_estimate: float
    user_id: str
    timestamp: str             # ISO 8601


# ── Classifier Agent Contracts (§4.8) ─────────────────────────────

@dataclass(frozen=True)
class ClassificationResult:
    """Output of the Classifier Agent — assigns document_type and classification_label."""
    doc_id: str
    document_type: str              # e.g. "10-K", "annual_report", "contract", "regulatory_filing"
    classification_label: str       # e.g. "sec_filing", "basel_pillar3", "financial_statement"
    confidence: float               # 0.0–1.0
    classification_method: str      # "deterministic" | "llm" | "memory_match"
    evidence_signals: Dict[str, Any] = field(default_factory=dict)  # signals used for classification
    memory_matches: List[str] = field(default_factory=list)         # prior doc_ids that informed this


@dataclass(frozen=True)
class ClassificationMemoryEntry:
    """A learned classification pattern stored in classification memory."""
    pattern_id: str
    document_type: str
    classification_label: str
    filename_pattern: Optional[str] = None
    title_keywords: List[str] = field(default_factory=list)
    structural_signals: Dict[str, Any] = field(default_factory=dict)
    success_count: int = 0
    total_count: int = 0
    last_used: str = ""

    @property
    def accuracy(self) -> float:
        return self.success_count / self.total_count if self.total_count > 0 else 0.0


# ── Preprocessor Agent Contracts (§4.9) ─────────────────────────────

@dataclass(frozen=True)
class ChunkingStrategy:
    """Defines HOW a document should be chunked.

    processing_level controls the depth of processing:
    - ``skip``: No processing at all (empty/corrupt documents).
    - ``metadata_only``: Extract text and store as document facts; no
      chunking or embedding.  Suitable for identity documents (passport,
      driving licence) and simple proof-of-existence documents.
    - ``single_chunk``: Embed the entire document as one chunk.  For
      short, simple documents (1-3 pages) where splitting would lose
      context.
    - ``page_level``: One chunk per page with independent embeddings.
      No macro/child late-chunking overhead.  For moderately simple
      multi-page documents (bank statements, certificates).
    - ``late_chunking``: Full macro → child late-chunking pipeline with
      configurable token windows.  For complex, long-form documents.
    """
    strategy_name: str              # "late_chunking" | "table_aware" | "contract_clause" | "regulatory_section" | "skip"
    processing_level: str = "late_chunking"  # "skip" | "metadata_only" | "single_chunk" | "page_level" | "late_chunking"
    macro_max_tokens: int = 8192
    macro_overlap_tokens: int = 256
    child_target_tokens: int = 256
    table_extraction: str = "span"  # "span" | "full_page" | "none"
    heading_aware: bool = True
    rationale: str = ""             # Why this strategy was chosen


@dataclass(frozen=True)
class SectionStrategy:
    """Maps a page range within a document to a specific chunking strategy.

    Enables multi-chunking: different parts of the same document can use
    different algorithms.  For example, a 10-K might use:
    - ``semantic`` for narrative risk factors (pages 5-30)
    - ``table_aware`` for financial statements (pages 31-50)
    - ``clause_aware`` for legal exhibits (pages 51-80)

    The PreprocessorAgent produces a list of SectionStrategy objects when
    ``multi_chunking`` is enabled, covering all pages of the document.
    """
    page_start: int                 # Inclusive (1-based)
    page_end: int                   # Inclusive (1-based)
    content_type: str               # "narrative" | "tabular" | "legal" | "mixed" | "boilerplate"
    chunking_strategy: ChunkingStrategy
    rationale: str = ""


@dataclass(frozen=True)
class PreprocessorInput:
    """Input to the Preprocessor Agent — document metadata for chunking decision."""
    doc_id: str
    filename: str
    page_count: int
    document_type: Optional[str] = None
    classification_label: Optional[str] = None
    classification_confidence: float = 0.0
    triage_summary: Dict[str, Any] = field(default_factory=dict)  # Aggregated page triage stats
    front_matter_text: str = ""


@dataclass(frozen=True)
class PreprocessorResult:
    """Output of the Preprocessor Agent — chunking decision + strategy.

    When ``section_strategies`` is non-empty, multi-chunking is active and
    each section gets its own chunking algorithm.  The top-level
    ``chunking_strategy`` serves as the default/fallback.
    """
    doc_id: str
    requires_chunking: bool
    chunking_strategy: ChunkingStrategy
    confidence: float               # 0.0–1.0 confidence in strategy choice
    decision_method: str            # "deterministic" | "learned" | "default"
    learned_from_doc_ids: List[str] = field(default_factory=list)  # Prior docs that informed this
    warnings: List[str] = field(default_factory=list)
    section_strategies: List[SectionStrategy] = field(default_factory=list)  # Multi-chunking sections


@dataclass(frozen=True)
class ChunkingOutcome:
    """Feedback record stored after chunking completes — used for learning."""
    doc_id: str
    strategy_name: str
    document_type: str
    classification_label: str
    page_count: int
    chunk_count: int
    avg_chunk_tokens: float
    table_chunk_ratio: float        # Fraction of chunks that are tables
    heading_chunk_ratio: float      # Fraction of chunks that are headings
    boilerplate_ratio: float        # Fraction of chunks that are boilerplate
    processing_time_ms: float
    quality_score: float = 0.0      # Optional quality metric (0–1)


def new_id() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())
