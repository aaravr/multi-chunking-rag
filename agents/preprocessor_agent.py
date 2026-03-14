"""Preprocessor Agent — determines chunking strategy per document (MASTER_PROMPT §4.9).

Sits between document classification and the chunking pipeline. Decides:
1. Whether chunking is required at all (e.g., skip very short docs)
2. Which chunking strategy to use based on document type and learned outcomes

Uses a 3-tier decision strategy:
1. **Deterministic rules**: Document type → known optimal strategy (fast, free)
2. **Learned outcomes**: Query past chunking outcomes for same doc type/label
   from the chunking_outcomes store. If prior outcomes show a strategy works
   well (high quality_score, reasonable chunk counts), reuse it.
3. **Default fallback**: Standard late-chunking with default parameters.

Self-learning loop:
- After chunking completes, the ingestion pipeline calls ``record_outcome()``
  with chunk statistics (count, type ratios, processing time).
- These outcomes are stored in-memory (session) and optionally in PostgreSQL
  via the ``chunking_outcomes`` table.
- Future documents of the same type consult these outcomes to pick the best
  strategy, creating a feedback loop that improves over time.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from agents.base import BaseAgent
from agents.contracts import (
    AgentMessage,
    ChunkingOutcome,
    ChunkingStrategy,
    PreprocessorInput,
    PreprocessorResult,
    new_id,
)
from agents.message_bus import MessageBus
from agents.model_gateway import ModelGateway
from core.config import settings

logger = logging.getLogger(__name__)

# ── Minimum page threshold: docs below this are too small to benefit from
#    anything other than default chunking ──────────────────────────────────
MIN_PAGES_FOR_STRATEGY_SELECTION = 3

# ── Minimum outcomes needed before trusting learned strategy ─────────────
MIN_OUTCOMES_FOR_LEARNING = 2

# ── Quality score threshold: only reuse strategies that scored above this ─
QUALITY_THRESHOLD = 0.5

# ── Document type → chunking strategy rules (deterministic tier) ─────────
#    Maps (document_type OR classification_label) to a named strategy with
#    tuned parameters based on domain knowledge of financial document types.

_STRATEGY_RULES: Dict[str, ChunkingStrategy] = {
    # SEC filings: dense text, long sections, benefit from larger macro chunks
    "10-K": ChunkingStrategy(
        strategy_name="sec_filing",
        macro_max_tokens=8192,
        macro_overlap_tokens=512,
        child_target_tokens=384,
        table_extraction="span",
        heading_aware=True,
        rationale="SEC 10-K filings have long narrative sections with embedded tables; "
                  "larger overlap preserves cross-reference context.",
    ),
    "10-Q": ChunkingStrategy(
        strategy_name="sec_filing",
        macro_max_tokens=8192,
        macro_overlap_tokens=512,
        child_target_tokens=384,
        table_extraction="span",
        heading_aware=True,
        rationale="SEC 10-Q quarterly filings follow 10-K structure.",
    ),
    "20-F": ChunkingStrategy(
        strategy_name="sec_filing",
        macro_max_tokens=8192,
        macro_overlap_tokens=512,
        child_target_tokens=384,
        table_extraction="span",
        heading_aware=True,
        rationale="SEC 20-F foreign filings follow 10-K structure.",
    ),
    "sec_filing": ChunkingStrategy(
        strategy_name="sec_filing",
        macro_max_tokens=8192,
        macro_overlap_tokens=512,
        child_target_tokens=384,
        table_extraction="span",
        heading_aware=True,
        rationale="SEC filings benefit from larger overlap for cross-references.",
    ),

    # Annual/financial reports: mix of narrative and tables
    "annual_report": ChunkingStrategy(
        strategy_name="financial_report",
        macro_max_tokens=8192,
        macro_overlap_tokens=256,
        child_target_tokens=256,
        table_extraction="span",
        heading_aware=True,
        rationale="Annual reports have balanced narrative/tabular content; "
                  "standard parameters work well.",
    ),
    "financial_report": ChunkingStrategy(
        strategy_name="financial_report",
        macro_max_tokens=8192,
        macro_overlap_tokens=256,
        child_target_tokens=256,
        table_extraction="span",
        heading_aware=True,
        rationale="Financial reports: balanced narrative and tables.",
    ),

    # Basel/regulatory: highly structured, table-heavy
    "pillar3_disclosure": ChunkingStrategy(
        strategy_name="regulatory_section",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=192,
        table_extraction="full_page",
        heading_aware=True,
        rationale="Pillar 3 disclosures are table-heavy with strict section structure; "
                  "smaller macro chunks preserve table boundaries.",
    ),
    "basel_regulatory": ChunkingStrategy(
        strategy_name="regulatory_section",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=192,
        table_extraction="full_page",
        heading_aware=True,
        rationale="Basel regulatory docs are table-heavy with strict sections.",
    ),
    "regulatory_disclosure": ChunkingStrategy(
        strategy_name="regulatory_section",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=192,
        table_extraction="full_page",
        heading_aware=True,
        rationale="Regulatory disclosures: structured sections and tables.",
    ),

    # Legal documents: clause-oriented, need tight boundaries
    "contract": ChunkingStrategy(
        strategy_name="contract_clause",
        macro_max_tokens=4096,
        macro_overlap_tokens=128,
        child_target_tokens=192,
        table_extraction="none",
        heading_aware=True,
        rationale="Contracts are clause-structured; smaller chunks keep clauses intact.",
    ),
    "loan_agreement": ChunkingStrategy(
        strategy_name="contract_clause",
        macro_max_tokens=4096,
        macro_overlap_tokens=128,
        child_target_tokens=192,
        table_extraction="none",
        heading_aware=True,
        rationale="Loan agreements follow contract clause patterns.",
    ),
    "legal_document": ChunkingStrategy(
        strategy_name="contract_clause",
        macro_max_tokens=4096,
        macro_overlap_tokens=128,
        child_target_tokens=192,
        table_extraction="none",
        heading_aware=True,
        rationale="Legal documents: clause-oriented structure.",
    ),
    "indenture": ChunkingStrategy(
        strategy_name="contract_clause",
        macro_max_tokens=4096,
        macro_overlap_tokens=128,
        child_target_tokens=192,
        table_extraction="none",
        heading_aware=True,
        rationale="Indentures follow contract clause structure.",
    ),

    # ESG / sustainability reports: narrative-heavy with infographics
    "esg_report": ChunkingStrategy(
        strategy_name="sustainability_report",
        macro_max_tokens=8192,
        macro_overlap_tokens=256,
        child_target_tokens=320,
        table_extraction="span",
        heading_aware=True,
        rationale="ESG reports have long narrative sections with embedded metrics; "
                  "larger child targets keep metric context together.",
    ),
    "sustainability": ChunkingStrategy(
        strategy_name="sustainability_report",
        macro_max_tokens=8192,
        macro_overlap_tokens=256,
        child_target_tokens=320,
        table_extraction="span",
        heading_aware=True,
        rationale="Sustainability reports: narrative-heavy with metrics.",
    ),
}

# ── Default strategy for unknown document types ──────────────────────────
_DEFAULT_STRATEGY = ChunkingStrategy(
    strategy_name="late_chunking",
    macro_max_tokens=8192,
    macro_overlap_tokens=256,
    child_target_tokens=256,
    table_extraction="span",
    heading_aware=True,
    rationale="Default late-chunking strategy for unclassified documents.",
)

# ── Skip strategy for very short documents ───────────────────────────────
_SKIP_STRATEGY = ChunkingStrategy(
    strategy_name="skip",
    macro_max_tokens=0,
    macro_overlap_tokens=0,
    child_target_tokens=0,
    table_extraction="none",
    heading_aware=False,
    rationale="Document is empty or has no extractable text content.",
)


# ── Outcome Store: learns from past chunking results ─────────────────────

class OutcomeStore:
    """In-memory store for chunking outcomes, keyed by (document_type, classification_label).

    Persists outcomes to PostgreSQL ``chunking_outcomes`` table when available,
    falls back to in-memory storage for tests or when no DB is configured.
    """

    def __init__(self, use_db: bool = False) -> None:
        self._use_db = use_db
        self._outcomes: Dict[str, List[ChunkingOutcome]] = defaultdict(list)

    def record(self, outcome: ChunkingOutcome) -> None:
        """Store a chunking outcome for future strategy selection."""
        key = _outcome_key(outcome.document_type, outcome.classification_label)
        self._outcomes[key].append(outcome)
        if self._use_db:
            self._persist_to_db(outcome)

    def lookup(
        self,
        document_type: str,
        classification_label: str,
        min_outcomes: int = MIN_OUTCOMES_FOR_LEARNING,
        min_quality: float = QUALITY_THRESHOLD,
    ) -> Optional[ChunkingStrategy]:
        """Find the best strategy for a given doc type based on past outcomes.

        Returns None if insufficient data or no strategy meets quality threshold.
        """
        key = _outcome_key(document_type, classification_label)
        outcomes = self._outcomes.get(key, [])

        # Also try DB if in-memory is insufficient
        if len(outcomes) < min_outcomes and self._use_db:
            db_outcomes = self._fetch_from_db(document_type, classification_label)
            # Merge, dedup by doc_id
            seen = {o.doc_id for o in outcomes}
            for o in db_outcomes:
                if o.doc_id not in seen:
                    outcomes.append(o)
                    seen.add(o.doc_id)

        if len(outcomes) < min_outcomes:
            return None

        # Group by strategy_name and pick the one with best average quality
        strategy_scores: Dict[str, List[float]] = defaultdict(list)
        strategy_params: Dict[str, ChunkingOutcome] = {}
        for o in outcomes:
            strategy_scores[o.strategy_name].append(o.quality_score)
            strategy_params[o.strategy_name] = o  # Keep latest params

        best_strategy = None
        best_avg_quality = -1.0
        for name, scores in strategy_scores.items():
            avg = sum(scores) / len(scores)
            if avg >= min_quality and avg > best_avg_quality:
                best_avg_quality = avg
                best_strategy = name

        if best_strategy is None:
            return None

        # Reconstruct strategy from the deterministic rules if available,
        # otherwise use the outcome's parameters as-is
        if best_strategy in _STRATEGY_RULES:
            return _STRATEGY_RULES[best_strategy]

        # Build from learned outcome
        ref = strategy_params[best_strategy]
        return ChunkingStrategy(
            strategy_name=best_strategy,
            macro_max_tokens=8192,
            macro_overlap_tokens=256,
            child_target_tokens=256,
            table_extraction="span" if ref.table_chunk_ratio > 0.1 else "none",
            heading_aware=ref.heading_chunk_ratio > 0.05,
            rationale=f"Learned from {len(strategy_scores[best_strategy])} prior "
                      f"documents (avg quality={best_avg_quality:.2f}).",
        )

    def get_doc_ids_for_type(
        self, document_type: str, classification_label: str
    ) -> List[str]:
        """Return doc_ids of prior documents that informed strategy for this type."""
        key = _outcome_key(document_type, classification_label)
        return [o.doc_id for o in self._outcomes.get(key, [])]

    @property
    def total_outcomes(self) -> int:
        return sum(len(v) for v in self._outcomes.values())

    def _persist_to_db(self, outcome: ChunkingOutcome) -> None:
        """Store outcome in PostgreSQL (best-effort)."""
        try:
            from storage.db_pool import get_connection
            from storage import repo
            with get_connection() as conn:
                repo.insert_chunking_outcome(conn, outcome)
                conn.commit()
        except Exception as exc:
            logger.debug("Failed to persist chunking outcome to DB: %s", exc)

    def _fetch_from_db(
        self, document_type: str, classification_label: str
    ) -> List[ChunkingOutcome]:
        """Fetch outcomes from PostgreSQL (best-effort)."""
        try:
            from storage.db_pool import get_connection
            from storage import repo
            with get_connection() as conn:
                return repo.fetch_chunking_outcomes(
                    conn, document_type, classification_label
                )
        except Exception as exc:
            logger.debug("Failed to fetch chunking outcomes from DB: %s", exc)
            return []


def _outcome_key(document_type: str, classification_label: str) -> str:
    return f"{document_type}::{classification_label}"


# ── Module-level singleton ───────────────────────────────────────────────

_outcome_store: Optional[OutcomeStore] = None


def get_outcome_store() -> OutcomeStore:
    """Get or create the module-level OutcomeStore singleton."""
    global _outcome_store
    if _outcome_store is None:
        _outcome_store = OutcomeStore(use_db=bool(settings.database_url))
    return _outcome_store


# ── Preprocessor Agent ───────────────────────────────────────────────────

class PreprocessorAgent(BaseAgent):
    """Determines if chunking is required and selects the optimal strategy.

    Decision flow (3-tier):
    1. Check if document is too short/empty → skip
    2. Deterministic: document_type or classification_label → known strategy
    3. Learned: consult OutcomeStore for past outcomes on same doc type
    4. Default: standard late-chunking with default parameters

    The agent also exposes ``record_outcome()`` for the ingestion pipeline
    to feed back chunking results, enabling the learning loop.
    """

    agent_name = "preprocessor"

    def __init__(
        self,
        bus: MessageBus,
        gateway: Optional[ModelGateway] = None,
        outcome_store: Optional[OutcomeStore] = None,
    ) -> None:
        super().__init__(bus, gateway)
        self._outcome_store = outcome_store or get_outcome_store()

    def handle_message(self, message: AgentMessage) -> PreprocessorResult:
        """Handle an incoming preprocessor request via the message bus."""
        payload = message.payload
        inp = PreprocessorInput(
            doc_id=payload.get("doc_id", ""),
            filename=payload.get("filename", ""),
            page_count=payload.get("page_count", 0),
            document_type=payload.get("document_type"),
            classification_label=payload.get("classification_label"),
            classification_confidence=payload.get("classification_confidence", 0.0),
            triage_summary=payload.get("triage_summary", {}),
            front_matter_text=payload.get("front_matter_text", ""),
        )
        return self.determine_strategy(inp)

    def determine_strategy(self, inp: PreprocessorInput) -> PreprocessorResult:
        """Core decision logic: determine chunking strategy for a document.

        Returns a PreprocessorResult with:
        - requires_chunking: whether to proceed with chunking
        - chunking_strategy: the selected ChunkingStrategy
        - confidence: how confident the decision is
        - decision_method: which tier made the decision
        """
        start_ms = time.monotonic() * 1000
        warnings: List[str] = []

        # ── Tier 0: Skip check ───────────────────────────────────────
        if inp.page_count == 0:
            return PreprocessorResult(
                doc_id=inp.doc_id,
                requires_chunking=False,
                chunking_strategy=_SKIP_STRATEGY,
                confidence=1.0,
                decision_method="deterministic",
                warnings=["Document has 0 pages — skipping chunking."],
            )

        # Check triage summary for empty documents
        triage = inp.triage_summary
        if triage:
            total_text = triage.get("total_text_length", -1)
            if total_text == 0:
                return PreprocessorResult(
                    doc_id=inp.doc_id,
                    requires_chunking=False,
                    chunking_strategy=_SKIP_STRATEGY,
                    confidence=1.0,
                    decision_method="deterministic",
                    warnings=["Document has no extractable text — skipping chunking."],
                )

        # ── Tier 1: Deterministic rules ──────────────────────────────
        strategy = self._deterministic_lookup(
            inp.document_type, inp.classification_label
        )
        if strategy is not None:
            # Adjust strategy based on triage signals
            strategy = self._adjust_for_triage(strategy, triage, warnings)
            elapsed = time.monotonic() * 1000 - start_ms
            logger.info(
                "Preprocessor: deterministic strategy '%s' for doc %s "
                "(type=%s, label=%s) in %.1fms",
                strategy.strategy_name, inp.doc_id,
                inp.document_type, inp.classification_label, elapsed,
            )
            return PreprocessorResult(
                doc_id=inp.doc_id,
                requires_chunking=True,
                chunking_strategy=strategy,
                confidence=0.95,
                decision_method="deterministic",
                warnings=warnings,
            )

        # ── Tier 2: Learned from past outcomes ───────────────────────
        if inp.document_type and inp.classification_label:
            learned = self._outcome_store.lookup(
                inp.document_type, inp.classification_label
            )
            if learned is not None:
                learned = self._adjust_for_triage(learned, triage, warnings)
                prior_ids = self._outcome_store.get_doc_ids_for_type(
                    inp.document_type, inp.classification_label
                )
                elapsed = time.monotonic() * 1000 - start_ms
                logger.info(
                    "Preprocessor: learned strategy '%s' for doc %s "
                    "from %d prior docs in %.1fms",
                    learned.strategy_name, inp.doc_id,
                    len(prior_ids), elapsed,
                )
                return PreprocessorResult(
                    doc_id=inp.doc_id,
                    requires_chunking=True,
                    chunking_strategy=learned,
                    confidence=0.75,
                    decision_method="learned",
                    learned_from_doc_ids=prior_ids[:10],
                    warnings=warnings,
                )

        # ── Tier 3: Default fallback ─────────────────────────────────
        default = self._adjust_for_triage(_DEFAULT_STRATEGY, triage, warnings)
        elapsed = time.monotonic() * 1000 - start_ms
        logger.info(
            "Preprocessor: default strategy for doc %s in %.1fms",
            inp.doc_id, elapsed,
        )
        return PreprocessorResult(
            doc_id=inp.doc_id,
            requires_chunking=True,
            chunking_strategy=default,
            confidence=0.5,
            decision_method="default",
            warnings=warnings,
        )

    def record_outcome(self, outcome: ChunkingOutcome) -> None:
        """Record a chunking outcome for future learning.

        Called by the ingestion pipeline after chunking completes.
        The outcome's statistics (chunk count, type ratios, timing)
        are used to evaluate and improve strategy selection.
        """
        self._outcome_store.record(outcome)
        logger.info(
            "Preprocessor: recorded outcome for doc %s — strategy=%s, "
            "chunks=%d, quality=%.2f",
            outcome.doc_id, outcome.strategy_name,
            outcome.chunk_count, outcome.quality_score,
        )

    @property
    def outcome_store(self) -> OutcomeStore:
        return self._outcome_store

    # ── Internal helpers ─────────────────────────────────────────────

    def _deterministic_lookup(
        self,
        document_type: Optional[str],
        classification_label: Optional[str],
    ) -> Optional[ChunkingStrategy]:
        """Look up a known strategy by document_type or classification_label."""
        if document_type and document_type in _STRATEGY_RULES:
            return _STRATEGY_RULES[document_type]
        if classification_label and classification_label in _STRATEGY_RULES:
            return _STRATEGY_RULES[classification_label]
        return None

    def _adjust_for_triage(
        self,
        strategy: ChunkingStrategy,
        triage: Dict[str, Any],
        warnings: List[str],
    ) -> ChunkingStrategy:
        """Adjust strategy parameters based on page triage signals.

        If triage data shows the document is heavily image-based or has
        high layout complexity, we may want to tweak parameters.
        """
        if not triage:
            return strategy

        di_page_ratio = triage.get("di_page_ratio", 0.0)
        avg_image_coverage = triage.get("avg_image_coverage", 0.0)

        # If most pages needed DI (Azure Document Intelligence), the document
        # is likely image-heavy / scanned. Use full-page table extraction.
        if di_page_ratio > 0.5 and strategy.table_extraction == "span":
            warnings.append(
                f"High DI ratio ({di_page_ratio:.0%}) — upgrading table "
                "extraction to full_page."
            )
            return ChunkingStrategy(
                strategy_name=strategy.strategy_name,
                macro_max_tokens=strategy.macro_max_tokens,
                macro_overlap_tokens=strategy.macro_overlap_tokens,
                child_target_tokens=strategy.child_target_tokens,
                table_extraction="full_page",
                heading_aware=strategy.heading_aware,
                rationale=strategy.rationale + " [adjusted: full_page table extraction for DI-heavy doc]",
            )

        # If average image coverage is very high, warn about potential
        # low text yield
        if avg_image_coverage > 0.7:
            warnings.append(
                f"Very high image coverage ({avg_image_coverage:.0%}) — "
                "chunking may produce limited text content."
            )

        return strategy
