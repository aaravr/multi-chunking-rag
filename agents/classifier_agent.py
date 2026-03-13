"""Classifier Agent — self-learning document classification (MASTER_PROMPT §4.8).

Classifies ANY document into document_type and classification_label to determine
the processing pipeline. Uses a 4-tier classification strategy:

1. **Deterministic rules**: Filename patterns, structural signals (fast, free)
2. **Embedding similarity**: ModernBERT embeddings + cosine similarity against
   prior classified documents in memory (semantic matching)
3. **Incremental classifier**: sklearn SGDClassifier with partial_fit() —
   trained online from every classification, improves over time
4. **LLM classification**: Tier-2 model for ambiguous documents (fallback)

Self-learning loop:
- After each classification, the front-matter embedding + label are stored
- The SGDClassifier is incrementally trained via partial_fit()
- Cosine similarity against stored embeddings provides memory-based matching
- The orchestrator can send feedback to reinforce/correct classifications
- Memory persists via JSON export/import (embeddings + classifier state)
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder

from agents.base import BaseAgent
from agents.contracts import (
    AgentMessage,
    ClassificationMemoryEntry,
    ClassificationResult,
    new_id,
)
from agents.message_bus import MessageBus
from agents.model_gateway import ModelGateway

logger = logging.getLogger(__name__)

# ── Known document type patterns (deterministic tier) ─────────────────

_FILENAME_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # (regex, document_type, classification_label)
    (re.compile(r"10-K", re.IGNORECASE), "10-K", "sec_filing"),
    (re.compile(r"10-Q", re.IGNORECASE), "10-Q", "sec_filing"),
    (re.compile(r"20-F", re.IGNORECASE), "20-F", "sec_filing"),
    (re.compile(r"8-K", re.IGNORECASE), "8-K", "sec_filing"),
    (re.compile(r"annual[\s_-]?report", re.IGNORECASE), "annual_report", "financial_report"),
    (re.compile(r"interim[\s_-]?report", re.IGNORECASE), "interim_report", "financial_report"),
    (re.compile(r"quarterly[\s_-]?report", re.IGNORECASE), "quarterly_report", "financial_report"),
    (re.compile(r"pillar\s*3", re.IGNORECASE), "pillar3_disclosure", "basel_regulatory"),
    (re.compile(r"basel", re.IGNORECASE), "regulatory_disclosure", "basel_regulatory"),
    (re.compile(r"proxy[\s_-]?statement", re.IGNORECASE), "proxy_statement", "sec_filing"),
    (re.compile(r"prospectus", re.IGNORECASE), "prospectus", "offering_document"),
    (re.compile(r"contract|agreement", re.IGNORECASE), "contract", "legal_document"),
    (re.compile(r"loan[\s_-]?agreement", re.IGNORECASE), "loan_agreement", "legal_document"),
    (re.compile(r"indenture", re.IGNORECASE), "indenture", "legal_document"),
    (re.compile(r"policy", re.IGNORECASE), "policy_document", "governance"),
    (re.compile(r"esg|sustainability", re.IGNORECASE), "esg_report", "sustainability"),
]

_CONTENT_SIGNALS: List[Tuple[re.Pattern, str, str, float]] = [
    # (regex matching first-page text, document_type, classification_label, weight)
    (re.compile(r"UNITED STATES SECURITIES AND EXCHANGE COMMISSION", re.IGNORECASE), "sec_filing", "sec_filing", 0.95),
    (re.compile(r"ANNUAL REPORT PURSUANT TO SECTION 13", re.IGNORECASE), "10-K", "sec_filing", 0.95),
    (re.compile(r"QUARTERLY REPORT PURSUANT TO SECTION 13", re.IGNORECASE), "10-Q", "sec_filing", 0.95),
    (re.compile(r"Form\s+10-K", re.IGNORECASE), "10-K", "sec_filing", 0.90),
    (re.compile(r"Form\s+10-Q", re.IGNORECASE), "10-Q", "sec_filing", 0.90),
    (re.compile(r"Form\s+20-F", re.IGNORECASE), "20-F", "sec_filing", 0.90),
    (re.compile(r"Pillar\s*3\s+Disclosures?", re.IGNORECASE), "pillar3_disclosure", "basel_regulatory", 0.90),
    (re.compile(r"Basel\s+III", re.IGNORECASE), "regulatory_disclosure", "basel_regulatory", 0.80),
    (re.compile(r"Capital\s+Adequacy\s+Report", re.IGNORECASE), "capital_adequacy_report", "basel_regulatory", 0.85),
    (re.compile(r"Annual\s+Report\s+(?:and\s+)?(?:Financial\s+Statements?|Accounts)", re.IGNORECASE), "annual_report", "financial_report", 0.90),
    (re.compile(r"Consolidated\s+Financial\s+Statements?", re.IGNORECASE), "financial_statements", "financial_report", 0.80),
    (re.compile(r"Independent\s+Auditor.?s?\s+Report", re.IGNORECASE), "audited_report", "financial_report", 0.70),
    (re.compile(r"THIS\s+(?:AGREEMENT|CONTRACT)\s+is\s+(?:made|entered)", re.IGNORECASE), "contract", "legal_document", 0.90),
    (re.compile(r"Environmental,?\s*Social\s+and\s+Governance", re.IGNORECASE), "esg_report", "sustainability", 0.85),
    (re.compile(r"Sustainability\s+Report", re.IGNORECASE), "esg_report", "sustainability", 0.85),
    (re.compile(r"Risk\s+Management\s+Report", re.IGNORECASE), "risk_report", "risk_management", 0.80),
]

# Minimum accuracy for a memory pattern to be "trusted" (used before LLM)
MEMORY_TRUST_THRESHOLD = 0.75
# Minimum observations before a pattern can be trusted
MEMORY_MIN_OBSERVATIONS = 3
# Maximum memory entries to prevent unbounded growth
MEMORY_MAX_ENTRIES = 10000
# Cosine similarity threshold for embedding-based memory match
EMBEDDING_SIMILARITY_THRESHOLD = 0.85
# Minimum training samples before the SGD classifier is used
SGD_MIN_SAMPLES = 5


class ClassificationMemory:
    """Self-learning classification memory with embeddings + incremental classifier.

    Two ML mechanisms for learning from past classifications:

    1. **Embedding store** — stores ModernBERT embeddings of front-matter text
       alongside their labels. New documents are compared via cosine similarity
       to find the closest match. This provides semantic generalization beyond
       exact keyword/filename matching.

    2. **SGDClassifier** — sklearn's SGDClassifier with partial_fit() for true
       incremental (online) learning. Trained on embeddings as features, learns
       a linear decision boundary. Once it has seen enough samples, its
       predictions supplement embedding similarity.

    Also retains the original keyword/filename index for fast exact-match
    lookups (free, no model inference needed).
    """

    def __init__(self, embedding_dim: int = 768) -> None:
        # Original pattern store (keyword/filename index)
        self._patterns: Dict[str, ClassificationMemoryEntry] = {}
        self._filename_index: Dict[str, List[str]] = defaultdict(list)
        self._keyword_index: Dict[str, List[str]] = defaultdict(list)

        # ── Embedding-based memory ────────────────────────────────────
        self._embedding_dim = embedding_dim
        self._embeddings: List[np.ndarray] = []       # (N, embedding_dim)
        self._embedding_labels: List[str] = []         # "doc_type::label" compound key
        self._embedding_doc_types: List[str] = []
        self._embedding_class_labels: List[str] = []

        # ── Incremental SGD classifier ────────────────────────────────
        self._sgd: SGDClassifier = SGDClassifier(
            loss="modified_huber",     # outputs probability estimates
            penalty="l2",
            alpha=1e-4,
            max_iter=1,
            tol=None,
            warm_start=True,
            random_state=42,
        )
        self._label_encoder: LabelEncoder = LabelEncoder()
        self._sgd_classes: List[str] = []
        self._sgd_fitted: bool = False
        self._sgd_sample_count: int = 0

    # ── Embedding memory operations ──────────────────────────────────

    def store_embedding(
        self,
        embedding: np.ndarray,
        document_type: str,
        classification_label: str,
    ) -> None:
        """Store a document embedding with its classification for future similarity lookups.

        Also incrementally trains the SGDClassifier on this new sample.
        """
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        if emb.shape[0] != self._embedding_dim:
            logger.warning(
                "Embedding dim mismatch: expected %d, got %d",
                self._embedding_dim, emb.shape[0],
            )
            return

        compound_label = f"{document_type}::{classification_label}"
        self._embeddings.append(emb)
        self._embedding_labels.append(compound_label)
        self._embedding_doc_types.append(document_type)
        self._embedding_class_labels.append(classification_label)

        # ── Incremental training ──────────────────────────────────────
        self._train_incremental(emb, compound_label)

    def lookup_by_embedding(
        self,
        query_embedding: np.ndarray,
        threshold: float = EMBEDDING_SIMILARITY_THRESHOLD,
    ) -> Optional[Tuple[str, str, float]]:
        """Find the most similar stored embedding via cosine similarity.

        Returns (document_type, classification_label, similarity) or None
        if no match exceeds the threshold.
        """
        if not self._embeddings:
            return None

        query = np.asarray(query_embedding, dtype=np.float32).ravel()
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return None

        # Vectorized cosine similarity against all stored embeddings
        matrix = np.stack(self._embeddings)  # (N, dim)
        norms = np.linalg.norm(matrix, axis=1)
        # Avoid division by zero
        valid = norms > 0
        similarities = np.zeros(len(self._embeddings))
        if valid.any():
            similarities[valid] = (matrix[valid] @ query) / (norms[valid] * query_norm)

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim < threshold:
            return None

        return (
            self._embedding_doc_types[best_idx],
            self._embedding_class_labels[best_idx],
            best_sim,
        )

    def predict_sgd(
        self, query_embedding: np.ndarray
    ) -> Optional[Tuple[str, str, float]]:
        """Use the trained SGDClassifier to predict document classification.

        Returns (document_type, classification_label, confidence) or None
        if the classifier is not ready.
        """
        if not self._sgd_fitted or self._sgd_sample_count < SGD_MIN_SAMPLES:
            return None

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        try:
            proba = self._sgd.predict_proba(query)[0]
            best_idx = int(np.argmax(proba))
            confidence = float(proba[best_idx])
            compound_label = self._sgd_classes[best_idx]
            parts = compound_label.split("::", 1)
            if len(parts) != 2:
                return None
            return (parts[0], parts[1], confidence)
        except Exception as exc:
            logger.warning("SGD prediction failed: %s", exc)
            return None

    def _train_incremental(self, embedding: np.ndarray, compound_label: str) -> None:
        """Incrementally train the SGDClassifier with a single new sample.

        SGDClassifier.partial_fit() requires >= 2 distinct classes. Training
        is deferred until we have samples from at least 2 classes, then a
        batch fit bootstraps the classifier with all accumulated samples.
        """
        X = embedding.reshape(1, -1)

        if compound_label not in self._sgd_classes:
            self._sgd_classes.append(compound_label)

        # SGDClassifier requires >= 2 classes; defer until we have them
        if len(self._sgd_classes) < 2:
            self._sgd_sample_count += 1
            return

        # If this is the first time we have 2+ classes, bootstrap with all
        # stored embeddings rather than just this one sample
        if not self._sgd_fitted and len(self._embeddings) >= 2:
            self._retrain_sgd_from_store()
            return

        try:
            self._sgd.partial_fit(
                X,
                [compound_label],
                classes=self._sgd_classes,
            )
            self._sgd_fitted = True
            self._sgd_sample_count += 1
        except Exception as exc:
            logger.warning("SGD incremental training failed: %s", exc)

    # ── Original pattern store (keyword/filename) ─────────────────────

    def store_pattern(
        self,
        document_type: str,
        classification_label: str,
        filename: Optional[str] = None,
        title_keywords: Optional[List[str]] = None,
        structural_signals: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a new classification pattern or update an existing one.

        Returns the pattern_id.
        """
        existing = self._find_exact_match(
            document_type, classification_label, filename
        )
        if existing:
            old = self._patterns[existing]
            updated = ClassificationMemoryEntry(
                pattern_id=old.pattern_id,
                document_type=old.document_type,
                classification_label=old.classification_label,
                filename_pattern=old.filename_pattern,
                title_keywords=list(set(old.title_keywords + (title_keywords or []))),
                structural_signals={**old.structural_signals, **(structural_signals or {})},
                success_count=old.success_count,
                total_count=old.total_count + 1,
                last_used=datetime.now(timezone.utc).isoformat(),
            )
            self._patterns[existing] = updated
            return existing

        pattern_id = new_id()
        entry = ClassificationMemoryEntry(
            pattern_id=pattern_id,
            document_type=document_type,
            classification_label=classification_label,
            filename_pattern=filename,
            title_keywords=title_keywords or [],
            structural_signals=structural_signals or {},
            success_count=0,
            total_count=1,
            last_used=datetime.now(timezone.utc).isoformat(),
        )
        self._patterns[pattern_id] = entry

        if filename:
            normalized = self._normalize_filename(filename)
            self._filename_index[normalized].append(pattern_id)
        for kw in (title_keywords or []):
            self._keyword_index[kw.lower()].append(pattern_id)

        if len(self._patterns) > MEMORY_MAX_ENTRIES:
            self._prune_lowest_accuracy()

        return pattern_id

    def lookup(
        self,
        filename: Optional[str] = None,
        title_keywords: Optional[List[str]] = None,
        min_accuracy: float = MEMORY_TRUST_THRESHOLD,
        min_observations: int = MEMORY_MIN_OBSERVATIONS,
    ) -> Optional[ClassificationMemoryEntry]:
        """Find the best matching trusted pattern from the keyword/filename index."""
        candidates: List[ClassificationMemoryEntry] = []

        if filename:
            normalized = self._normalize_filename(filename)
            for pid in self._filename_index.get(normalized, []):
                entry = self._patterns.get(pid)
                if entry and entry.total_count >= min_observations and entry.accuracy >= min_accuracy:
                    candidates.append(entry)

        if title_keywords:
            for kw in title_keywords:
                for pid in self._keyword_index.get(kw.lower(), []):
                    entry = self._patterns.get(pid)
                    if entry and entry.total_count >= min_observations and entry.accuracy >= min_accuracy:
                        candidates.append(entry)

        if not candidates:
            return None

        return max(candidates, key=lambda e: (e.accuracy, e.total_count))

    def record_feedback(self, pattern_id: str, correct: bool) -> None:
        """Record whether a classification was correct (reinforcement learning)."""
        entry = self._patterns.get(pattern_id)
        if not entry:
            return
        updated = ClassificationMemoryEntry(
            pattern_id=entry.pattern_id,
            document_type=entry.document_type,
            classification_label=entry.classification_label,
            filename_pattern=entry.filename_pattern,
            title_keywords=entry.title_keywords,
            structural_signals=entry.structural_signals,
            success_count=entry.success_count + (1 if correct else 0),
            total_count=entry.total_count,
            last_used=entry.last_used,
        )
        self._patterns[pattern_id] = updated

    def get_all_patterns(self) -> List[ClassificationMemoryEntry]:
        """Return all stored patterns (for debugging/export)."""
        return list(self._patterns.values())

    def get_stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        patterns = list(self._patterns.values())
        trusted = [p for p in patterns if p.accuracy >= MEMORY_TRUST_THRESHOLD and p.total_count >= MEMORY_MIN_OBSERVATIONS]
        return {
            "total_patterns": len(patterns),
            "trusted_patterns": len(trusted),
            "total_observations": sum(p.total_count for p in patterns),
            "avg_accuracy": (
                sum(p.accuracy for p in patterns) / len(patterns) if patterns else 0.0
            ),
            "embedding_count": len(self._embeddings),
            "sgd_fitted": self._sgd_fitted,
            "sgd_sample_count": self._sgd_sample_count,
            "sgd_classes": list(self._sgd_classes),
        }

    def export_json(self) -> str:
        """Export memory to JSON for persistence (patterns + embeddings + SGD state)."""
        data = {
            "patterns": [],
            "embeddings": [],
            "sgd_classes": self._sgd_classes,
            "sgd_sample_count": self._sgd_sample_count,
        }
        for entry in self._patterns.values():
            data["patterns"].append({
                "pattern_id": entry.pattern_id,
                "document_type": entry.document_type,
                "classification_label": entry.classification_label,
                "filename_pattern": entry.filename_pattern,
                "title_keywords": entry.title_keywords,
                "structural_signals": entry.structural_signals,
                "success_count": entry.success_count,
                "total_count": entry.total_count,
                "last_used": entry.last_used,
            })
        for i, emb in enumerate(self._embeddings):
            data["embeddings"].append({
                "embedding": emb.tolist(),
                "document_type": self._embedding_doc_types[i],
                "classification_label": self._embedding_class_labels[i],
            })
        return json.dumps(data, indent=2)

    def import_json(self, raw: str) -> int:
        """Import patterns + embeddings from JSON. Returns count of items loaded."""
        data = json.loads(raw)
        count = 0

        # Handle both old format (list) and new format (dict with keys)
        if isinstance(data, list):
            patterns = data
            embeddings = []
        else:
            patterns = data.get("patterns", [])
            embeddings = data.get("embeddings", [])
            self._sgd_classes = data.get("sgd_classes", [])
            self._sgd_sample_count = data.get("sgd_sample_count", 0)

        for item in patterns:
            pid = item["pattern_id"]
            entry = ClassificationMemoryEntry(
                pattern_id=pid,
                document_type=item["document_type"],
                classification_label=item["classification_label"],
                filename_pattern=item.get("filename_pattern"),
                title_keywords=item.get("title_keywords", []),
                structural_signals=item.get("structural_signals", {}),
                success_count=item.get("success_count", 0),
                total_count=item.get("total_count", 1),
                last_used=item.get("last_used", ""),
            )
            self._patterns[pid] = entry
            if entry.filename_pattern:
                self._filename_index[self._normalize_filename(entry.filename_pattern)].append(pid)
            for kw in entry.title_keywords:
                self._keyword_index[kw.lower()].append(pid)
            count += 1

        for item in embeddings:
            emb = np.array(item["embedding"], dtype=np.float32)
            self._embeddings.append(emb)
            self._embedding_doc_types.append(item["document_type"])
            self._embedding_class_labels.append(item["classification_label"])
            compound = f"{item['document_type']}::{item['classification_label']}"
            self._embedding_labels.append(compound)
            count += 1

        # Retrain SGD on imported embeddings
        if self._embeddings and self._sgd_classes:
            self._retrain_sgd_from_store()

        return count

    def _retrain_sgd_from_store(self) -> None:
        """Retrain SGDClassifier from all stored embeddings (used after import or bootstrap)."""
        if not self._embeddings or len(self._sgd_classes) < 2:
            return
        X = np.stack(self._embeddings)
        y = self._embedding_labels
        try:
            self._sgd.partial_fit(X, y, classes=self._sgd_classes)
            self._sgd_fitted = True
            self._sgd_sample_count = len(self._embeddings)
        except Exception as exc:
            logger.warning("SGD retrain failed: %s", exc)

    def _find_exact_match(
        self, document_type: str, classification_label: str, filename: Optional[str]
    ) -> Optional[str]:
        for pid, entry in self._patterns.items():
            if (
                entry.document_type == document_type
                and entry.classification_label == classification_label
                and entry.filename_pattern == filename
            ):
                return pid
        return None

    def _normalize_filename(self, filename: str) -> str:
        name = re.sub(r"\.[^.]+$", "", filename)
        name = re.sub(r"[\s_-]+", " ", name.lower().strip())
        return name

    def _prune_lowest_accuracy(self) -> None:
        sorted_patterns = sorted(
            self._patterns.items(),
            key=lambda kv: (kv[1].accuracy, kv[1].total_count),
        )
        to_remove = max(1, len(sorted_patterns) // 10)
        for pid, _ in sorted_patterns[:to_remove]:
            del self._patterns[pid]


# Singleton memory instance
_classification_memory: Optional[ClassificationMemory] = None


def get_classification_memory() -> ClassificationMemory:
    """Get or create the singleton classification memory."""
    global _classification_memory
    if _classification_memory is None:
        _classification_memory = ClassificationMemory()
    return _classification_memory


def reset_classification_memory() -> None:
    """Reset the singleton (for testing)."""
    global _classification_memory
    _classification_memory = None


# ── Embedder helper ───────────────────────────────────────────────────

def _get_embedder():
    """Lazy-load the ModernBERT embedder singleton.

    Returns None if the model cannot be loaded (e.g. in test environments
    without GPU/model weights).
    """
    try:
        from embedding.model_registry import get_embedding_model
        return get_embedding_model()
    except Exception as exc:
        logger.debug("Could not load embedding model: %s", exc)
        return None


class ClassifierAgent(BaseAgent):
    """Self-learning document classifier (§4.8).

    Classification strategy (4-tier, cheapest first):

    1. **Deterministic rules** — regex on filename and first-page content.
       Fast, free, high confidence for known document types.

    2. **Embedding similarity** — ModernBERT embeds the front-matter text,
       cosine similarity finds the closest prior document in memory.

    3. **SGD classifier** — sklearn SGDClassifier trained incrementally
       via partial_fit() on every classification. Predicts from embeddings.

    4. **LLM classification** — Tier-2 model analyzes front-matter text.
       Used only for truly ambiguous documents.

    After classification, the result + embedding are stored in memory.
    The SGDClassifier is trained on each new sample. The orchestrator or
    user can send feedback to reinforce/correct.

    MUST:
    - Try deterministic rules before expensive methods
    - Store every classification in memory with its embedding
    - Incrementally train the SGD classifier
    - Respect token budget

    MUST NOT:
    - Call LLM when deterministic match has confidence >= 0.85
    - Exceed 1 LLM call per classification
    """

    agent_name = "classifier"

    def __init__(
        self,
        bus: MessageBus,
        gateway: Optional[ModelGateway] = None,
        memory: Optional[ClassificationMemory] = None,
        embedder: Any = None,
    ) -> None:
        super().__init__(bus, gateway)
        self._memory = memory or get_classification_memory()
        self._embedder = embedder  # lazy-loaded if None

    def _get_embedder(self):
        """Get or lazy-load the embedder."""
        if self._embedder is None:
            self._embedder = _get_embedder()
        return self._embedder

    def handle_message(self, message: AgentMessage) -> ClassificationResult:
        """Handle a classification_request message.

        Expected payload keys:
            doc_id: str
            filename: str
            front_matter_text: str — concatenated text from first N pages
            page_count: int
            structural_signals: dict — optional signals from page triage
        """
        payload = message.payload
        return self.classify(
            doc_id=payload["doc_id"],
            filename=payload["filename"],
            front_matter_text=payload.get("front_matter_text", ""),
            page_count=payload.get("page_count", 0),
            structural_signals=payload.get("structural_signals", {}),
            query_id=message.query_id,
        )

    def classify(
        self,
        doc_id: str,
        filename: str,
        front_matter_text: str = "",
        page_count: int = 0,
        structural_signals: Optional[Dict[str, Any]] = None,
        query_id: str = "",
    ) -> ClassificationResult:
        """Classify a document using the 4-tier strategy.

        Returns a ClassificationResult with document_type, classification_label,
        confidence, and the method used.
        """
        start = time.monotonic()
        signals = dict(structural_signals or {})
        signals["filename"] = filename
        signals["page_count"] = page_count

        # ── Compute embedding (used for memory storage + tiers 2-3) ──
        embedding = self._embed_text(front_matter_text) if front_matter_text else None

        # ── Tier 1: Deterministic rules ───────────────────────────────
        det_result = self._classify_deterministic(filename, front_matter_text)
        if det_result and det_result[2] >= 0.85:
            doc_type, label, confidence = det_result
            result = ClassificationResult(
                doc_id=doc_id,
                document_type=doc_type,
                classification_label=label,
                confidence=confidence,
                classification_method="deterministic",
                evidence_signals=signals,
            )
            self._store_in_memory(result, filename, front_matter_text, embedding)
            self._log_classification(result, start, query_id)
            return result

        # ── Tier 2: Embedding similarity ─────────────────────────────
        if embedding is not None:
            sim_result = self._memory.lookup_by_embedding(embedding)
            if sim_result:
                doc_type, label, similarity = sim_result
                result = ClassificationResult(
                    doc_id=doc_id,
                    document_type=doc_type,
                    classification_label=label,
                    confidence=similarity,
                    classification_method="embedding_similarity",
                    evidence_signals=signals,
                )
                self._store_in_memory(result, filename, front_matter_text, embedding)
                self._log_classification(result, start, query_id)
                return result

        # ── Tier 3: SGD classifier ───────────────────────────────────
        if embedding is not None:
            sgd_result = self._memory.predict_sgd(embedding)
            if sgd_result and sgd_result[2] >= 0.6:
                doc_type, label, confidence = sgd_result
                result = ClassificationResult(
                    doc_id=doc_id,
                    document_type=doc_type,
                    classification_label=label,
                    confidence=confidence,
                    classification_method="sgd_classifier",
                    evidence_signals=signals,
                )
                self._store_in_memory(result, filename, front_matter_text, embedding)
                self._log_classification(result, start, query_id)
                return result

        # ── Tier 3.5: Keyword/filename memory match ──────────────────
        title_keywords = self._extract_title_keywords(front_matter_text)
        mem_match = self._memory.lookup(
            filename=filename,
            title_keywords=title_keywords,
        )
        if mem_match:
            result = ClassificationResult(
                doc_id=doc_id,
                document_type=mem_match.document_type,
                classification_label=mem_match.classification_label,
                confidence=min(mem_match.accuracy, 0.90),
                classification_method="memory_match",
                evidence_signals=signals,
                memory_matches=[mem_match.pattern_id],
            )
            self._memory.store_pattern(
                document_type=mem_match.document_type,
                classification_label=mem_match.classification_label,
                filename=filename,
                title_keywords=title_keywords,
            )
            self._log_classification(result, start, query_id)
            return result

        # ── Tier 4: LLM classification ───────────────────────────────
        if self.gateway and front_matter_text:
            llm_result = self._classify_with_llm(
                doc_id=doc_id,
                filename=filename,
                front_matter_text=front_matter_text,
                page_count=page_count,
                query_id=query_id,
            )
            if llm_result:
                result = ClassificationResult(
                    doc_id=doc_id,
                    document_type=llm_result[0],
                    classification_label=llm_result[1],
                    confidence=llm_result[2],
                    classification_method="llm",
                    evidence_signals=signals,
                )
                self._store_in_memory(result, filename, front_matter_text, embedding)
                self._log_classification(result, start, query_id)
                return result

        # ── Fallback ─────────────────────────────────────────────────
        if det_result:
            doc_type, label, confidence = det_result
        else:
            doc_type, label, confidence = "unknown", "unclassified", 0.0

        result = ClassificationResult(
            doc_id=doc_id,
            document_type=doc_type,
            classification_label=label,
            confidence=confidence,
            classification_method="deterministic" if det_result else "default",
            evidence_signals=signals,
        )
        self._store_in_memory(result, filename, front_matter_text, embedding)
        self._log_classification(result, start, query_id)
        return result

    def handle_feedback(self, pattern_id: str, correct: bool) -> None:
        """Process feedback on a classification (called by orchestrator)."""
        self._memory.record_feedback(pattern_id, correct)
        logger.info(
            "Classification feedback: pattern=%s correct=%s",
            pattern_id, correct,
        )

    def get_memory_stats(self) -> Dict[str, Any]:
        """Return classification memory statistics."""
        return self._memory.get_stats()

    # ── Embedding helper ─────────────────────────────────────────────

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """Embed front-matter text using the ModernBERT model.

        Returns a numpy array of shape (embedding_dim,) or None if
        the embedder is not available.
        """
        embedder = self._get_embedder()
        if embedder is None:
            return None
        try:
            # Truncate to first 2000 chars to keep embedding fast
            sample = text[:2000]
            vec = embedder.embed_text(sample)
            return np.array(vec, dtype=np.float32)
        except Exception as exc:
            logger.warning("Embedding failed: %s", exc)
            return None

    # ── Tier 1: Deterministic classification ──────────────────────────

    def _classify_deterministic(
        self, filename: str, front_matter_text: str
    ) -> Optional[Tuple[str, str, float]]:
        """Try deterministic classification using filename and content patterns."""
        best: Optional[Tuple[str, str, float]] = None

        for pattern, doc_type, label in _FILENAME_PATTERNS:
            if pattern.search(filename):
                confidence = 0.80
                if best is None or confidence > best[2]:
                    best = (doc_type, label, confidence)

        if front_matter_text:
            for pattern, doc_type, label, weight in _CONTENT_SIGNALS:
                if pattern.search(front_matter_text):
                    if best is None or weight > best[2]:
                        best = (doc_type, label, weight)

        return best

    # ── Tier 4: LLM classification ───────────────────────────────────

    def _classify_with_llm(
        self,
        doc_id: str,
        filename: str,
        front_matter_text: str,
        page_count: int,
        query_id: str,
    ) -> Optional[Tuple[str, str, float]]:
        """Use Tier-2 LLM to classify an ambiguous document."""
        text_sample = front_matter_text[:4000]
        prompt = _build_classification_prompt(filename, text_sample, page_count)

        try:
            result = self.gateway.call_model(
                model_id="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _CLASSIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                query_id=query_id,
                agent_id=self.agent_name,
                step_id=new_id(),
            )
            return self._parse_llm_response(result.get("content", ""))
        except Exception as exc:
            logger.warning("LLM classification failed: %s", exc)
            return None

    def _parse_llm_response(
        self, response: str
    ) -> Optional[Tuple[str, str, float]]:
        """Parse the LLM classification response."""
        try:
            json_str = response.strip()
            if "```" in json_str:
                json_str = re.search(r"```(?:json)?\s*(.*?)```", json_str, re.DOTALL)
                if json_str:
                    json_str = json_str.group(1).strip()
                else:
                    return None

            data = json.loads(json_str)
            doc_type = data.get("document_type", "unknown")
            label = data.get("classification_label", "unclassified")
            confidence = float(data.get("confidence", 0.5))
            return (doc_type, label, min(confidence, 0.95))
        except (json.JSONDecodeError, ValueError, AttributeError) as exc:
            logger.warning("Failed to parse LLM classification response: %s", exc)
            return None

    # ── Memory helpers ────────────────────────────────────────────────

    def _store_in_memory(
        self,
        result: ClassificationResult,
        filename: str,
        front_matter_text: str,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Store a classification result in memory for future learning."""
        title_keywords = self._extract_title_keywords(front_matter_text)
        self._memory.store_pattern(
            document_type=result.document_type,
            classification_label=result.classification_label,
            filename=filename,
            title_keywords=title_keywords,
            structural_signals=result.evidence_signals,
        )
        # Store embedding for similarity-based and SGD-based learning
        if embedding is not None:
            self._memory.store_embedding(
                embedding=embedding,
                document_type=result.document_type,
                classification_label=result.classification_label,
            )

    def _extract_title_keywords(self, text: str) -> List[str]:
        """Extract likely title/heading keywords from front-matter text."""
        if not text:
            return []
        sample = text[:500]
        phrases = re.findall(r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+", sample)
        caps = re.findall(r"\b[A-Z]{3,}(?:\s+[A-Z]{3,})+\b", sample)
        keywords = list(set(phrases + caps))[:10]
        return keywords

    def _log_classification(
        self, result: ClassificationResult, start: float, query_id: str
    ) -> None:
        """Log the classification result and record eval metrics."""
        latency_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Classifier: doc=%s type=%s label=%s conf=%.2f method=%s (%.0fms)",
            result.doc_id,
            result.document_type,
            result.classification_label,
            result.confidence,
            result.classification_method,
            latency_ms,
        )

        from agents.agent_eval import EvalCase, get_evaluator
        get_evaluator().record(EvalCase(
            query_id=query_id or new_id(),
            agent_name=self.agent_name,
            latency_ms=latency_ms,
            answer_confidence=result.confidence,
        ))


# ── LLM Prompt for classification ────────────────────────────────────

_CLASSIFICATION_SYSTEM_PROMPT = """\
You are a document classification expert. Your task is to determine the type \
and category of a document based on its filename and front-matter text.

You MUST respond with a JSON object containing exactly these fields:
- "document_type": A specific document type identifier (e.g., "10-K", "annual_report", \
"contract", "pillar3_disclosure", "loan_agreement", "esg_report", "policy_document", \
"research_paper", "invoice", "regulatory_filing")
- "classification_label": A broader category label (e.g., "sec_filing", "financial_report", \
"legal_document", "basel_regulatory", "sustainability", "governance", "academic", \
"commercial", "regulatory")
- "confidence": A float between 0 and 1 indicating your confidence

Respond with ONLY the JSON object, no other text."""


def _build_classification_prompt(
    filename: str, text_sample: str, page_count: int
) -> str:
    """Build the user prompt for LLM classification."""
    return f"""\
Classify this document:

Filename: {filename}
Page count: {page_count}

Front-matter text (first pages):
---
{text_sample}
---

Return your classification as JSON."""
