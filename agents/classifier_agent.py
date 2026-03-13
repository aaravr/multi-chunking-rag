"""Classifier Agent — self-learning document classification (MASTER_PROMPT §4.8).

Classifies ANY document into document_type and classification_label to determine
the processing pipeline. Uses a 4-tier classification strategy orchestrated by
a LangGraph StateGraph with conditional routing:

1. **Deterministic rules**: Filename patterns, structural signals (fast, free)
2. **pgvector similarity**: PostgreSQL + pgvector HNSW index over prior classified
   documents with cosine similarity retrieval for semantic memory matching
3. **Incremental classifier**: sklearn SGDClassifier with partial_fit() —
   trained online from every classification, improves over time
4. **LLM classification**: LangChain ChatPromptTemplate + Tier-2 model for
   ambiguous documents (fallback)

Storage backends:
- **pgvector**: classification_embeddings table with HNSW cosine index for
  persistent vector similarity (primary). Falls back to in-memory numpy when
  DATABASE_URL is not configured (e.g., unit tests).
- **Neo4j**: Knowledge graph for document→type→label relationships, entity
  mentions, and cross-document discovery. Feature-flagged via ENABLE_NEO4J.

Frameworks:
- **LangGraph**: StateGraph with conditional edges for classification flow
- **LangChain**: ChatPromptTemplate for structured LLM prompts
- **sklearn**: SGDClassifier for incremental online learning

Self-learning loop:
- After each classification, the front-matter embedding + label are stored
  in pgvector and the SGDClassifier is trained
- Classifications are also recorded in the Neo4j knowledge graph
- The orchestrator can send feedback to reinforce/correct classifications
- Memory persists via pgvector (embeddings) + classification_memory table (patterns)
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from sklearn.linear_model import SGDClassifier

from agents.base import BaseAgent
from agents.contracts import (
    AgentMessage,
    ClassificationMemoryEntry,
    ClassificationResult,
    new_id,
)
from agents.message_bus import MessageBus
from agents.model_gateway import ModelGateway
from core.config import settings

logger = logging.getLogger(__name__)

# ── Known document type patterns (deterministic tier) ─────────────────

_FILENAME_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
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

MEMORY_TRUST_THRESHOLD = 0.75
MEMORY_MIN_OBSERVATIONS = 3
MEMORY_MAX_ENTRIES = 10000
EMBEDDING_SIMILARITY_THRESHOLD = 0.85
SGD_MIN_SAMPLES = 5
FILENAME_MATCH_CONFIDENCE = 0.80
DETERMINISTIC_RESOLVE_THRESHOLD = 0.85
SGD_PREDICT_THRESHOLD = 0.6
LLM_CONFIDENCE_CAP = 0.95
FRONT_MATTER_EMBED_LIMIT = 2000
FRONT_MATTER_LLM_LIMIT = 4000


# ── LangChain Prompt Templates ────────────────────────────────────────

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a document classification expert. Your task is to determine the type "
        "and category of a document based on its filename and front-matter text.\n\n"
        "You MUST respond with a JSON object containing exactly these fields:\n"
        '- "document_type": A specific document type identifier (e.g., "10-K", "annual_report", '
        '"contract", "pillar3_disclosure", "loan_agreement", "esg_report", "policy_document", '
        '"research_paper", "invoice", "regulatory_filing")\n'
        '- "classification_label": A broader category label (e.g., "sec_filing", "financial_report", '
        '"legal_document", "basel_regulatory", "sustainability", "governance", "academic", '
        '"commercial", "regulatory")\n'
        '- "confidence": A float between 0 and 1 indicating your confidence\n\n'
        "Respond with ONLY the JSON object, no other text."
    )),
    ("human", (
        "Classify this document:\n\n"
        "Filename: {filename}\n"
        "Page count: {page_count}\n\n"
        "Front-matter text (first pages):\n---\n{front_matter_text}\n---\n\n"
        "Return your classification as JSON."
    )),
])


# ── LangGraph State Definition ────────────────────────────────────────

class ClassificationState(TypedDict):
    """State passed through the LangGraph classification flow."""
    doc_id: str
    filename: str
    front_matter_text: str
    page_count: int
    structural_signals: Dict[str, Any]
    query_id: str
    # Computed during flow
    embedding: Any          # Optional[np.ndarray]
    det_result: Any         # Optional[Tuple[str, str, float]]
    result: Any             # Optional[ClassificationResult]
    method: str             # which tier resolved it


# ── pgvector DB helpers ───────────────────────────────────────────────

def _pgvector_available() -> bool:
    """Check if pgvector-backed storage is available (DATABASE_URL configured)."""
    return bool(settings.database_url)


def _pgvector_store_embedding(
    embedding_id: str,
    embedding: np.ndarray,
    document_type: str,
    classification_label: str,
    source_doc_id: Optional[str] = None,
) -> bool:
    """Store embedding in pgvector. Returns True on success."""
    try:
        from storage.db_pool import get_connection
        from storage.repo import insert_classification_embedding
        with get_connection() as conn:
            insert_classification_embedding(
                conn, embedding_id, document_type, classification_label,
                embedding.tolist(), source_doc_id,
            )
            conn.commit()
        return True
    except Exception as exc:
        logger.debug("pgvector store failed (falling back to in-memory): %s", exc)
        return False


def _pgvector_search_embedding(
    query_embedding: np.ndarray,
    threshold: float = EMBEDDING_SIMILARITY_THRESHOLD,
) -> Optional[Tuple[str, str, float]]:
    """Search pgvector for nearest classification embedding.
    Returns (document_type, classification_label, similarity) or None.
    """
    try:
        from storage.db_pool import get_connection
        from storage.repo import search_classification_embeddings
        with get_connection() as conn:
            results = search_classification_embeddings(
                conn, query_embedding.tolist(), top_k=1, threshold=threshold,
            )
        if results:
            r = results[0]
            return (r["document_type"], r["classification_label"], r["similarity"])
        return None
    except Exception as exc:
        logger.debug("pgvector search failed (falling back to in-memory): %s", exc)
        return None


def _pgvector_fetch_all() -> List[dict]:
    """Fetch all classification embeddings from pgvector for SGD retraining."""
    try:
        from storage.db_pool import get_connection
        from storage.repo import fetch_all_classification_embeddings
        with get_connection() as conn:
            return fetch_all_classification_embeddings(conn)
    except Exception as exc:
        logger.debug("pgvector fetch_all failed: %s", exc)
        return []


def _pgvector_count() -> int:
    """Count classification embeddings in pgvector."""
    try:
        from storage.db_pool import get_connection
        from storage.repo import count_classification_embeddings
        with get_connection() as conn:
            return count_classification_embeddings(conn)
    except Exception:
        return 0


# ── Neo4j Knowledge Graph helpers ─────────────────────────────────────

def _neo4j_available() -> bool:
    """Check if Neo4j knowledge graph is enabled and available."""
    return settings.neo4j.enabled


def _neo4j_store_classification(
    doc_id: str,
    filename: str,
    document_type: str,
    classification_label: str,
    confidence: float,
    method: str,
    page_count: int = 0,
) -> bool:
    """Store classification in Neo4j knowledge graph. Returns True on success."""
    try:
        from storage.knowledge_graph import store_classification
        store_classification(
            doc_id=doc_id, filename=filename,
            document_type=document_type, classification_label=classification_label,
            confidence=confidence, method=method, page_count=page_count,
        )
        return True
    except Exception as exc:
        logger.debug("Neo4j store failed: %s", exc)
        return False


def _neo4j_find_similar(
    document_type: str,
    classification_label: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Find similar documents via Neo4j graph traversal."""
    try:
        from storage.knowledge_graph import find_similar_documents
        return find_similar_documents(document_type, classification_label, limit)
    except Exception as exc:
        logger.debug("Neo4j find_similar failed: %s", exc)
        return []


# ── Embedding Store (pgvector + in-memory fallback) ───────────────────

class EmbeddingStore:
    """Stores and retrieves document classification embeddings.

    Primary backend: pgvector HNSW index in PostgreSQL.
    Fallback: in-memory numpy cosine similarity (for unit tests / no DB).
    """

    def __init__(self, embedding_dim: int = 768, use_db: bool = False) -> None:
        self._embedding_dim = embedding_dim
        self._use_db = use_db
        self._mem_embeddings: List[np.ndarray] = []
        self._mem_doc_types: List[str] = []
        self._mem_class_labels: List[str] = []
        self._mem_labels: List[str] = []  # compound "type::label"

    @property
    def embedding_count(self) -> int:
        return len(self._mem_embeddings)

    @property
    def mem_labels(self) -> List[str]:
        return self._mem_labels

    def store(
        self,
        embedding: np.ndarray,
        document_type: str,
        classification_label: str,
        source_doc_id: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Store embedding; returns the normalized array or None on dim mismatch."""
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        if emb.shape[0] != self._embedding_dim:
            logger.warning(
                "Embedding dim mismatch: expected %d, got %d",
                self._embedding_dim, emb.shape[0],
            )
            return None

        if self._use_db:
            _pgvector_store_embedding(
                new_id(), emb, document_type, classification_label, source_doc_id,
            )

        self._mem_embeddings.append(emb)
        self._mem_labels.append(f"{document_type}::{classification_label}")
        self._mem_doc_types.append(document_type)
        self._mem_class_labels.append(classification_label)
        return emb

    def lookup(
        self,
        query_embedding: np.ndarray,
        threshold: float = EMBEDDING_SIMILARITY_THRESHOLD,
    ) -> Optional[Tuple[str, str, float]]:
        """Find the most similar stored document. Returns (doc_type, label, sim) or None."""
        query_emb = np.asarray(query_embedding, dtype=np.float32).ravel()

        if self._use_db:
            result = _pgvector_search_embedding(query_emb, threshold)
            if result is not None:
                return result

        return self._lookup_in_memory(query_emb, threshold)

    def _lookup_in_memory(
        self,
        query_emb: np.ndarray,
        threshold: float,
    ) -> Optional[Tuple[str, str, float]]:
        if not self._mem_embeddings:
            return None

        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        best_sim = -1.0
        best_idx = -1
        for i, stored_emb in enumerate(self._mem_embeddings):
            stored_norm = stored_emb / (np.linalg.norm(stored_emb) + 1e-10)
            sim = float(np.dot(query_norm, stored_norm))
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim < threshold or best_idx < 0:
            return None

        return (self._mem_doc_types[best_idx], self._mem_class_labels[best_idx], best_sim)

    def load_from_rows(self, rows: List[dict]) -> None:
        """Bulk-load embeddings from pgvector rows into in-memory store."""
        for row in rows:
            emb = np.array(row["embedding"], dtype=np.float32)
            compound = f"{row['document_type']}::{row['classification_label']}"
            self._mem_embeddings.append(emb)
            self._mem_labels.append(compound)
            self._mem_doc_types.append(row["document_type"])
            self._mem_class_labels.append(row["classification_label"])

    def export_embeddings(self) -> List[dict]:
        result = []
        for i, emb in enumerate(self._mem_embeddings):
            result.append({
                "embedding": emb.tolist(),
                "document_type": self._mem_doc_types[i],
                "classification_label": self._mem_class_labels[i],
            })
        return result

    def import_embeddings(self, embeddings: List[dict]) -> int:
        count = 0
        for item in embeddings:
            emb = np.array(item["embedding"], dtype=np.float32)
            compound = f"{item['document_type']}::{item['classification_label']}"
            self._mem_embeddings.append(emb)
            self._mem_doc_types.append(item["document_type"])
            self._mem_class_labels.append(item["classification_label"])
            self._mem_labels.append(compound)
            count += 1
        return count


# ── SGD Classifier Wrapper ────────────────────────────────────────────

class SGDClassifierWrapper:
    """Online-learning SGD classifier trained incrementally on embeddings."""

    def __init__(self) -> None:
        self._sgd: SGDClassifier = SGDClassifier(
            loss="modified_huber",
            penalty="l2",
            alpha=1e-4,
            max_iter=1,
            tol=None,
            warm_start=True,
            random_state=42,
        )
        self._sgd_classes: List[str] = []
        self._sgd_fitted: bool = False
        self._sgd_sample_count: int = 0

    @property
    def fitted(self) -> bool:
        return self._sgd_fitted

    @property
    def sample_count(self) -> int:
        return self._sgd_sample_count

    @property
    def classes(self) -> List[str]:
        return list(self._sgd_classes)

    def predict(self, query_embedding: np.ndarray) -> Optional[Tuple[str, str, float]]:
        """Predict classification from embedding. Returns (doc_type, label, confidence) or None."""
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

    def train_incremental(self, embedding: np.ndarray, compound_label: str,
                          embedding_store: EmbeddingStore) -> None:
        """Train on a single new sample, bootstrapping from store if needed."""
        if compound_label not in self._sgd_classes:
            self._sgd_classes.append(compound_label)

        if len(self._sgd_classes) < 2:
            self._sgd_sample_count += 1
            return

        if not self._sgd_fitted and embedding_store.embedding_count >= 2:
            self._retrain_from_store(embedding_store)
            return

        X = embedding.reshape(1, -1)
        try:
            self._sgd.partial_fit(X, [compound_label], classes=self._sgd_classes)
            self._sgd_fitted = True
            self._sgd_sample_count += 1
        except Exception as exc:
            logger.warning("SGD incremental training failed: %s", exc)

    def _retrain_from_store(self, embedding_store: EmbeddingStore) -> None:
        """Retrain SGD from all stored embeddings."""
        if not embedding_store._mem_embeddings or len(self._sgd_classes) < 2:
            return
        X = np.stack(embedding_store._mem_embeddings)
        y = embedding_store.mem_labels
        try:
            self._sgd.partial_fit(X, y, classes=self._sgd_classes)
            self._sgd_fitted = True
            self._sgd_sample_count = len(embedding_store._mem_embeddings)
        except Exception as exc:
            logger.warning("SGD retrain failed: %s", exc)


# ── Pattern Store ─────────────────────────────────────────────────────

class PatternStore:
    """Keyword/filename pattern index for zero-cost exact matches."""

    def __init__(self) -> None:
        self._patterns: Dict[str, ClassificationMemoryEntry] = {}
        self._filename_index: Dict[str, List[str]] = defaultdict(list)
        self._keyword_index: Dict[str, List[str]] = defaultdict(list)

    def store(
        self,
        document_type: str,
        classification_label: str,
        filename: Optional[str] = None,
        title_keywords: Optional[List[str]] = None,
        structural_signals: Optional[Dict[str, Any]] = None,
    ) -> str:
        existing = self._find_exact_match(document_type, classification_label, filename)
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
            self._filename_index[self._normalize_filename(filename)].append(pattern_id)
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

    def get_all(self) -> List[ClassificationMemoryEntry]:
        return list(self._patterns.values())

    def export_patterns(self) -> List[dict]:
        result = []
        for entry in self._patterns.values():
            result.append({
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
        return result

    def import_patterns(self, patterns: List[dict]) -> int:
        count = 0
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
        return count

    def _find_exact_match(self, document_type, classification_label, filename):
        for pid, entry in self._patterns.items():
            if entry.document_type == document_type and entry.classification_label == classification_label and entry.filename_pattern == filename:
                return pid
        return None

    @staticmethod
    def _normalize_filename(filename: str) -> str:
        name = re.sub(r"\.[^.]+$", "", filename)
        return re.sub(r"[\s_-]+", " ", name.lower().strip())

    def _prune_lowest_accuracy(self) -> None:
        sorted_patterns = sorted(self._patterns.items(), key=lambda kv: (kv[1].accuracy, kv[1].total_count))
        to_remove = max(1, len(sorted_patterns) // 10)
        for pid, _ in sorted_patterns[:to_remove]:
            del self._patterns[pid]


# ── Classification Memory (facade composing the three stores) ─────────

class ClassificationMemory:
    """Self-learning classification memory composing EmbeddingStore, SGDClassifierWrapper, PatternStore.

    Delegates to focused sub-components for each responsibility:
    - EmbeddingStore: pgvector/in-memory embedding storage + cosine lookup
    - SGDClassifierWrapper: sklearn online learning
    - PatternStore: keyword/filename pattern index
    """

    def __init__(self, embedding_dim: int = 768, use_db: Optional[bool] = None) -> None:
        resolved_use_db = use_db if use_db is not None else _pgvector_available()
        self._embeddings = EmbeddingStore(embedding_dim, resolved_use_db)
        self._sgd = SGDClassifierWrapper()
        self._patterns = PatternStore()
        self._use_db = resolved_use_db

    # ── Backward-compatible properties for test access ────────────────

    @property
    def _mem_embeddings(self) -> List[np.ndarray]:
        return self._embeddings._mem_embeddings

    @property
    def _sgd_sample_count(self) -> int:
        return self._sgd.sample_count

    @property
    def _sgd_fitted(self) -> bool:
        return self._sgd.fitted

    @property
    def _sgd_classes(self) -> List[str]:
        return self._sgd.classes

    # ── Embedding operations ──────────────────────────────────────────

    def store_embedding(
        self,
        embedding: np.ndarray,
        document_type: str,
        classification_label: str,
        source_doc_id: Optional[str] = None,
    ) -> None:
        emb = self._embeddings.store(embedding, document_type, classification_label, source_doc_id)
        if emb is not None:
            compound = f"{document_type}::{classification_label}"
            self._sgd.train_incremental(emb, compound, self._embeddings)

    def lookup_by_embedding(
        self,
        query_embedding: np.ndarray,
        threshold: float = EMBEDDING_SIMILARITY_THRESHOLD,
    ) -> Optional[Tuple[str, str, float]]:
        return self._embeddings.lookup(query_embedding, threshold)

    def predict_sgd(self, query_embedding: np.ndarray) -> Optional[Tuple[str, str, float]]:
        return self._sgd.predict(query_embedding)

    def bootstrap_sgd_from_pgvector(self) -> int:
        if not self._use_db:
            return 0
        rows = _pgvector_fetch_all()
        if not rows:
            return 0
        self._embeddings.load_from_rows(rows)
        for row in rows:
            compound = f"{row['document_type']}::{row['classification_label']}"
            if compound not in self._sgd._sgd_classes:
                self._sgd._sgd_classes.append(compound)
        if len(self._sgd._sgd_classes) >= 2:
            self._sgd._retrain_from_store(self._embeddings)
        else:
            self._sgd._sgd_sample_count = self._embeddings.embedding_count
        logger.info("Bootstrapped SGD from pgvector: %d embeddings", len(rows))
        return len(rows)

    # ── Pattern operations ────────────────────────────────────────────

    def store_pattern(
        self,
        document_type: str,
        classification_label: str,
        filename: Optional[str] = None,
        title_keywords: Optional[List[str]] = None,
        structural_signals: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._patterns.store(document_type, classification_label, filename, title_keywords, structural_signals)

    def lookup(
        self,
        filename: Optional[str] = None,
        title_keywords: Optional[List[str]] = None,
        min_accuracy: float = MEMORY_TRUST_THRESHOLD,
        min_observations: int = MEMORY_MIN_OBSERVATIONS,
    ) -> Optional[ClassificationMemoryEntry]:
        return self._patterns.lookup(filename, title_keywords, min_accuracy, min_observations)

    def record_feedback(self, pattern_id: str, correct: bool) -> None:
        self._patterns.record_feedback(pattern_id, correct)

    def get_all_patterns(self) -> List[ClassificationMemoryEntry]:
        return self._patterns.get_all()

    # ── Stats / export / import ───────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        patterns = self._patterns.get_all()
        trusted = [p for p in patterns if p.accuracy >= MEMORY_TRUST_THRESHOLD and p.total_count >= MEMORY_MIN_OBSERVATIONS]
        pgvector_count = _pgvector_count() if self._use_db else 0
        return {
            "total_patterns": len(patterns),
            "trusted_patterns": len(trusted),
            "total_observations": sum(p.total_count for p in patterns),
            "avg_accuracy": (sum(p.accuracy for p in patterns) / len(patterns) if patterns else 0.0),
            "embedding_count": self._embeddings.embedding_count,
            "pgvector_count": pgvector_count,
            "vector_backend": "pgvector" if self._use_db else "in-memory",
            "neo4j_enabled": _neo4j_available(),
            "sgd_fitted": self._sgd.fitted,
            "sgd_sample_count": self._sgd.sample_count,
            "sgd_classes": self._sgd.classes,
        }

    def export_json(self) -> str:
        data = {
            "patterns": self._patterns.export_patterns(),
            "embeddings": self._embeddings.export_embeddings(),
            "sgd_classes": self._sgd._sgd_classes,
            "sgd_sample_count": self._sgd.sample_count,
        }
        return json.dumps(data, indent=2)

    def import_json(self, raw: str) -> int:
        data = json.loads(raw)
        count = 0
        if isinstance(data, list):
            patterns_data, embeddings_data = data, []
        else:
            patterns_data = data.get("patterns", [])
            embeddings_data = data.get("embeddings", [])
            self._sgd._sgd_classes = data.get("sgd_classes", [])
            self._sgd._sgd_sample_count = data.get("sgd_sample_count", 0)

        count += self._patterns.import_patterns(patterns_data)
        count += self._embeddings.import_embeddings(embeddings_data)

        if self._embeddings._mem_embeddings and self._sgd._sgd_classes:
            self._sgd._retrain_from_store(self._embeddings)
        return count


# ── Singleton ─────────────────────────────────────────────────────────

_classification_memory: Optional[ClassificationMemory] = None


def get_classification_memory() -> ClassificationMemory:
    global _classification_memory
    if _classification_memory is None:
        _classification_memory = ClassificationMemory()
    return _classification_memory


def reset_classification_memory() -> None:
    global _classification_memory
    _classification_memory = None


# ── Embedder helper ───────────────────────────────────────────────────

def _get_embedder():
    try:
        from embedding.model_registry import get_embedding_model
        return get_embedding_model()
    except Exception as exc:
        logger.debug("Could not load embedding model: %s", exc)
        return None


# ── LangGraph Node Functions ──────────────────────────────────────────

def _node_compute_embedding(state: ClassificationState, embedder) -> dict:
    """Compute the front-matter embedding for downstream tiers."""
    text = state.get("front_matter_text", "")
    if not text or embedder is None:
        return {"embedding": None}
    try:
        sample = text[:FRONT_MATTER_EMBED_LIMIT]
        vec = embedder.embed_text(sample)
        return {"embedding": np.array(vec, dtype=np.float32)}
    except Exception as exc:
        logger.warning("Embedding failed: %s", exc)
        return {"embedding": None}


def _node_deterministic(state: ClassificationState) -> dict:
    """Tier 1: deterministic regex classification."""
    filename = state["filename"]
    text = state.get("front_matter_text", "")
    best: Optional[Tuple[str, str, float]] = None

    for pattern, doc_type, label in _FILENAME_PATTERNS:
        if pattern.search(filename):
            if best is None or FILENAME_MATCH_CONFIDENCE > best[2]:
                best = (doc_type, label, FILENAME_MATCH_CONFIDENCE)

    if text:
        for pattern, doc_type, label, weight in _CONTENT_SIGNALS:
            if pattern.search(text):
                if best is None or weight > best[2]:
                    best = (doc_type, label, weight)

    if best and best[2] >= DETERMINISTIC_RESOLVE_THRESHOLD:
        return {
            "det_result": best,
            "result": ClassificationResult(
                doc_id=state["doc_id"],
                document_type=best[0],
                classification_label=best[1],
                confidence=best[2],
                classification_method="deterministic",
                evidence_signals=state["structural_signals"],
            ),
            "method": "deterministic",
        }
    return {"det_result": best, "result": None, "method": ""}


def _node_pgvector_similarity(state: ClassificationState, memory: ClassificationMemory) -> dict:
    """Tier 2: pgvector cosine similarity lookup (falls back to in-memory numpy)."""
    embedding = state.get("embedding")
    if embedding is None:
        return {"result": None, "method": ""}

    sim_result = memory.lookup_by_embedding(embedding)
    if sim_result:
        doc_type, label, similarity = sim_result
        return {
            "result": ClassificationResult(
                doc_id=state["doc_id"],
                document_type=doc_type,
                classification_label=label,
                confidence=similarity,
                classification_method="embedding_similarity",
                evidence_signals=state["structural_signals"],
            ),
            "method": "embedding_similarity",
        }
    return {"result": None, "method": ""}


def _node_sgd_classifier(state: ClassificationState, memory: ClassificationMemory) -> dict:
    """Tier 3: sklearn SGDClassifier prediction."""
    embedding = state.get("embedding")
    if embedding is None:
        return {"result": None, "method": ""}

    sgd_result = memory.predict_sgd(embedding)
    if sgd_result and sgd_result[2] >= SGD_PREDICT_THRESHOLD:
        doc_type, label, confidence = sgd_result
        return {
            "result": ClassificationResult(
                doc_id=state["doc_id"],
                document_type=doc_type,
                classification_label=label,
                confidence=confidence,
                classification_method="sgd_classifier",
                evidence_signals=state["structural_signals"],
            ),
            "method": "sgd_classifier",
        }
    return {"result": None, "method": ""}


def _node_llm_classify(state: ClassificationState, gateway: Optional[ModelGateway]) -> dict:
    """Tier 4: LLM classification via LangChain ChatPromptTemplate."""
    text = state.get("front_matter_text", "")
    if not gateway or not text:
        return {"result": None, "method": ""}

    # Format prompt using LangChain template
    messages = CLASSIFICATION_PROMPT.format_messages(
        filename=state["filename"],
        page_count=state["page_count"],
        front_matter_text=text[:FRONT_MATTER_LLM_LIMIT],
    )

    # Convert LangChain messages to gateway format
    gateway_messages = [
        {"role": "system" if m.type == "system" else "user", "content": m.content}
        for m in messages
    ]

    try:
        result = gateway.call_model(
            model_id="gpt-4o-mini",
            messages=gateway_messages,
            temperature=0.0,
            query_id=state.get("query_id", ""),
            agent_id="classifier",
            step_id=new_id(),
        )
        parsed = _parse_llm_response(result.get("content", ""))
        if parsed:
            return {
                "result": ClassificationResult(
                    doc_id=state["doc_id"],
                    document_type=parsed[0],
                    classification_label=parsed[1],
                    confidence=parsed[2],
                    classification_method="llm",
                    evidence_signals=state["structural_signals"],
                ),
                "method": "llm",
            }
    except Exception as exc:
        logger.warning("LLM classification failed: %s", exc)

    return {"result": None, "method": ""}


def _node_fallback(state: ClassificationState) -> dict:
    """Final fallback when all tiers fail."""
    det = state.get("det_result")
    if det:
        doc_type, label, confidence = det
        method = "deterministic"
    else:
        doc_type, label, confidence = "unknown", "unclassified", 0.0
        method = "default"

    return {
        "result": ClassificationResult(
            doc_id=state["doc_id"],
            document_type=doc_type,
            classification_label=label,
            confidence=confidence,
            classification_method=method,
            evidence_signals=state["structural_signals"],
        ),
        "method": method,
    }


def _parse_llm_response(response: str) -> Optional[Tuple[str, str, float]]:
    """Parse LLM JSON response."""
    try:
        json_str = response.strip()
        if "```" in json_str:
            match = re.search(r"```(?:json)?\s*(.*?)```", json_str, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                return None
        data = json.loads(json_str)
        doc_type = data.get("document_type", "unknown")
        label = data.get("classification_label", "unclassified")
        confidence = float(data.get("confidence", 0.5))
        return (doc_type, label, min(confidence, LLM_CONFIDENCE_CAP))
    except (json.JSONDecodeError, ValueError, AttributeError) as exc:
        logger.warning("Failed to parse LLM classification response: %s", exc)
        return None


# ── LangGraph Builder ─────────────────────────────────────────────────

def _build_classification_graph(
    memory: ClassificationMemory,
    gateway: Optional[ModelGateway],
    embedder: Any,
) -> StateGraph:
    """Build the LangGraph StateGraph for the classification flow.

    Graph topology:
        embed → deterministic → [resolved?] → END
                                [not resolved?] → pgvector_similarity → [resolved?] → END
                                                                         [not?] → sgd → [resolved?] → END
                                                                                         [not?] → llm → [resolved?] → END
                                                                                                         [not?] → fallback → END
    Uses conditional edges to short-circuit as soon as a tier resolves.
    """
    graph = StateGraph(ClassificationState)

    # Add nodes — each node is a lambda that closes over its dependencies
    graph.add_node("embed", lambda s: _node_compute_embedding(s, embedder))
    graph.add_node("deterministic", _node_deterministic)
    graph.add_node("pgvector_similarity", lambda s: _node_pgvector_similarity(s, memory))
    graph.add_node("sgd", lambda s: _node_sgd_classifier(s, memory))
    graph.add_node("llm", lambda s: _node_llm_classify(s, gateway))
    graph.add_node("fallback", _node_fallback)

    # Conditional routing helpers
    def _is_resolved(state: ClassificationState) -> str:
        return "resolved" if state.get("result") is not None else "unresolved"

    # Wire the graph
    graph.set_entry_point("embed")
    graph.add_edge("embed", "deterministic")

    graph.add_conditional_edges("deterministic", _is_resolved, {
        "resolved": END,
        "unresolved": "pgvector_similarity",
    })
    graph.add_conditional_edges("pgvector_similarity", _is_resolved, {
        "resolved": END,
        "unresolved": "sgd",
    })
    graph.add_conditional_edges("sgd", _is_resolved, {
        "resolved": END,
        "unresolved": "llm",
    })
    graph.add_conditional_edges("llm", _is_resolved, {
        "resolved": END,
        "unresolved": "fallback",
    })
    graph.add_edge("fallback", END)

    return graph


# ── Classifier Agent ──────────────────────────────────────────────────

class ClassifierAgent(BaseAgent):
    """Self-learning document classifier (§4.8).

    Orchestrated by a LangGraph StateGraph with conditional edges:
    embed → deterministic → pgvector_similarity → sgd → llm → fallback

    Each tier short-circuits via conditional edges when resolved.
    Results stored in pgvector (PostgreSQL) + sklearn SGDClassifier.
    Classifications also recorded in Neo4j knowledge graph (when enabled).
    Prompts structured via LangChain ChatPromptTemplate.
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
        self._embedder = embedder
        self._graph = None  # Built lazily

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = _get_embedder()
        return self._embedder

    def _get_graph(self):
        """Build or return the compiled LangGraph."""
        if self._graph is None:
            sg = _build_classification_graph(
                memory=self._memory,
                gateway=self.gateway,
                embedder=self._get_embedder(),
            )
            self._graph = sg.compile()
        return self._graph

    def handle_message(self, message: AgentMessage) -> ClassificationResult:
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
        """Classify a document by invoking the LangGraph classification flow."""
        start = time.monotonic()
        signals = dict(structural_signals or {})
        signals["filename"] = filename
        signals["page_count"] = page_count

        # Build initial state
        initial_state: ClassificationState = {
            "doc_id": doc_id,
            "filename": filename,
            "front_matter_text": front_matter_text,
            "page_count": page_count,
            "structural_signals": signals,
            "query_id": query_id,
            "embedding": None,
            "det_result": None,
            "result": None,
            "method": "",
        }

        # Run the LangGraph
        graph = self._get_graph()
        final_state = graph.invoke(initial_state)

        result = final_state["result"]
        embedding = final_state.get("embedding")

        # Store in memory for future learning
        self._store_in_memory(result, filename, front_matter_text, embedding, page_count)
        self._log_classification(result, start, query_id)

        return result

    def handle_feedback(self, pattern_id: str, correct: bool) -> None:
        self._memory.record_feedback(pattern_id, correct)
        logger.info("Classification feedback: pattern=%s correct=%s", pattern_id, correct)

    def get_memory_stats(self) -> Dict[str, Any]:
        return self._memory.get_stats()

    # Expose for tests
    def _parse_llm_response(self, response: str) -> Optional[Tuple[str, str, float]]:
        return _parse_llm_response(response)

    def _store_in_memory(
        self,
        result: ClassificationResult,
        filename: str,
        front_matter_text: str,
        embedding: Optional[np.ndarray] = None,
        page_count: int = 0,
    ) -> None:
        title_keywords = self._extract_title_keywords(front_matter_text)
        self._memory.store_pattern(
            document_type=result.document_type,
            classification_label=result.classification_label,
            filename=filename,
            title_keywords=title_keywords,
            structural_signals=result.evidence_signals,
        )
        if embedding is not None:
            self._memory.store_embedding(
                embedding=embedding,
                document_type=result.document_type,
                classification_label=result.classification_label,
                source_doc_id=result.doc_id,
            )
        # Store in Neo4j knowledge graph (when enabled)
        if _neo4j_available():
            _neo4j_store_classification(
                doc_id=result.doc_id,
                filename=filename,
                document_type=result.document_type,
                classification_label=result.classification_label,
                confidence=result.confidence,
                method=result.classification_method,
                page_count=page_count,
            )

    def _extract_title_keywords(self, text: str) -> List[str]:
        if not text:
            return []
        sample = text[:500]
        phrases = re.findall(r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+", sample)
        caps = re.findall(r"\b[A-Z]{3,}(?:\s+[A-Z]{3,})+\b", sample)
        return list(set(phrases + caps))[:10]

    def _log_classification(
        self, result: ClassificationResult, start: float, query_id: str
    ) -> None:
        latency_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Classifier: doc=%s type=%s label=%s conf=%.2f method=%s (%.0fms)",
            result.doc_id, result.document_type, result.classification_label,
            result.confidence, result.classification_method, latency_ms,
        )
        from agents.agent_eval import EvalCase, get_evaluator
        get_evaluator().record(EvalCase(
            query_id=query_id or new_id(),
            agent_name=self.agent_name,
            latency_ms=latency_ms,
            answer_confidence=result.confidence,
        ))
