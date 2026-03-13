"""Classifier Agent — self-learning document classification (MASTER_PROMPT §4.8).

Classifies ANY document into document_type and classification_label to determine
the processing pipeline. Uses a 4-tier classification strategy orchestrated by
a LangGraph StateGraph with conditional routing:

1. **Deterministic rules**: Filename patterns, structural signals (fast, free)
2. **LlamaIndex similarity**: VectorStoreIndex over prior classified documents
   with cosine similarity retrieval for semantic memory matching
3. **Incremental classifier**: sklearn SGDClassifier with partial_fit() —
   trained online from every classification, improves over time
4. **LLM classification**: LangChain ChatPromptTemplate + Tier-2 model for
   ambiguous documents (fallback)

Frameworks:
- **LangGraph**: StateGraph with conditional edges for classification flow
- **LangChain**: ChatPromptTemplate for structured LLM prompts
- **LlamaIndex**: VectorStoreIndex for embedding-based memory retrieval
- **sklearn**: SGDClassifier for incremental online learning

Self-learning loop:
- After each classification, the front-matter embedding + label are stored
  in the LlamaIndex VectorStoreIndex and the SGDClassifier is trained
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
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery
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


# ── Classification Memory (LlamaIndex + sklearn) ─────────────────────

class ClassificationMemory:
    """Self-learning classification memory using LlamaIndex + sklearn.

    - **LlamaIndex VectorStoreIndex**: stores prior classification embeddings
      as TextNodes. Retrieval via cosine similarity replaces the hand-rolled
      numpy cosine search.
    - **SGDClassifier**: online learning from embeddings for trainable
      classification beyond nearest-neighbor.
    - **Pattern store**: keyword/filename index for zero-cost exact matches.
    """

    def __init__(self, embedding_dim: int = 768) -> None:
        self._embedding_dim = embedding_dim

        # ── LlamaIndex SimpleVectorStore ──────────────────────────────
        self._index_nodes: List[TextNode] = []
        self._vector_store: Optional[SimpleVectorStore] = None
        self._embedding_doc_types: List[str] = []
        self._embedding_class_labels: List[str] = []
        # Keep raw embeddings for SGD + export
        self._embeddings: List[np.ndarray] = []
        self._embedding_labels: List[str] = []

        # ── sklearn SGD classifier ────────────────────────────────────
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

        # ── Pattern store (keyword/filename) ──────────────────────────
        self._patterns: Dict[str, ClassificationMemoryEntry] = {}
        self._filename_index: Dict[str, List[str]] = defaultdict(list)
        self._keyword_index: Dict[str, List[str]] = defaultdict(list)

    # ── LlamaIndex embedding operations ──────────────────────────────

    def store_embedding(
        self,
        embedding: np.ndarray,
        document_type: str,
        classification_label: str,
    ) -> None:
        """Store a document embedding in the LlamaIndex index + train SGD."""
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        if emb.shape[0] != self._embedding_dim:
            logger.warning(
                "Embedding dim mismatch: expected %d, got %d",
                self._embedding_dim, emb.shape[0],
            )
            return

        compound_label = f"{document_type}::{classification_label}"

        # Store in LlamaIndex as a TextNode with pre-computed embedding
        node = TextNode(
            text=compound_label,
            id_=new_id(),
            embedding=emb.tolist(),
            metadata={
                "document_type": document_type,
                "classification_label": classification_label,
            },
        )
        self._index_nodes.append(node)
        self._vector_store = None  # Invalidate — rebuilt on next query

        # Keep raw data for SGD + export
        self._embeddings.append(emb)
        self._embedding_labels.append(compound_label)
        self._embedding_doc_types.append(document_type)
        self._embedding_class_labels.append(classification_label)

        self._train_incremental(emb, compound_label)

    def lookup_by_embedding(
        self,
        query_embedding: np.ndarray,
        threshold: float = EMBEDDING_SIMILARITY_THRESHOLD,
    ) -> Optional[Tuple[str, str, float]]:
        """Find the most similar stored document via LlamaIndex SimpleVectorStore.

        Uses SimpleVectorStore.add() to preserve pre-computed embeddings on
        TextNodes, then queries with cosine similarity.
        Returns (document_type, classification_label, similarity) or None.
        """
        if not self._index_nodes:
            return None

        query_emb = np.asarray(query_embedding, dtype=np.float32).ravel()

        # Build/rebuild the vector store lazily. SimpleVectorStore.add()
        # preserves pre-computed embeddings on TextNodes directly, unlike
        # VectorStoreIndex which re-embeds via the embed_model.
        if self._vector_store is None:
            self._vector_store = SimpleVectorStore()
            self._vector_store.add(self._index_nodes)

        try:
            query_result = self._vector_store.query(
                VectorStoreQuery(
                    query_embedding=query_emb.tolist(),
                    similarity_top_k=1,
                )
            )

            if not query_result.ids or not query_result.similarities:
                return None

            best_sim = float(query_result.similarities[0])
            if best_sim < threshold:
                return None

            # SimpleVectorStore returns ids, not full nodes — look up metadata
            best_id = query_result.ids[0]
            metadata = self._vector_store._data.metadata_dict.get(best_id, {})
            doc_type = metadata.get("document_type", "unknown")
            label = metadata.get("classification_label", "unclassified")
            return (doc_type, label, best_sim)

        except Exception as exc:
            logger.warning("LlamaIndex retrieval failed: %s", exc)
            return None

    def predict_sgd(
        self, query_embedding: np.ndarray
    ) -> Optional[Tuple[str, str, float]]:
        """Use the SGDClassifier to predict classification."""
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
        """Incrementally train SGDClassifier with a single new sample."""
        X = embedding.reshape(1, -1)

        if compound_label not in self._sgd_classes:
            self._sgd_classes.append(compound_label)

        if len(self._sgd_classes) < 2:
            self._sgd_sample_count += 1
            return

        if not self._sgd_fitted and len(self._embeddings) >= 2:
            self._retrain_sgd_from_store()
            return

        try:
            self._sgd.partial_fit(X, [compound_label], classes=self._sgd_classes)
            self._sgd_fitted = True
            self._sgd_sample_count += 1
        except Exception as exc:
            logger.warning("SGD incremental training failed: %s", exc)

    def _retrain_sgd_from_store(self) -> None:
        """Retrain SGDClassifier from all stored embeddings."""
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

    # ── Pattern store operations ──────────────────────────────────────

    def store_pattern(
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

    def get_all_patterns(self) -> List[ClassificationMemoryEntry]:
        return list(self._patterns.values())

    def get_stats(self) -> Dict[str, Any]:
        patterns = list(self._patterns.values())
        trusted = [p for p in patterns if p.accuracy >= MEMORY_TRUST_THRESHOLD and p.total_count >= MEMORY_MIN_OBSERVATIONS]
        return {
            "total_patterns": len(patterns),
            "trusted_patterns": len(trusted),
            "total_observations": sum(p.total_count for p in patterns),
            "avg_accuracy": (sum(p.accuracy for p in patterns) / len(patterns) if patterns else 0.0),
            "embedding_count": len(self._embeddings),
            "llama_index_nodes": len(self._index_nodes),
            "sgd_fitted": self._sgd_fitted,
            "sgd_sample_count": self._sgd_sample_count,
            "sgd_classes": list(self._sgd_classes),
        }

    def export_json(self) -> str:
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
        data = json.loads(raw)
        count = 0
        if isinstance(data, list):
            patterns, embeddings = data, []
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
            compound = f"{item['document_type']}::{item['classification_label']}"
            node = TextNode(
                text=compound,
                id_=new_id(),
                embedding=emb.tolist(),
                metadata={
                    "document_type": item["document_type"],
                    "classification_label": item["classification_label"],
                },
            )
            self._index_nodes.append(node)
            self._embeddings.append(emb)
            self._embedding_doc_types.append(item["document_type"])
            self._embedding_class_labels.append(item["classification_label"])
            self._embedding_labels.append(compound)
            count += 1

        self._vector_store = None  # Force rebuild
        if self._embeddings and self._sgd_classes:
            self._retrain_sgd_from_store()
        return count

    def _find_exact_match(self, document_type, classification_label, filename):
        for pid, entry in self._patterns.items():
            if entry.document_type == document_type and entry.classification_label == classification_label and entry.filename_pattern == filename:
                return pid
        return None

    def _normalize_filename(self, filename: str) -> str:
        name = re.sub(r"\.[^.]+$", "", filename)
        return re.sub(r"[\s_-]+", " ", name.lower().strip())

    def _prune_lowest_accuracy(self) -> None:
        sorted_patterns = sorted(self._patterns.items(), key=lambda kv: (kv[1].accuracy, kv[1].total_count))
        to_remove = max(1, len(sorted_patterns) // 10)
        for pid, _ in sorted_patterns[:to_remove]:
            del self._patterns[pid]


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
        sample = text[:2000]
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
            if best is None or 0.80 > best[2]:
                best = (doc_type, label, 0.80)

    if text:
        for pattern, doc_type, label, weight in _CONTENT_SIGNALS:
            if pattern.search(text):
                if best is None or weight > best[2]:
                    best = (doc_type, label, weight)

    if best and best[2] >= 0.85:
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


def _node_llama_index_similarity(state: ClassificationState, memory: ClassificationMemory) -> dict:
    """Tier 2: LlamaIndex VectorStoreIndex cosine similarity lookup."""
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
    if sgd_result and sgd_result[2] >= 0.6:
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
        front_matter_text=text[:4000],
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
        return (doc_type, label, min(confidence, 0.95))
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
                                [not resolved?] → llama_similarity → [resolved?] → END
                                                                      [not?] → sgd → [resolved?] → END
                                                                                      [not?] → llm → [resolved?] → END
                                                                                                      [not?] → fallback → END
    Uses conditional edges to short-circuit as soon as a tier resolves.
    """
    graph = StateGraph(ClassificationState)

    # Add nodes — each node is a lambda that closes over its dependencies
    graph.add_node("embed", lambda s: _node_compute_embedding(s, embedder))
    graph.add_node("deterministic", _node_deterministic)
    graph.add_node("llama_similarity", lambda s: _node_llama_index_similarity(s, memory))
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
        "unresolved": "llama_similarity",
    })
    graph.add_conditional_edges("llama_similarity", _is_resolved, {
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
    embed → deterministic → llama_similarity → sgd → llm → fallback

    Each tier short-circuits via conditional edges when resolved.
    Results stored in LlamaIndex VectorStoreIndex + sklearn SGDClassifier.
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
        self._store_in_memory(result, filename, front_matter_text, embedding)
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
