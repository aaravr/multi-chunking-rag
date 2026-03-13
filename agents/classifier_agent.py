"""Classifier Agent — self-learning document classification (MASTER_PROMPT §4.8).

Classifies ANY document into document_type and classification_label to determine
the processing pipeline. Uses a 3-tier classification strategy:

1. **Deterministic rules**: Filename patterns, structural signals (fast, free)
2. **Memory match**: Prior classification patterns learned from past documents
3. **LLM classification**: Tier-2 model for ambiguous documents (fallback)

Self-learning loop:
- After each classification, the result is stored in classification memory
- Memory patterns accumulate accuracy stats (success_count / total_count)
- High-accuracy patterns are promoted to "trusted" and used before LLM
- Low-accuracy patterns are demoted and eventually pruned
- The orchestrator can send feedback to reinforce/correct classifications

Memory persistence:
- In-process dict (default) or database-backed via classification_memory table
- Each pattern tracks: filename_pattern, title_keywords, structural_signals,
  success/total counts, accuracy
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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


class ClassificationMemory:
    """In-process classification memory store.

    Stores learned patterns from past classifications. Each pattern records:
    - What signals led to the classification
    - How many times the pattern was used (total_count)
    - How many times the classification was confirmed correct (success_count)

    Patterns with high accuracy (>= MEMORY_TRUST_THRESHOLD) and sufficient
    observations (>= MEMORY_MIN_OBSERVATIONS) are promoted to "trusted" and
    used in preference to LLM classification.
    """

    def __init__(self) -> None:
        self._patterns: Dict[str, ClassificationMemoryEntry] = {}
        self._filename_index: Dict[str, List[str]] = defaultdict(list)
        self._keyword_index: Dict[str, List[str]] = defaultdict(list)

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
        # Check for existing pattern with same type+label+filename
        existing = self._find_exact_match(
            document_type, classification_label, filename
        )
        if existing:
            # Increment usage count on existing pattern
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

        # Index for fast lookup
        if filename:
            normalized = self._normalize_filename(filename)
            self._filename_index[normalized].append(pattern_id)
        for kw in (title_keywords or []):
            self._keyword_index[kw.lower()].append(pattern_id)

        # Prune if over limit
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
        """Find the best matching trusted pattern from memory.

        Returns the highest-accuracy pattern that matches the filename or
        keywords AND meets the trust threshold.
        """
        candidates: List[ClassificationMemoryEntry] = []

        # Filename match
        if filename:
            normalized = self._normalize_filename(filename)
            for pid in self._filename_index.get(normalized, []):
                entry = self._patterns.get(pid)
                if entry and entry.total_count >= min_observations and entry.accuracy >= min_accuracy:
                    candidates.append(entry)

        # Keyword match
        if title_keywords:
            for kw in title_keywords:
                for pid in self._keyword_index.get(kw.lower(), []):
                    entry = self._patterns.get(pid)
                    if entry and entry.total_count >= min_observations and entry.accuracy >= min_accuracy:
                        candidates.append(entry)

        if not candidates:
            return None

        # Return the highest-accuracy candidate
        return max(candidates, key=lambda e: (e.accuracy, e.total_count))

    def record_feedback(self, pattern_id: str, correct: bool) -> None:
        """Record whether a classification was correct (reinforcement learning).

        Called by the orchestrator or user feedback loop.
        """
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
        }

    def export_json(self) -> str:
        """Export memory to JSON for persistence."""
        data = []
        for entry in self._patterns.values():
            data.append({
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
        return json.dumps(data, indent=2)

    def import_json(self, data: str) -> int:
        """Import patterns from JSON. Returns count of patterns loaded."""
        entries = json.loads(data)
        count = 0
        for item in entries:
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

    def _find_exact_match(
        self, document_type: str, classification_label: str, filename: Optional[str]
    ) -> Optional[str]:
        """Find an existing pattern matching type+label+filename."""
        for pid, entry in self._patterns.items():
            if (
                entry.document_type == document_type
                and entry.classification_label == classification_label
                and entry.filename_pattern == filename
            ):
                return pid
        return None

    def _normalize_filename(self, filename: str) -> str:
        """Normalize filename for pattern matching."""
        # Strip extension, lower, collapse whitespace
        name = re.sub(r"\.[^.]+$", "", filename)
        name = re.sub(r"[\s_-]+", " ", name.lower().strip())
        return name

    def _prune_lowest_accuracy(self) -> None:
        """Remove the lowest-accuracy patterns when memory is full."""
        sorted_patterns = sorted(
            self._patterns.items(),
            key=lambda kv: (kv[1].accuracy, kv[1].total_count),
        )
        # Remove bottom 10%
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


class ClassifierAgent(BaseAgent):
    """Self-learning document classifier (§4.8).

    Classification strategy (3-tier, cheapest first):

    1. **Deterministic rules** — regex on filename and first-page content.
       Fast, free, high confidence for known document types.

    2. **Memory match** — check classification memory for trusted patterns
       learned from prior documents. Only used when pattern accuracy >=
       MEMORY_TRUST_THRESHOLD and observations >= MEMORY_MIN_OBSERVATIONS.

    3. **LLM classification** — Tier-2 model analyzes front-matter text
       to determine document type. Used only for truly ambiguous documents.

    After classification, the result is stored in memory for future learning.
    The orchestrator or user can send feedback to reinforce/correct.

    MUST:
    - Try deterministic rules before LLM
    - Store every classification in memory
    - Track accuracy of learned patterns
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
    ) -> None:
        super().__init__(bus, gateway)
        self._memory = memory or get_classification_memory()

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
        """Classify a document using the 3-tier strategy.

        Returns a ClassificationResult with document_type, classification_label,
        confidence, and the method used.
        """
        start = time.monotonic()
        signals = dict(structural_signals or {})
        signals["filename"] = filename
        signals["page_count"] = page_count

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
            self._store_in_memory(result, filename, front_matter_text)
            self._log_classification(result, start, query_id)
            return result

        # ── Tier 2: Memory match ──────────────────────────────────────
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
            # Reinforce the memory pattern
            self._memory.store_pattern(
                document_type=mem_match.document_type,
                classification_label=mem_match.classification_label,
                filename=filename,
                title_keywords=title_keywords,
            )
            self._log_classification(result, start, query_id)
            return result

        # ── Tier 3: LLM classification ────────────────────────────────
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
                self._store_in_memory(result, filename, front_matter_text)
                self._log_classification(result, start, query_id)
                return result

        # ── Fallback: use deterministic result even if low confidence,
        #    or default to "unknown" ───────────────────────────────────
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
        self._store_in_memory(result, filename, front_matter_text)
        self._log_classification(result, start, query_id)
        return result

    def handle_feedback(self, pattern_id: str, correct: bool) -> None:
        """Process feedback on a classification (called by orchestrator).

        Reinforces correct patterns or demotes incorrect ones in memory.
        """
        self._memory.record_feedback(pattern_id, correct)
        logger.info(
            "Classification feedback: pattern=%s correct=%s",
            pattern_id, correct,
        )

    def get_memory_stats(self) -> Dict[str, Any]:
        """Return classification memory statistics."""
        return self._memory.get_stats()

    # ── Tier 1: Deterministic classification ──────────────────────────

    def _classify_deterministic(
        self, filename: str, front_matter_text: str
    ) -> Optional[Tuple[str, str, float]]:
        """Try deterministic classification using filename and content patterns.

        Returns (document_type, classification_label, confidence) or None.
        """
        best: Optional[Tuple[str, str, float]] = None

        # Check filename patterns
        for pattern, doc_type, label in _FILENAME_PATTERNS:
            if pattern.search(filename):
                confidence = 0.80
                if best is None or confidence > best[2]:
                    best = (doc_type, label, confidence)

        # Check content signals (higher weight than filename)
        if front_matter_text:
            for pattern, doc_type, label, weight in _CONTENT_SIGNALS:
                if pattern.search(front_matter_text):
                    if best is None or weight > best[2]:
                        best = (doc_type, label, weight)

        return best

    # ── Tier 3: LLM classification ────────────────────────────────────

    def _classify_with_llm(
        self,
        doc_id: str,
        filename: str,
        front_matter_text: str,
        page_count: int,
        query_id: str,
    ) -> Optional[Tuple[str, str, float]]:
        """Use Tier-2 LLM to classify an ambiguous document.

        Returns (document_type, classification_label, confidence) or None.
        """
        # Truncate front matter to avoid exceeding token limits
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
        """Parse the LLM classification response.

        Expected JSON format:
        {"document_type": "...", "classification_label": "...", "confidence": 0.85}
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
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

    def _extract_title_keywords(self, text: str) -> List[str]:
        """Extract likely title/heading keywords from front-matter text.

        Looks at the first ~500 chars for capitalized phrases.
        """
        if not text:
            return []
        sample = text[:500]
        # Find capitalized multi-word phrases (likely titles/headings)
        phrases = re.findall(r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+", sample)
        # Also grab any ALL-CAPS phrases
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
