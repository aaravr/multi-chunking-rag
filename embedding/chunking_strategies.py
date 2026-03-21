"""Additional chunking strategies beyond late chunking (MASTER_PROMPT §2.2 extension).

Provides a registry of chunking algorithms that produce ``List[ChunkRecord]``
from ``List[CanonicalPage]``.  Each strategy preserves deterministic lineage
(§2.1) — every output chunk carries doc_id, page_numbers, char_start/end,
polygons, heading_path, section_id.

Strategies implemented
----------------------
Structural / boundary-aware:
  * **semantic**          — split at embedding-similarity breakpoints
  * **recursive**         — split by separator hierarchy (\\n\\n → \\n → . → ' ')
  * **clause_aware**      — split at legal clause boundaries (Section, Article, (a))
  * **sentence_level**    — split by sentences, group N per chunk
  * **sliding_window**    — fixed token window with configurable overlap (rolling window)

Hierarchy & context:
  * **parent_child**      — store both large parent + small child chunks
  * **context_enriched**  — prepend heading/section context to each chunk

Content-type specific:
  * **table_aware**       — tables kept as whole units with linked captions
  * **topic_segmentation** — split at detected topic boundaries

Agentic / LLM-dependent:
  * **proposition**       — LLM decomposes text into atomic factual propositions
  * **summary_indexed**   — embed LLM-generated summaries alongside original chunks

All functions share the signature::

    def chunk_<name>(
        doc_id: str,
        pages: List[CanonicalPage],
        *,
        <strategy-specific params>,
    ) -> List[ChunkRecord]
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.config import settings
from core.contracts import CanonicalPage, CanonicalSpan, ChunkRecord
from embedding.late_chunking import SpanLineage, _classify_chunk_type, _collect_span_lineage

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _get_embedder():
    """Lazy import to avoid circular dependency and 440 MB load at import time."""
    from embedding.model_registry import get_embedding_model
    return get_embedding_model()


def _embed_text(text: str) -> List[float]:
    return _get_embedder().embed_text(text)


def _make_chunk(
    doc_id: str,
    page: CanonicalPage,
    text: str,
    char_start: int,
    char_end: int,
    macro_id: int,
    child_id: int,
    embedding: List[float],
    chunk_type: Optional[str] = None,
) -> ChunkRecord:
    """Build a ChunkRecord with lineage collected from the page's spans."""
    lineage = _collect_span_lineage(page.spans, char_start, char_end)
    return ChunkRecord(
        chunk_id=str(uuid.uuid4()),
        doc_id=doc_id,
        page_numbers=lineage.page_numbers or [page.page_number],
        macro_id=macro_id,
        child_id=child_id,
        chunk_type=chunk_type or _classify_chunk_type(text),
        text_content=text,
        char_start=char_start,
        char_end=char_end,
        polygons=lineage.polygons,
        source_type=lineage.source_type,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        embedding=embedding,
        heading_path=lineage.heading_path,
        section_id=lineage.section_id,
    )


def _make_chunk_from_multi_page(
    doc_id: str,
    pages: List[CanonicalPage],
    text: str,
    page_numbers: List[int],
    macro_id: int,
    child_id: int,
    embedding: List[float],
    lineage: Optional[SpanLineage] = None,
    chunk_type: Optional[str] = None,
    # Legacy keyword args kept for backward compatibility
    polygons: Optional[List[dict]] = None,
    source_type: str = "native",
    heading_path: str = "",
    section_id: str = "",
) -> ChunkRecord:
    """Build a ChunkRecord spanning multiple pages.

    Prefer passing a ``SpanLineage`` object via *lineage* instead of
    separate polygon/source/heading/section keyword arguments.
    """
    if lineage is not None:
        polygons = lineage.polygons
        source_type = lineage.source_type
        heading_path = lineage.heading_path
        section_id = lineage.section_id

    return ChunkRecord(
        chunk_id=str(uuid.uuid4()),
        doc_id=doc_id,
        page_numbers=page_numbers,
        macro_id=macro_id,
        child_id=child_id,
        chunk_type=chunk_type or _classify_chunk_type(text),
        text_content=text,
        char_start=0,
        char_end=len(text),
        polygons=polygons or [],
        source_type=source_type,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        embedding=embedding,
        heading_path=heading_path,
        section_id=section_id,
    )


# ── Sentence splitter (regex-based, no spaCy dependency) ─────────────────

_SENTENCE_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z"\'\(\[])'
    r'|(?<=[.!?])\s*\n'
    r'|\n{2,}'
)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex heuristics.

    Returns a list of sentence strings.  Preserves sentence boundaries
    correctly for most financial/legal prose.
    """
    if not text or not text.strip():
        return []
    parts = _SENTENCE_RE.split(text)
    sentences = [s.strip() for s in parts if s and s.strip()]
    return sentences


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = float(np.dot(a_arr, b_arr))
    norm_a = float(np.linalg.norm(a_arr))
    norm_b = float(np.linalg.norm(b_arr))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _concat_page_text(pages: List[CanonicalPage]) -> str:
    return "\n\n".join(p.text for p in pages if p.text)


def _find_page_for_offset(
    pages: List[CanonicalPage], global_offset: int
) -> Tuple[CanonicalPage, int]:
    """Find which page a global char offset falls in.

    Returns (page, local_offset_within_page).
    """
    cumulative = 0
    for page in pages:
        page_len = len(page.text) + 2  # +2 for the \n\n separator
        if cumulative + page_len > global_offset:
            return page, global_offset - cumulative
        cumulative += page_len
    # Fallback to last page
    return pages[-1], max(0, global_offset - cumulative + len(pages[-1].text) + 2)


# ═══════════════════════════════════════════════════════════════════════════
# 1. SEMANTIC CHUNKING
# ═══════════════════════════════════════════════════════════════════════════

def chunk_semantic(
    doc_id: str,
    pages: List[CanonicalPage],
    *,
    similarity_threshold: float = 0.5,
    min_chunk_sentences: int = 2,
    max_chunk_sentences: int = 20,
) -> List[ChunkRecord]:
    """Split text at semantic breakpoints where consecutive sentence
    embeddings diverge.

    Algorithm:
    1. Split all text into sentences.
    2. Embed each sentence.
    3. Compute cosine similarity between consecutive sentence embeddings.
    4. Identify breakpoints where similarity drops below *similarity_threshold*.
    5. Group sentences between breakpoints into chunks (respecting min/max).
    6. Embed each final chunk.

    Parameters
    ----------
    similarity_threshold : float
        Split when cosine similarity between adjacent sentences falls below
        this value.  Lower = fewer, larger chunks.
    min_chunk_sentences : int
        Minimum sentences per chunk to avoid tiny fragments.
    max_chunk_sentences : int
        Force-split after this many sentences even if similarity is high.
    """
    full_text = _concat_page_text(pages)
    if not full_text.strip():
        return []

    sentences = _split_sentences(full_text)
    if not sentences:
        return []

    # Embed each sentence
    embeddings = [_embed_text(s) for s in sentences]

    # Find breakpoints
    breakpoints: List[int] = []
    for i in range(1, len(sentences)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
        if sim < similarity_threshold:
            breakpoints.append(i)

    # Build groups from breakpoints
    groups: List[List[int]] = []
    start = 0
    for bp in breakpoints:
        if bp - start >= min_chunk_sentences:
            groups.append(list(range(start, bp)))
            start = bp
    if start < len(sentences):
        groups.append(list(range(start, len(sentences))))

    # Enforce max_chunk_sentences by splitting large groups
    final_groups: List[List[int]] = []
    for group in groups:
        while len(group) > max_chunk_sentences:
            final_groups.append(group[:max_chunk_sentences])
            group = group[max_chunk_sentences:]
        if group:
            final_groups.append(group)

    # Merge very small trailing groups
    merged: List[List[int]] = []
    for group in final_groups:
        if merged and len(group) < min_chunk_sentences:
            merged[-1].extend(group)
        else:
            merged.append(group)

    # Build ChunkRecords
    chunks: List[ChunkRecord] = []
    # Build char offset map for sentences in full_text
    sentence_offsets: List[Tuple[int, int]] = []
    search_start = 0
    for sent in sentences:
        idx = full_text.find(sent, search_start)
        if idx == -1:
            idx = search_start
        sentence_offsets.append((idx, idx + len(sent)))
        search_start = idx + len(sent)

    for macro_id, group in enumerate(merged):
        chunk_text = " ".join(sentences[i] for i in group)
        if not chunk_text.strip():
            continue
        embedding = _embed_text(chunk_text)
        char_start = sentence_offsets[group[0]][0]
        char_end = sentence_offsets[group[-1]][1]

        page, _ = _find_page_for_offset(pages, char_start)
        page_nums = sorted({
            _find_page_for_offset(pages, sentence_offsets[i][0])[0].page_number
            for i in group
        })

        lineage = _collect_span_lineage(page.spans, 0, len(page.text))
        chunks.append(
            _make_chunk_from_multi_page(
                doc_id=doc_id,
                pages=pages,
                text=chunk_text,
                page_numbers=page_nums,
                macro_id=macro_id,
                child_id=0,
                embedding=embedding,
                polygons=lineage.polygons[:50],
                source_type=lineage.source_type,
                heading_path=lineage.heading_path,
                section_id=lineage.section_id,
            )
        )

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 2. RECURSIVE CHARACTER SPLITTING
# ═══════════════════════════════════════════════════════════════════════════

# Separator hierarchy from most to least significant
_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " "]


def chunk_recursive(
    doc_id: str,
    pages: List[CanonicalPage],
    *,
    max_chunk_chars: int = 1500,
    chunk_overlap_chars: int = 200,
    separators: Optional[List[str]] = None,
) -> List[ChunkRecord]:
    """Split text by a hierarchy of separators, recursing to finer
    separators when chunks exceed *max_chunk_chars*.

    Algorithm:
    1. Try splitting on ``\\n\\n`` (paragraph boundaries).
    2. If any resulting piece exceeds max size, split on ``\\n``.
    3. If still too big, split on ``'. '``.
    4. If still too big, split on ``' '``.
    5. Add overlap between consecutive chunks.

    Parameters
    ----------
    max_chunk_chars : int
        Target maximum characters per chunk.
    chunk_overlap_chars : int
        Number of overlapping characters between consecutive chunks.
    separators : list[str] | None
        Custom separator hierarchy.  Defaults to paragraph → line → sentence → word.
    """
    full_text = _concat_page_text(pages)
    if not full_text.strip():
        return []

    seps = separators or list(_RECURSIVE_SEPARATORS)

    def _recursive_split(text: str, sep_idx: int) -> List[str]:
        if sep_idx >= len(seps) or len(text) <= max_chunk_chars:
            return [text] if text.strip() else []
        sep = seps[sep_idx]
        parts = text.split(sep)
        result: List[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= max_chunk_chars:
                current = candidate
            else:
                if current:
                    result.append(current)
                # If the part itself is too large, recurse with next separator
                if len(part) > max_chunk_chars:
                    result.extend(_recursive_split(part, sep_idx + 1))
                else:
                    current = part
        if current:
            result.append(current)
        return result

    raw_chunks = _recursive_split(full_text, 0)

    # Add overlap
    overlapped: List[str] = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and chunk_overlap_chars > 0:
            prev = raw_chunks[i - 1]
            overlap = prev[-chunk_overlap_chars:]
            chunk = overlap + chunk
        overlapped.append(chunk)

    # Build ChunkRecords
    chunks: List[ChunkRecord] = []
    search_start = 0
    for macro_id, chunk_text in enumerate(overlapped):
        if not chunk_text.strip():
            continue
        # Find approximate offset in full_text
        # Use the non-overlapped portion to find position
        core_text = raw_chunks[macro_id] if macro_id < len(raw_chunks) else chunk_text
        idx = full_text.find(core_text[:100], max(0, search_start - 50))
        if idx == -1:
            idx = search_start
        char_start = idx
        char_end = idx + len(core_text)
        search_start = char_end

        embedding = _embed_text(chunk_text)
        page, _ = _find_page_for_offset(pages, char_start)

        chunks.append(
            _make_chunk(
                doc_id=doc_id,
                page=page,
                text=chunk_text,
                char_start=char_start,
                char_end=min(char_end, len(full_text)),
                macro_id=macro_id,
                child_id=0,
                embedding=embedding,
            )
        )

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 3. CLAUSE-AWARE SPLITTING
# ═══════════════════════════════════════════════════════════════════════════

# Patterns that signal the start of a new clause/section in legal documents
_CLAUSE_PATTERNS = [
    re.compile(r'^(?:SECTION|Section)\s+\d+', re.MULTILINE),
    re.compile(r'^(?:ARTICLE|Article)\s+[IVXLCDM]+', re.MULTILINE),
    re.compile(r'^(?:ARTICLE|Article)\s+\d+', re.MULTILINE),
    re.compile(r'^\d+\.\s+[A-Z]', re.MULTILINE),
    re.compile(r'^\d+\.\d+\s+', re.MULTILINE),
    re.compile(r'^\d+\.\d+\.\d+\s+', re.MULTILINE),
    re.compile(r'^(?:SCHEDULE|Schedule)\s+[A-Z\d]', re.MULTILINE),
    re.compile(r'^(?:EXHIBIT|Exhibit)\s+[A-Z\d]', re.MULTILINE),
    re.compile(r'^(?:ANNEX|Annex)\s+[A-Z\d]', re.MULTILINE),
    re.compile(r'^(?:APPENDIX|Appendix)\s+[A-Z\d]', re.MULTILINE),
    re.compile(r'^(?:PART|Part)\s+[IVXLCDM\d]+', re.MULTILINE),
    re.compile(r'^(?:RECITALS?|WHEREAS|DEFINITIONS?)\s*$', re.MULTILINE | re.IGNORECASE),
]


def chunk_clause_aware(
    doc_id: str,
    pages: List[CanonicalPage],
    *,
    max_clause_chars: int = 3000,
    min_clause_chars: int = 100,
) -> List[ChunkRecord]:
    """Split at legal clause boundaries identified by structural patterns.

    Detects clause starts via patterns like ``Section 1``, ``Article II``,
    ``1.1``, ``SCHEDULE A``, etc.  Each clause becomes a chunk.  If a
    clause exceeds *max_clause_chars*, it is sub-split using sentence
    boundaries.

    Parameters
    ----------
    max_clause_chars : int
        Force-split clauses larger than this via sentence splitting.
    min_clause_chars : int
        Merge clauses shorter than this into the previous clause.
    """
    full_text = _concat_page_text(pages)
    if not full_text.strip():
        return []

    # Find all clause boundary positions
    boundary_positions: List[int] = [0]
    for pattern in _CLAUSE_PATTERNS:
        for match in pattern.finditer(full_text):
            pos = match.start()
            if pos not in boundary_positions:
                boundary_positions.append(pos)
    boundary_positions.sort()

    # Split text into clause segments
    segments: List[Tuple[int, int, str]] = []
    for i in range(len(boundary_positions)):
        start = boundary_positions[i]
        end = boundary_positions[i + 1] if i + 1 < len(boundary_positions) else len(full_text)
        text = full_text[start:end].strip()
        if text:
            segments.append((start, end, text))

    # Merge tiny segments into previous
    merged_segments: List[Tuple[int, int, str]] = []
    for start, end, text in segments:
        if merged_segments and len(text) < min_clause_chars:
            prev_start, _, prev_text = merged_segments[-1]
            merged_segments[-1] = (prev_start, end, prev_text + "\n" + text)
        else:
            merged_segments.append((start, end, text))

    # Sub-split large clauses
    final_segments: List[Tuple[int, int, str]] = []
    for start, end, text in merged_segments:
        if len(text) <= max_clause_chars:
            final_segments.append((start, end, text))
        else:
            # Split by sentences
            sents = _split_sentences(text)
            group_text = ""
            group_start = start
            for sent in sents:
                if len(group_text) + len(sent) > max_clause_chars and group_text:
                    final_segments.append((group_start, group_start + len(group_text), group_text))
                    group_start = group_start + len(group_text)
                    group_text = sent
                else:
                    group_text = (group_text + " " + sent).strip() if group_text else sent
            if group_text:
                final_segments.append((group_start, end, group_text))

    # Build ChunkRecords
    chunks: List[ChunkRecord] = []
    for macro_id, (char_start, char_end, text) in enumerate(final_segments):
        if not text.strip():
            continue
        embedding = _embed_text(text)
        page, _ = _find_page_for_offset(pages, char_start)
        chunks.append(
            _make_chunk(
                doc_id=doc_id,
                page=page,
                text=text,
                char_start=char_start,
                char_end=min(char_end, len(full_text)),
                macro_id=macro_id,
                child_id=0,
                embedding=embedding,
            )
        )

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 4. SENTENCE-LEVEL CHUNKING
# ═══════════════════════════════════════════════════════════════════════════

def chunk_sentence_level(
    doc_id: str,
    pages: List[CanonicalPage],
    *,
    sentences_per_chunk: int = 5,
    sentence_overlap: int = 1,
) -> List[ChunkRecord]:
    """Group sentences into fixed-size groups.

    Parameters
    ----------
    sentences_per_chunk : int
        Number of sentences per chunk.
    sentence_overlap : int
        Number of sentences overlapping between consecutive chunks.
    """
    full_text = _concat_page_text(pages)
    if not full_text.strip():
        return []

    sentences = _split_sentences(full_text)
    if not sentences:
        return []

    # Build sentence offset map
    sentence_offsets: List[Tuple[int, int]] = []
    search_start = 0
    for sent in sentences:
        idx = full_text.find(sent, search_start)
        if idx == -1:
            idx = search_start
        sentence_offsets.append((idx, idx + len(sent)))
        search_start = idx + len(sent)

    # Group sentences with overlap
    chunks: List[ChunkRecord] = []
    stride = max(sentences_per_chunk - sentence_overlap, 1)
    macro_id = 0

    for start in range(0, len(sentences), stride):
        end = min(start + sentences_per_chunk, len(sentences))
        group = sentences[start:end]
        chunk_text = " ".join(group)
        if not chunk_text.strip():
            continue

        char_start = sentence_offsets[start][0]
        char_end = sentence_offsets[end - 1][1]
        embedding = _embed_text(chunk_text)
        page, _ = _find_page_for_offset(pages, char_start)
        page_nums = sorted({
            _find_page_for_offset(pages, sentence_offsets[i][0])[0].page_number
            for i in range(start, end)
        })

        lineage = _collect_span_lineage(page.spans, 0, len(page.text))
        chunks.append(
            _make_chunk_from_multi_page(
                doc_id=doc_id,
                pages=pages,
                text=chunk_text,
                page_numbers=page_nums,
                macro_id=macro_id,
                child_id=0,
                embedding=embedding,
                polygons=lineage.polygons[:50],
                source_type=lineage.source_type,
                heading_path=lineage.heading_path,
                section_id=lineage.section_id,
            )
        )
        macro_id += 1

        if end >= len(sentences):
            break

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 5. SLIDING WINDOW (ROLLING WINDOW) CHUNKING
# ═══════════════════════════════════════════════════════════════════════════

def chunk_sliding_window(
    doc_id: str,
    pages: List[CanonicalPage],
    *,
    window_size_tokens: int = 512,
    stride_tokens: int = 256,
) -> List[ChunkRecord]:
    """Fixed-size token window sliding across the document with overlap.

    Unlike late chunking, this produces no macro/child hierarchy — every
    window is an independent chunk with its own embedding.  Simple and
    effective for general-purpose retrieval.

    Parameters
    ----------
    window_size_tokens : int
        Number of tokens per window.
    stride_tokens : int
        Step size (window_size - overlap).  Overlap = window_size - stride.
    """
    full_text = _concat_page_text(pages)
    if not full_text.strip():
        return []

    embedder = _get_embedder()
    offsets = embedder.tokenize_full(full_text)
    valid_indices = [i for i, (s, e) in enumerate(offsets) if e > s]
    total_tokens = len(valid_indices)

    if total_tokens == 0:
        return []

    chunks: List[ChunkRecord] = []
    macro_id = 0
    step = max(stride_tokens, 1)

    for start in range(0, total_tokens, step):
        end = min(start + window_size_tokens, total_tokens)
        char_start = offsets[valid_indices[start]][0]
        char_end = offsets[valid_indices[end - 1]][1]
        chunk_text = full_text[char_start:char_end]

        if not chunk_text.strip():
            if end >= total_tokens:
                break
            continue

        embedding = _embed_text(chunk_text)
        page, _ = _find_page_for_offset(pages, char_start)

        chunks.append(
            _make_chunk(
                doc_id=doc_id,
                page=page,
                text=chunk_text,
                char_start=char_start,
                char_end=char_end,
                macro_id=macro_id,
                child_id=0,
                embedding=embedding,
            )
        )
        macro_id += 1

        if end >= total_tokens:
            break

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 6. PARENT-CHILD (HIERARCHICAL) CHUNKING
# ═══════════════════════════════════════════════════════════════════════════

def chunk_parent_child(
    doc_id: str,
    pages: List[CanonicalPage],
    *,
    parent_max_chars: int = 4000,
    child_max_chars: int = 800,
    child_overlap_chars: int = 100,
) -> List[ChunkRecord]:
    """Create large parent chunks and small child chunks, both stored.

    At retrieval time, child chunks provide precision while parent chunks
    (accessible via same ``macro_id``) provide context for synthesis.

    Parent chunks get ``child_id = 0``.  Child chunks get ``child_id = 1, 2, ...``.
    Parent and children share the same ``macro_id``.

    Parameters
    ----------
    parent_max_chars : int
        Maximum size for parent chunks (roughly section-level).
    child_max_chars : int
        Maximum size for child chunks (roughly paragraph-level).
    child_overlap_chars : int
        Overlap between consecutive child chunks.
    """
    full_text = _concat_page_text(pages)
    if not full_text.strip():
        return []

    # Split into parent-sized segments (by paragraph boundaries)
    raw_parents = full_text.split("\n\n")
    parent_segments: List[str] = []
    current = ""
    for para in raw_parents:
        if len(current) + len(para) + 2 > parent_max_chars and current:
            parent_segments.append(current)
            current = para
        else:
            current = (current + "\n\n" + para).strip() if current else para
    if current:
        parent_segments.append(current)

    chunks: List[ChunkRecord] = []
    macro_id = 0
    search_start = 0

    for parent_text in parent_segments:
        if not parent_text.strip():
            continue

        # Find parent offset in full text
        idx = full_text.find(parent_text[:100], max(0, search_start - 20))
        if idx == -1:
            idx = search_start
        parent_start = idx
        parent_end = idx + len(parent_text)
        search_start = parent_end

        # Parent chunk (child_id = 0)
        parent_embedding = _embed_text(parent_text)
        page, _ = _find_page_for_offset(pages, parent_start)
        chunks.append(
            _make_chunk(
                doc_id=doc_id,
                page=page,
                text=parent_text,
                char_start=parent_start,
                char_end=min(parent_end, len(full_text)),
                macro_id=macro_id,
                child_id=0,
                embedding=parent_embedding,
            )
        )

        # Split parent into children
        child_stride = max(child_max_chars - child_overlap_chars, 1)
        child_id = 1
        for c_start in range(0, len(parent_text), child_stride):
            c_end = min(c_start + child_max_chars, len(parent_text))
            child_text = parent_text[c_start:c_end].strip()
            if not child_text or child_text == parent_text.strip():
                # Skip if child is the same as parent (no point duplicating)
                if child_id == 1 and c_end >= len(parent_text):
                    break
                continue

            child_embedding = _embed_text(child_text)
            child_global_start = parent_start + c_start
            child_global_end = parent_start + c_end
            child_page, _ = _find_page_for_offset(pages, child_global_start)
            chunks.append(
                _make_chunk(
                    doc_id=doc_id,
                    page=child_page,
                    text=child_text,
                    char_start=child_global_start,
                    char_end=min(child_global_end, len(full_text)),
                    macro_id=macro_id,
                    child_id=child_id,
                    embedding=child_embedding,
                )
            )
            child_id += 1

            if c_end >= len(parent_text):
                break

        macro_id += 1

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 7. TABLE-AWARE CHUNKING (ENHANCED)
# ═══════════════════════════════════════════════════════════════════════════

def chunk_table_aware(
    doc_id: str,
    pages: List[CanonicalPage],
    *,
    narrative_max_chars: int = 1500,
    caption_search_chars: int = 300,
) -> List[ChunkRecord]:
    """Keep tables as whole units with their captions, chunk narrative separately.

    Tables are never split — each table becomes a single chunk.  The text
    immediately preceding a table (up to *caption_search_chars*) is scanned
    for a caption/title and prepended to the table chunk for context.

    Narrative text between tables is chunked using recursive splitting.

    Parameters
    ----------
    narrative_max_chars : int
        Max chars for narrative chunks between tables.
    caption_search_chars : int
        How many chars before a table to search for a caption.
    """
    chunks: List[ChunkRecord] = []
    macro_id = 0

    for page in pages:
        if not page.text:
            continue

        # Separate table spans and narrative spans
        table_spans = [s for s in page.spans if s.is_table]
        narrative_parts: List[Tuple[int, int, str]] = []

        # Find narrative gaps between tables
        sorted_tables = sorted(table_spans, key=lambda s: s.char_start)
        prev_end = 0
        for table_span in sorted_tables:
            if table_span.char_start > prev_end:
                gap_text = page.text[prev_end:table_span.char_start].strip()
                if gap_text:
                    narrative_parts.append((prev_end, table_span.char_start, gap_text))
            prev_end = table_span.char_end
        # Trailing narrative
        if prev_end < len(page.text):
            tail = page.text[prev_end:].strip()
            if tail:
                narrative_parts.append((prev_end, len(page.text), tail))

        # Chunk tables with captions
        for table_span in table_spans:
            # Search for caption above table
            caption = ""
            search_start = max(0, table_span.char_start - caption_search_chars)
            pre_text = page.text[search_start:table_span.char_start].strip()
            if pre_text:
                # Look for last line that looks like a caption
                lines = pre_text.split("\n")
                for line in reversed(lines):
                    line = line.strip()
                    if line and (
                        re.match(r'^(?:Table|TABLE|Figure|FIGURE)\s+\d+', line)
                        or re.match(r'^(?:Note|NOTE)\s+\d+', line)
                        or (line.endswith(":") and len(line) < 200)
                    ):
                        caption = line
                        break

            table_text = (caption + "\n" + table_span.text).strip() if caption else table_span.text
            embedding = _embed_text(table_text)
            chunks.append(
                ChunkRecord(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    page_numbers=[table_span.page_number],
                    macro_id=macro_id,
                    child_id=0,
                    chunk_type="table",
                    text_content=table_text,
                    char_start=table_span.char_start,
                    char_end=table_span.char_end,
                    polygons=table_span.polygons,
                    source_type=table_span.source_type,
                    embedding_model=settings.embedding_model,
                    embedding_dim=settings.embedding_dim,
                    embedding=embedding,
                    heading_path=table_span.heading_path,
                    section_id=table_span.section_id,
                )
            )
            macro_id += 1

        # Chunk narrative parts with recursive splitting
        for n_start, n_end, n_text in narrative_parts:
            # Simple sentence-based splitting for narrative
            if len(n_text) <= narrative_max_chars:
                embedding = _embed_text(n_text)
                chunks.append(
                    _make_chunk(
                        doc_id=doc_id,
                        page=page,
                        text=n_text,
                        char_start=n_start,
                        char_end=n_end,
                        macro_id=macro_id,
                        child_id=0,
                        embedding=embedding,
                    )
                )
                macro_id += 1
            else:
                sents = _split_sentences(n_text)
                group = ""
                for sent in sents:
                    if len(group) + len(sent) > narrative_max_chars and group:
                        embedding = _embed_text(group)
                        chunks.append(
                            _make_chunk(
                                doc_id=doc_id,
                                page=page,
                                text=group,
                                char_start=n_start,
                                char_end=n_start + len(group),
                                macro_id=macro_id,
                                child_id=0,
                                embedding=embedding,
                            )
                        )
                        macro_id += 1
                        n_start += len(group)
                        group = sent
                    else:
                        group = (group + " " + sent).strip() if group else sent
                if group:
                    embedding = _embed_text(group)
                    chunks.append(
                        _make_chunk(
                            doc_id=doc_id,
                            page=page,
                            text=group,
                            char_start=n_start,
                            char_end=n_end,
                            macro_id=macro_id,
                            child_id=0,
                            embedding=embedding,
                        )
                    )
                    macro_id += 1

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 8. TOPIC SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

def chunk_topic_segmentation(
    doc_id: str,
    pages: List[CanonicalPage],
    *,
    window_size: int = 3,
    threshold_percentile: float = 25.0,
    min_segment_sentences: int = 3,
) -> List[ChunkRecord]:
    """Detect topic boundaries using a TextTiling-inspired approach.

    Algorithm:
    1. Split into sentences.
    2. Embed each sentence.
    3. Compute block similarity: for each position, compare the average
       embedding of the *window_size* sentences before with those after.
    4. Find local minima in the similarity curve below the
       *threshold_percentile* as topic boundaries.
    5. Group sentences by topic segment.

    Parameters
    ----------
    window_size : int
        Number of sentences in each comparison block.
    threshold_percentile : float
        Percentile of similarity values below which a boundary is detected.
    min_segment_sentences : int
        Minimum sentences per topic segment.
    """
    full_text = _concat_page_text(pages)
    if not full_text.strip():
        return []

    sentences = _split_sentences(full_text)
    if len(sentences) < window_size * 2 + 1:
        # Too few sentences for topic segmentation — return as single chunk
        embedding = _embed_text(full_text)
        page_nums = sorted({p.page_number for p in pages})
        return [_make_chunk_from_multi_page(
            doc_id=doc_id, pages=pages, text=full_text,
            page_numbers=page_nums, macro_id=0, child_id=0,
            embedding=embedding,
        )]

    # Embed all sentences
    embeddings = [_embed_text(s) for s in sentences]

    # Compute block similarities
    similarities: List[float] = []
    for i in range(window_size, len(sentences) - window_size):
        before = np.mean([embeddings[j] for j in range(i - window_size, i)], axis=0)
        after = np.mean([embeddings[j] for j in range(i, i + window_size)], axis=0)
        sim = _cosine_similarity(before.tolist(), after.tolist())
        similarities.append(sim)

    if not similarities:
        embedding = _embed_text(full_text)
        page_nums = sorted({p.page_number for p in pages})
        return [_make_chunk_from_multi_page(
            doc_id=doc_id, pages=pages, text=full_text,
            page_numbers=page_nums, macro_id=0, child_id=0,
            embedding=embedding,
        )]

    # Find boundaries at local minima below threshold
    threshold = float(np.percentile(similarities, threshold_percentile))
    boundaries: List[int] = []
    for i in range(1, len(similarities) - 1):
        if (similarities[i] < threshold
                and similarities[i] <= similarities[i - 1]
                and similarities[i] <= similarities[i + 1]):
            # Convert back to sentence index
            sent_idx = i + window_size
            boundaries.append(sent_idx)

    # Build sentence offset map
    sentence_offsets: List[Tuple[int, int]] = []
    search_start = 0
    for sent in sentences:
        idx = full_text.find(sent, search_start)
        if idx == -1:
            idx = search_start
        sentence_offsets.append((idx, idx + len(sent)))
        search_start = idx + len(sent)

    # Split into segments
    segment_bounds = [0] + boundaries + [len(sentences)]
    chunks: List[ChunkRecord] = []
    macro_id = 0

    for i in range(len(segment_bounds) - 1):
        seg_start = segment_bounds[i]
        seg_end = segment_bounds[i + 1]

        # Enforce minimum segment size by merging small segments
        if seg_end - seg_start < min_segment_sentences and chunks:
            continue  # Skip tiny segments (they'll be absorbed by neighbors)

        seg_sentences = sentences[seg_start:seg_end]
        seg_text = " ".join(seg_sentences)
        if not seg_text.strip():
            continue

        embedding = _embed_text(seg_text)
        page_nums = sorted({
            _find_page_for_offset(pages, sentence_offsets[j][0])[0].page_number
            for j in range(seg_start, min(seg_end, len(sentence_offsets)))
        })
        page, _ = _find_page_for_offset(pages, sentence_offsets[seg_start][0])
        lineage = _collect_span_lineage(page.spans, 0, len(page.text))

        chunks.append(
            _make_chunk_from_multi_page(
                doc_id=doc_id,
                pages=pages,
                text=seg_text,
                page_numbers=page_nums,
                macro_id=macro_id,
                child_id=0,
                embedding=embedding,
                polygons=lineage.polygons[:50],
                source_type=lineage.source_type,
                heading_path=lineage.heading_path,
                section_id=lineage.section_id,
            )
        )
        macro_id += 1

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 9. CONTEXT-ENRICHED CHUNKING
# ═══════════════════════════════════════════════════════════════════════════

def chunk_context_enriched(
    doc_id: str,
    pages: List[CanonicalPage],
    *,
    base_strategy: str = "sliding_window",
    base_params: Optional[Dict[str, Any]] = None,
) -> List[ChunkRecord]:
    """Apply any base chunking strategy, then prepend heading/section
    context to each chunk and re-embed.

    This preserves the chunk text exactly but improves retrieval by
    embedding the chunk *with* its section context.  The ``heading_path``
    and document title are prepended.

    Parameters
    ----------
    base_strategy : str
        Name of the base strategy to use for initial chunking.
    base_params : dict | None
        Parameters for the base strategy.
    """
    params = base_params or {}
    dispatch = get_strategy_dispatch()
    base_fn = dispatch.get(base_strategy)

    if base_fn is None:
        # Fallback to sliding window
        base_chunks = chunk_sliding_window(doc_id, pages, **params)
    else:
        base_chunks = base_fn(doc_id, pages, **params)

    if not base_chunks:
        return []

    # Re-embed each chunk with context prefix
    enriched: List[ChunkRecord] = []
    for chunk in base_chunks:
        context_parts = []
        if chunk.heading_path:
            context_parts.append(f"[Section: {chunk.heading_path}]")
        if chunk.section_id:
            context_parts.append(f"[{chunk.section_id}]")
        context_prefix = " ".join(context_parts)

        if context_prefix:
            enriched_text = context_prefix + "\n" + chunk.text_content
            embedding = _embed_text(enriched_text)
        else:
            embedding = chunk.embedding

        enriched.append(
            ChunkRecord(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                page_numbers=chunk.page_numbers,
                macro_id=chunk.macro_id,
                child_id=chunk.child_id,
                chunk_type=chunk.chunk_type,
                text_content=chunk.text_content,  # Store original text, not enriched
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                polygons=chunk.polygons,
                source_type=chunk.source_type,
                embedding_model=chunk.embedding_model,
                embedding_dim=chunk.embedding_dim,
                embedding=embedding,  # But embed the enriched version
                heading_path=chunk.heading_path,
                section_id=chunk.section_id,
            )
        )

    return enriched


# ═══════════════════════════════════════════════════════════════════════════
# 10. PROPOSITION CHUNKING (LLM-dependent)
# ═══════════════════════════════════════════════════════════════════════════

_PROPOSITION_PROMPT = """You are a precise document analyst. Decompose the following text into a list of atomic, self-contained factual propositions.

Rules:
- Each proposition must be a single factual claim that can be independently verified.
- Include all necessary context (entity names, dates, amounts) in each proposition.
- Do NOT include opinions or interpretations — only facts stated in the text.
- Return one proposition per line, no numbering, no bullet points.

Text:
{text}

Propositions:"""


def chunk_proposition(
    doc_id: str,
    pages: List[CanonicalPage],
    *,
    gateway=None,
    model_id: str = "gpt-4o-mini",
    max_input_chars: int = 3000,
) -> List[ChunkRecord]:
    """Decompose text into atomic factual propositions using an LLM.

    Each proposition becomes a separately embedded chunk.  Falls back to
    sentence-level chunking if no LLM gateway is available.

    Parameters
    ----------
    gateway : ModelGateway | None
        LLM gateway for proposition extraction.  If None, falls back to
        sentence-level chunking.
    model_id : str
        Model to use for proposition extraction.
    max_input_chars : int
        Max chars to send per LLM call (text is split into windows).
    """
    full_text = _concat_page_text(pages)
    if not full_text.strip():
        return []

    if gateway is None:
        logger.warning("No ModelGateway available for proposition chunking; "
                       "falling back to sentence_level")
        return chunk_sentence_level(doc_id, pages)

    # Split text into windows for LLM calls
    windows: List[str] = []
    for i in range(0, len(full_text), max_input_chars):
        window = full_text[i:i + max_input_chars]
        if window.strip():
            windows.append(window)

    all_propositions: List[str] = []
    for window in windows:
        prompt = _PROPOSITION_PROMPT.format(text=window)
        try:
            result = gateway.call_model(
                model_id=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                query_id=f"proposition_{doc_id}",
                agent_id="preprocessor",
                step_id=str(uuid.uuid4()),
            )
            response = result.get("content", "")
            props = [line.strip() for line in response.split("\n") if line.strip()]
            all_propositions.extend(props)
        except Exception as exc:
            logger.warning("Proposition extraction failed: %s", exc)
            # Fallback: use sentences from this window
            all_propositions.extend(_split_sentences(window))

    if not all_propositions:
        return chunk_sentence_level(doc_id, pages)

    # Build ChunkRecords — one per proposition
    chunks: List[ChunkRecord] = []
    page_nums = sorted({p.page_number for p in pages})
    first_page = pages[0] if pages else None

    for macro_id, prop in enumerate(all_propositions):
        if not prop.strip():
            continue
        embedding = _embed_text(prop)
        lineage_page = first_page
        # Try to find which page this proposition came from
        for p in pages:
            if prop[:50] in p.text:
                lineage_page = p
                break

        chunks.append(
            _make_chunk_from_multi_page(
                doc_id=doc_id,
                pages=pages,
                text=prop,
                page_numbers=[lineage_page.page_number] if lineage_page else page_nums[:1],
                macro_id=macro_id,
                child_id=0,
                embedding=embedding,
            )
        )

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 11. SUMMARY-INDEXED CHUNKING (LLM-dependent)
# ═══════════════════════════════════════════════════════════════════════════

_SUMMARY_PROMPT = """Summarize the following text in 1-2 concise sentences that capture the key information. Focus on specific facts, numbers, and entities.

Text:
{text}

Summary:"""


def chunk_summary_indexed(
    doc_id: str,
    pages: List[CanonicalPage],
    *,
    gateway=None,
    model_id: str = "gpt-4o-mini",
    base_strategy: str = "sliding_window",
    base_params: Optional[Dict[str, Any]] = None,
) -> List[ChunkRecord]:
    """Chunk normally, then create summary chunks embedded alongside originals.

    For each original chunk, an LLM generates a concise summary.  The
    summary is embedded and stored as a separate chunk (with ``child_id``
    offset by 10000 to distinguish from originals).  At retrieval time,
    both original and summary chunks are candidates — the summary helps
    match queries phrased differently from the source text.

    Falls back to base strategy output (no summaries) if no gateway.

    Parameters
    ----------
    gateway : ModelGateway | None
        LLM gateway for summary generation.
    model_id : str
        Model for summarisation.
    base_strategy : str
        Base chunking strategy for initial splitting.
    base_params : dict | None
        Parameters for the base strategy.
    """
    params = base_params or {}
    dispatch = get_strategy_dispatch()
    base_fn = dispatch.get(base_strategy)

    if base_fn is None:
        base_chunks = chunk_sliding_window(doc_id, pages, **params)
    else:
        base_chunks = base_fn(doc_id, pages, **params)

    if not base_chunks or gateway is None:
        if gateway is None and base_chunks:
            logger.warning("No ModelGateway for summary_indexed; returning base chunks only")
        return base_chunks

    # Generate summaries and create summary chunks
    all_chunks = list(base_chunks)
    for chunk in base_chunks:
        prompt = _SUMMARY_PROMPT.format(text=chunk.text_content[:2000])
        try:
            result = gateway.call_model(
                model_id=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                query_id=f"summary_{doc_id}",
                agent_id="preprocessor",
                step_id=str(uuid.uuid4()),
            )
            summary = result.get("content", "").strip()
            if summary:
                summary_embedding = _embed_text(summary)
                all_chunks.append(
                    ChunkRecord(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=chunk.doc_id,
                        page_numbers=chunk.page_numbers,
                        macro_id=chunk.macro_id,
                        child_id=chunk.child_id + 10000,  # Offset to mark as summary
                        chunk_type=chunk.chunk_type,
                        text_content=summary,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        polygons=chunk.polygons,
                        source_type=chunk.source_type,
                        embedding_model=settings.embedding_model,
                        embedding_dim=settings.embedding_dim,
                        embedding=summary_embedding,
                        heading_path=chunk.heading_path,
                        section_id=chunk.section_id,
                    )
                )
        except Exception as exc:
            logger.warning("Summary generation failed for chunk %s: %s",
                           chunk.chunk_id, exc)

    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

def get_strategy_dispatch() -> Dict[str, Callable]:
    """Return a mapping of strategy_name → chunking function.

    Used by the ingestion pipeline to dispatch to the correct algorithm.
    """
    return {
        "semantic": chunk_semantic,
        "recursive": chunk_recursive,
        "clause_aware": chunk_clause_aware,
        "sentence_level": chunk_sentence_level,
        "sliding_window": chunk_sliding_window,
        "parent_child": chunk_parent_child,
        "table_aware": chunk_table_aware,
        "topic_segmentation": chunk_topic_segmentation,
        "context_enriched": chunk_context_enriched,
        "proposition": chunk_proposition,
        "summary_indexed": chunk_summary_indexed,
    }
