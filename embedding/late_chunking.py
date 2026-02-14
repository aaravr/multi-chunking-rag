from __future__ import annotations

import re
import uuid
from typing import List, Tuple

import numpy as np

from core.config import settings
from core.contracts import CanonicalPage, CanonicalSpan, ChunkRecord
from embedding.model_registry import get_embedding_model


def late_chunk_embeddings(
    pages: List[CanonicalPage],
    macro_max_tokens: int = 8192,
    macro_overlap_tokens: int = 256,
    child_target_tokens: int = 256,
    progress_cb=None,
) -> List[ChunkRecord]:
    embedder = get_embedding_model(max_length=macro_max_tokens)
    chunks: List[ChunkRecord] = []
    macro_id = 0
    total_macros = 0
    macro_chunks_per_page: List[Tuple[CanonicalPage, List[Tuple[str, int]]]] = []

    for page in pages:
        if not page.text:
            macro_chunks_per_page.append((page, []))
            continue
        macro_chunks = _build_macro_chunks(
            page.text, embedder, macro_max_tokens, macro_overlap_tokens
        )
        macro_chunks_per_page.append((page, macro_chunks))
        total_macros += len(macro_chunks)

    processed_macros = 0

    for page, macro_chunks in macro_chunks_per_page:
        for span in page.spans:
            if not span.is_table:
                continue
            tokenized = embedder.tokenize(span.text)
            token_embeddings = embedder.encode(tokenized)
            pooled = token_embeddings.mean(dim=0).cpu().numpy().astype(np.float32)
            chunks.append(
                ChunkRecord(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=page.doc_id,
                    page_numbers=[span.page_number],
                    macro_id=macro_id,
                    child_id=0,
                    chunk_type="table",
                    text_content=span.text,
                    char_start=span.char_start,
                    char_end=span.char_end,
                    polygons=span.polygons,
                    source_type=span.source_type,
                    embedding_model=settings.embedding_model,
                    embedding_dim=settings.embedding_dim,
                    embedding=pooled.tolist(),
                    heading_path=span.heading_path,
                    section_id=span.section_id,
                )
            )
            macro_id += 1
        if not macro_chunks:
            continue
        for macro_text, base_offset in macro_chunks:
            if progress_cb:
                progress_cb(
                    "embed",
                    processed_macros,
                    total_macros,
                )
            tokenized = embedder.tokenize(macro_text)
            token_embeddings = embedder.encode(tokenized)
            child_spans = _build_child_spans(
                tokenized.offsets,
                child_target_tokens,
            )
            child_id = 0
            for char_start, char_end, token_indices in child_spans:
                if char_end <= char_start:
                    continue
                span_text = macro_text[char_start:char_end]
                if not span_text.strip():
                    continue
                span_embeddings = token_embeddings[token_indices]
                pooled = span_embeddings.mean(dim=0).cpu().numpy().astype(np.float32)
                global_start = base_offset + char_start
                global_end = base_offset + char_end
                polygons, page_numbers, source_type, heading_path, section_id = _collect_span_lineage(
                    page.spans, global_start, global_end
                )
                chunks.append(
                    ChunkRecord(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=page.doc_id,
                        page_numbers=page_numbers,
                        macro_id=macro_id,
                        child_id=child_id,
                        chunk_type=_classify_chunk_type(span_text),
                        text_content=span_text,
                        char_start=global_start,
                        char_end=global_end,
                        polygons=polygons,
                        source_type=source_type,
                        embedding_model=settings.embedding_model,
                        embedding_dim=settings.embedding_dim,
                        embedding=pooled.tolist(),
                        heading_path=heading_path,
                        section_id=section_id,
                    )
                )
                child_id += 1
            macro_id += 1
            processed_macros += 1
            if progress_cb:
                progress_cb(
                    "embed",
                    processed_macros,
                    total_macros,
                )

    return chunks


def _build_macro_chunks(
    text: str,
    embedder,  # ModernBERTEmbedder from model_registry
    macro_max_tokens: int,
    macro_overlap_tokens: int,
) -> List[Tuple[str, int]]:
    offsets = embedder.tokenize_full(text)
    valid_indices = [i for i, (start, end) in enumerate(offsets) if end > start]
    total_tokens = len(valid_indices)
    if total_tokens <= macro_max_tokens:
        return [(text, 0)]

    chunks: List[Tuple[str, int]] = []
    step = max(macro_max_tokens - macro_overlap_tokens, 1)
    for start in range(0, total_tokens, step):
        end = min(start + macro_max_tokens, total_tokens)
        chunk_start = offsets[valid_indices[start]][0]
        chunk_end = offsets[valid_indices[end - 1]][1]
        chunks.append((text[chunk_start:chunk_end], chunk_start))
        if end >= total_tokens:
            break
    return chunks


def _build_child_spans(
    offsets: List[Tuple[int, int]], child_target_tokens: int
) -> List[Tuple[int, int, List[int]]]:
    spans: List[Tuple[int, int, List[int]]] = []
    valid_indices = [i for i, (start, end) in enumerate(offsets) if end > start]
    if not valid_indices:
        return spans
    stride = max(child_target_tokens, 1)
    total = len(valid_indices)
    for start in range(0, total, stride):
        end = min(start + child_target_tokens, total)
        token_indices = valid_indices[start:end]
        char_start = offsets[token_indices[0]][0]
        char_end = offsets[token_indices[-1]][1]
        spans.append((char_start, char_end, token_indices))
        if end >= total:
            break
    return spans


def _collect_span_lineage(
    spans: List[CanonicalSpan], char_start: int, char_end: int
) -> Tuple[List[dict], List[int], str, str, str]:
    polygons: List[dict] = []
    page_numbers: List[int] = []
    source_type = "native"
    heading_path = ""
    section_id = ""
    for span in spans:
        overlap = not (span.char_end <= char_start or span.char_start >= char_end)
        if not overlap:
            continue
        polygons.extend(span.polygons)
        if span.page_number not in page_numbers:
            page_numbers.append(span.page_number)
        source_type = span.source_type
        if not heading_path:
            heading_path = span.heading_path
            section_id = span.section_id
    return polygons, sorted(page_numbers), source_type, heading_path, section_id


def _classify_chunk_type(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return "boilerplate"
    if _looks_like_heading(cleaned):
        return "heading"
    if _looks_like_boilerplate(cleaned):
        return "boilerplate"
    return "narrative"


def _looks_like_heading(text: str) -> bool:
    if text.isupper() and len(text) <= 80:
        return True
    if text.endswith(":") and len(text) <= 80:
        return True
    if text.istitle() and len(text) <= 80:
        return True
    if re.match(r"^\d+(\.\d+)*\s+\S", text) and len(text) <= 100:
        return True
    if re.match(r"^Note\s+\d+", text, re.IGNORECASE):
        return True
    return False


def _looks_like_boilerplate(text: str) -> bool:
    if "ANNUAL REPORT" in text.upper() and len(text) <= 120:
        return True
    if "CONSOLIDATED FINANCIAL STATEMENTS" in text.upper() and len(text) <= 120:
        return True
    return False
