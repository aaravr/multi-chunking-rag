"""Shared chunk utilities — DRY helpers used across retrieval, ingestion, and synthesis.

Centralises common predicates and transformations so they live in one place.
"""

from typing import List

from core.contracts import RetrievedChunk
from core.enums import ChunkType


def is_table_chunk(chunk: RetrievedChunk) -> bool:
    """Return True if *chunk* represents a table (by type or [TABLE] prefix)."""
    return chunk.chunk_type == ChunkType.TABLE or chunk.text_content.lstrip().startswith("[TABLE]")


def filter_narrative_chunks(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """Return only non-table chunks."""
    return [c for c in chunks if not is_table_chunk(c)]


def merge_page_lists(page_lists: List[List[int]]) -> List[int]:
    """Merge multiple page-number lists into a sorted, deduplicated list."""
    return sorted({page for pages in page_lists for page in pages})


def format_sources(chunks: List[RetrievedChunk]) -> str:
    """Format chunks as numbered [C#] sources for LLM prompts."""
    return "\n".join(f"[C{idx}] {chunk.text_content}" for idx, chunk in enumerate(chunks, start=1))
