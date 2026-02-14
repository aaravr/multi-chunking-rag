"""Module-level singleton for embedding model to avoid per-query reloads (SPEC §13, WO-010)."""

from typing import Optional

_MODEL: Optional["ModernBERTEmbedder"] = None


def get_embedding_model(max_length: int = 8192) -> "ModernBERTEmbedder":
    """Return the shared ModernBERT embedder instance. Loads once per process."""
    global _MODEL
    if _MODEL is None:
        from embedding.modernbert import ModernBERTEmbedder

        _MODEL = ModernBERTEmbedder(max_length=max_length)
    return _MODEL


def _reset_for_testing() -> None:
    """Reset the singleton. For testing only."""
    global _MODEL
    _MODEL = None
