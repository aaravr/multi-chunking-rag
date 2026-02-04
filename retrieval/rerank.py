from typing import List

from core.config import settings
from core.contracts import RetrievedChunk

_cross_encoder = None


def rerank(query: str, candidates: List[RetrievedChunk]) -> List[RetrievedChunk]:
    if not candidates:
        return []
    encoder = _get_cross_encoder()
    pairs = [(query, c.text_content) for c in candidates]
    scores = encoder.predict(pairs)
    ranked = sorted(
        zip(candidates, scores), key=lambda item: float(item[1]), reverse=True
    )
    return [item[0] for item in ranked]


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(settings.reranker_model, device="cpu")
    return _cross_encoder
