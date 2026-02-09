from dataclasses import is_dataclass, replace
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from core.contracts import RetrievedChunk
from retrieval import vector_search
from storage.db import get_connection


def hybrid_search(doc_id: str, query: str, top_k: int = 3) -> List[RetrievedChunk]:
    vector_hits = vector_search.search(doc_id, query, top_k=top_k * 3)
    bm25_hits = _bm25_search(doc_id, query, top_k=top_k * 3)
    merged = _rrf_merge(vector_hits, bm25_hits, top_k=top_k)
    return merged


def _bm25_search(doc_id: str, query: str, top_k: int) -> List[RetrievedChunk]:
    rows = _fetch_chunk_rows(doc_id)
    if not rows:
        return []
    corpus = [row[6].lower().split() for row in rows]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results: List[RetrievedChunk] = []
    for idx, score in ranked:
        row = rows[idx]
        hit = vector_search._rows_to_chunks([row])[0]
        results.append(_with_score(hit, float(score)))
    return results


def bm25_heading_anchor(
    doc_id: str, phrases: List[str]
) -> Optional[RetrievedChunk]:
    rows = _fetch_chunk_rows(doc_id)
    if not rows:
        return None
    corpus = [
        f"{row[11] or ''} {row[6] or ''}".lower().split()
        for row in rows
    ]
    query_tokens = " ".join(phrases).lower().split()
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)
    best = max(enumerate(scores), key=lambda x: x[1], default=None)
    if not best or best[1] <= 0:
        return None
    row = rows[best[0]]
    hit = vector_search._rows_to_chunks([row])[0]
    return _with_score(hit, float(best[1]))


def bm25_heading_anchor_candidates(
    doc_id: str, phrases: List[str], top_k: int = 25
) -> List[RetrievedChunk]:
    rows = _fetch_chunk_rows(doc_id)
    if not rows:
        return []
    corpus = [
        f"{row[11] or ''} {row[6] or ''}".lower().split()
        for row in rows
    ]
    query_tokens = " ".join(phrases).lower().split()
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results: List[RetrievedChunk] = []
    for idx, score in ranked:
        if score <= 0:
            continue
        row = rows[idx]
        hit = vector_search._rows_to_chunks([row])[0]
        results.append(_with_score(hit, float(score)))
    return results


def lexical_anchor_candidates(
    doc_id: str, phrases: List[str], top_k: int = 25
) -> List[RetrievedChunk]:
    rows = _fetch_chunk_rows(doc_id)
    if not rows:
        return []
    results: List[RetrievedChunk] = []
    lowered = [phrase.lower() for phrase in phrases]
    for row in rows:
        text = f"{row[11] or ''} {row[6] or ''}".lower()
        if any(phrase in text for phrase in lowered):
            hit = vector_search._rows_to_chunks([row])[0]
            results.append(hit)
            if len(results) >= top_k:
                break
    return results


def _fetch_chunk_rows(doc_id: str) -> List[Tuple]:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id,
                       doc_id,
                       page_numbers,
                       macro_id,
                       child_id,
                       chunk_type,
                       text_content,
                       char_start,
                       char_end,
                       polygons,
                       source_type,
                       heading_path,
                       section_id,
                       0.0 AS score
                FROM chunks
                WHERE doc_id = %s
                """,
                (doc_id,),
            )
            return cursor.fetchall()


def _rrf_merge(
    vector_hits: List[RetrievedChunk],
    bm25_hits: List[RetrievedChunk],
    top_k: int,
    k: int = 60,
) -> List[RetrievedChunk]:
    scores: Dict[str, float] = {}
    by_id: Dict[str, RetrievedChunk] = {}

    for rank, hit in enumerate(vector_hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
        by_id[hit.chunk_id] = hit

    for rank, hit in enumerate(bm25_hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
        if hit.chunk_id not in by_id:
            by_id[hit.chunk_id] = hit

    ranked_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results: List[RetrievedChunk] = []
    for chunk_id, score in ranked_ids:
        hit = by_id[chunk_id]
        results.append(_with_score(hit, float(score)))
    return results


def _with_score(hit, score: float):
    if is_dataclass(hit):
        return replace(hit, score=score)
    if hasattr(hit, "score"):
        hit.score = score
    return hit
