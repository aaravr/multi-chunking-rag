import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi

from storage.db import get_connection


@dataclass(frozen=True)
class BM25Index:
    doc_id: str
    corpus_version: str
    rows: List[Tuple]
    bm25: BM25Okapi


class BM25IndexManager:
    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._cache = {}
        self._cache_dir = cache_dir or Path("storage") / "bm25_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def build_index(self, doc_id: str) -> BM25Index:
        rows = _fetch_chunk_rows(doc_id)
        if not rows:
            raise RuntimeError(f"No chunks found for doc_id={doc_id}")
        corpus_version = _fetch_corpus_version(doc_id)
        corpus = [row[6].lower().split() for row in rows]
        bm25 = BM25Okapi(corpus)
        index = BM25Index(doc_id=doc_id, corpus_version=corpus_version, rows=rows, bm25=bm25)
        self._cache[(doc_id, corpus_version)] = index
        self._save_index(index)
        return index

    def get_or_raise(self, doc_id: str) -> BM25Index:
        corpus_version = _fetch_corpus_version(doc_id)
        key = (doc_id, corpus_version)
        cached = self._cache.get(key)
        if cached:
            return cached
        loaded = self._load_index(doc_id, corpus_version)
        if loaded:
            self._cache[key] = loaded
            return loaded
        # BM25 index must be built outside the query path; do not rebuild here.
        raise RuntimeError(
            "BM25 index missing or stale. "
            f"Build index for doc_id={doc_id} before querying."
        )

    def _cache_path(self, doc_id: str, corpus_version: str) -> Path:
        suffix = hashlib.sha1(corpus_version.encode("utf-8")).hexdigest()[:12]
        return self._cache_dir / f"bm25_{doc_id}_{suffix}.pkl"

    def _save_index(self, index: BM25Index) -> None:
        path = self._cache_path(index.doc_id, index.corpus_version)
        with path.open("wb") as handle:
            pickle.dump(index, handle)

    def _load_index(self, doc_id: str, corpus_version: str) -> Optional[BM25Index]:
        path = self._cache_path(doc_id, corpus_version)
        if not path.exists():
            return None
        with path.open("rb") as handle:
            return pickle.load(handle)


def _fetch_corpus_version(doc_id: str) -> str:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*), MAX(created_at)
                FROM chunks
                WHERE doc_id = %s
                """,
                (doc_id,),
            )
            count, max_created = cursor.fetchone()
    ts = max_created.isoformat() if max_created else "none"
    return f"{count}:{ts}"


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


_BM25_MANAGER = BM25IndexManager()


def warm_bm25_index(doc_id: str) -> None:
    _BM25_MANAGER.build_index(doc_id)


def get_bm25_index(doc_id: str) -> BM25Index:
    return _BM25_MANAGER.get_or_raise(doc_id)
