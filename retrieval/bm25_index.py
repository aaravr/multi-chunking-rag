"""BM25 index manager with JSON-backed cache (no pickle; SPEC §13, WO-010)."""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from storage.db import get_connection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BM25Index:
    doc_id: str
    corpus_version: str
    rows: List[Tuple]
    bm25: BM25Okapi


def _row_to_json_compatible(row: Tuple) -> List[Any]:
    """Convert DB row tuple to JSON-serializable list."""
    result: List[Any] = []
    for v in row:
        if isinstance(v, list):
            result.append(v)
        elif isinstance(v, (dict, str, int, float, bool)) or v is None:
            result.append(v)
        else:
            result.append(str(v))
    return result


def _json_to_row(lst: List[Any]) -> Tuple:
    """Convert JSON list back to row tuple with correct types."""
    chunk_id, doc_id, page_numbers, macro_id, child_id, chunk_type, text_content = lst[:7]
    char_start, char_end, polygons, source_type, heading_path, section_id, score = lst[7:14]
    return (
        str(chunk_id),
        str(doc_id),
        list(page_numbers) if page_numbers else [],
        int(macro_id),
        int(child_id),
        str(chunk_type) if chunk_type else "narrative",
        str(text_content) if text_content else "",
        int(char_start),
        int(char_end),
        polygons or [],
        str(source_type) if source_type else "native",
        str(heading_path) if heading_path else "",
        str(section_id) if section_id else "",
        float(score) if score is not None else 0.0,
    )


class BM25IndexManager:
    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._cache: dict = {}
        self._cache_dir = cache_dir or Path("storage") / "bm25_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def build_index(self, doc_id: str) -> BM25Index:
        rows = _fetch_chunk_rows(doc_id)
        if not rows:
            raise RuntimeError(f"No chunks found for doc_id={doc_id}")
        corpus_version = _fetch_corpus_version(doc_id)
        corpus = [row[6].lower().split() for row in rows]
        bm25 = BM25Okapi(corpus)
        index = BM25Index(
            doc_id=doc_id,
            corpus_version=corpus_version,
            rows=rows,
            bm25=bm25,
        )
        self._cache[(doc_id, corpus_version)] = index
        self._save_index(index)
        return index

    def get_or_raise(self, doc_id: str) -> BM25Index:
        corpus_version = _fetch_corpus_version(doc_id)
        key = (doc_id, corpus_version)
        if key in self._cache:
            return self._cache[key]
        loaded = self._load_index(doc_id, corpus_version)
        if loaded:
            self._cache[key] = loaded
            return loaded
        raise RuntimeError(
            "BM25 index missing or stale. "
            f"Build index for doc_id={doc_id} before querying."
        )

    def _cache_path(self, doc_id: str, corpus_version: str) -> Path:
        suffix = hashlib.sha1(corpus_version.encode("utf-8")).hexdigest()[:12]
        return self._cache_dir / f"bm25_{doc_id}_{suffix}.json"

    def _save_index(self, index: BM25Index) -> None:
        path = self._cache_path(index.doc_id, index.corpus_version)
        corpus = [row[6].lower().split() for row in index.rows]
        payload = {
            "doc_id": index.doc_id,
            "corpus_version": index.corpus_version,
            "rows": [_row_to_json_compatible(r) for r in index.rows],
            "corpus": corpus,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def _load_index(self, doc_id: str, corpus_version: str) -> Optional[BM25Index]:
        path = self._cache_path(doc_id, corpus_version)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("BM25 cache corrupted or unreadable: %s", exc)
            return None
        rows = [_json_to_row(r) for r in payload["rows"]]
        corpus = payload.get("corpus")
        if not corpus:
            corpus = [r[6].lower().split() for r in rows]
        bm25 = BM25Okapi(corpus)
        return BM25Index(
            doc_id=payload["doc_id"],
            corpus_version=payload["corpus_version"],
            rows=rows,
            bm25=bm25,
        )


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
