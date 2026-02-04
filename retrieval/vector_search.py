from typing import List, Optional

from pgvector.psycopg2 import register_vector

from core.contracts import RetrievedChunk
from embedding.modernbert import ModernBERTEmbedder
from storage.db import get_connection


def search(
    doc_id: str,
    query: str,
    top_k: int = 3,
) -> List[RetrievedChunk]:
    embedder = ModernBERTEmbedder()
    query_embedding = embedder.embed_text(query)
    with get_connection() as conn:
        register_vector(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id,
                       doc_id,
                       page_numbers,
                       macro_id,
                       chunk_type,
                       text_content,
                       char_start,
                       char_end,
                       polygons,
                       source_type,
                       heading_path,
                       section_id,
                       1 - (embedding <=> %s::vector) AS score
                FROM chunks
                WHERE doc_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, doc_id, query_embedding, top_k),
            )
            rows = cursor.fetchall()
    results: List[RetrievedChunk] = []
    for row in rows:
        results.append(
            RetrievedChunk(
                chunk_id=str(row[0]),
                doc_id=str(row[1]),
                page_numbers=list(row[2] or []),
                macro_id=int(row[3] or 0),
                chunk_type=row[4] or "narrative",
                text_content=row[5],
                char_start=int(row[6]),
                char_end=int(row[7]),
                polygons=list(row[8] or []),
                source_type=row[9],
                heading_path=row[10],
                section_id=row[11],
                score=float(row[12]),
            )
        )
    return results


def search_on_pages(
    doc_id: str,
    query: str,
    page_numbers: List[int],
    top_k: int = 3,
) -> List[RetrievedChunk]:
    if not page_numbers:
        return []
    embedder = ModernBERTEmbedder()
    query_embedding = embedder.embed_text(query)
    with get_connection() as conn:
        register_vector(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id,
                       doc_id,
                       page_numbers,
                       macro_id,
                       chunk_type,
                       text_content,
                       char_start,
                       char_end,
                       polygons,
                       source_type,
                       heading_path,
                       section_id,
                       1 - (embedding <=> %s::vector) AS score
                FROM chunks
                WHERE doc_id = %s
                  AND page_numbers && %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, doc_id, page_numbers, query_embedding, top_k),
            )
            rows = cursor.fetchall()
    return _rows_to_chunks(rows)


def fetch_by_section(
    doc_id: str,
    heading_path: Optional[str],
    section_id: Optional[str],
) -> List[RetrievedChunk]:
    if not heading_path and not section_id:
        return []
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id,
                       doc_id,
                       page_numbers,
                       macro_id,
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
                  AND (heading_path = %s OR section_id = %s)
                ORDER BY page_numbers[1] NULLS LAST, char_start
                """,
                (doc_id, heading_path, section_id),
            )
            rows = cursor.fetchall()
    return _rows_to_chunks(rows)


def fetch_by_page_window(
    doc_id: str, anchor_pages: List[int], window: int = 2
) -> List[RetrievedChunk]:
    if not anchor_pages:
        return []
    start = max(min(anchor_pages) - window, 1)
    end = max(anchor_pages) + window
    pages = list(range(start, end + 1))
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id,
                       doc_id,
                       page_numbers,
                       macro_id,
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
                  AND page_numbers && %s
                ORDER BY page_numbers[1] NULLS LAST, char_start
                """,
                (doc_id, pages),
            )
            rows = cursor.fetchall()
    return _rows_to_chunks(rows)


def fetch_by_macro_id(doc_id: str, macro_id: int) -> List[RetrievedChunk]:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id,
                       doc_id,
                       page_numbers,
                       macro_id,
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
                  AND macro_id = %s
                ORDER BY page_numbers[1] NULLS LAST, char_start
                """,
                (doc_id, macro_id),
            )
            rows = cursor.fetchall()
    return _rows_to_chunks(rows)


def _rows_to_chunks(rows) -> List[RetrievedChunk]:
    results: List[RetrievedChunk] = []
    for row in rows:
        results.append(
            RetrievedChunk(
                chunk_id=str(row[0]),
                doc_id=str(row[1]),
                page_numbers=list(row[2] or []),
                macro_id=int(row[3] or 0),
                chunk_type=row[4] or "narrative",
                text_content=row[5],
                char_start=int(row[6]),
                char_end=int(row[7]),
                polygons=list(row[8] or []),
                source_type=row[9],
                heading_path=row[10],
                section_id=row[11],
                score=float(row[12]),
            )
        )
    return results
