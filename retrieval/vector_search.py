from typing import List

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
                       text_content,
                       char_start,
                       char_end,
                       polygons,
                       source_type,
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
                text_content=row[3],
                char_start=int(row[4]),
                char_end=int(row[5]),
                polygons=list(row[6] or []),
                source_type=row[7],
                score=float(row[8]),
            )
        )
    return results
