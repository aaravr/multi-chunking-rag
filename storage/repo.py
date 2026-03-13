from typing import Iterable, List, Optional

from psycopg2.extras import Json, execute_values
from pgvector.psycopg2 import register_vector

from core.contracts import ChunkRecord, DocumentFact, DocumentRecord, PageRecord, TriageMetrics


def insert_document(conn, document: DocumentRecord) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO documents (doc_id, filename, sha256, page_count, document_type, classification_label)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO NOTHING
            """,
            (document.doc_id, document.filename, document.sha256, document.page_count,
             document.document_type, document.classification_label),
        )


def fetch_document_by_sha(conn, sha256: str) -> Optional[DocumentRecord]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT doc_id, filename, sha256, page_count
            FROM documents
            WHERE sha256 = %s
            """,
            (sha256,),
        )
        row = cursor.fetchone()
    if not row:
        return None
    return DocumentRecord(
        doc_id=str(row[0]),
        filename=row[1],
        sha256=row[2],
        page_count=row[3],
    )


def fetch_documents(conn) -> List[DocumentRecord]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT doc_id, filename, sha256, page_count
            FROM documents
            ORDER BY created_at DESC
            """
        )
        rows = cursor.fetchall()
    return [
        DocumentRecord(
            doc_id=str(row[0]),
            filename=row[1],
            sha256=row[2],
            page_count=row[3],
        )
        for row in rows
    ]


def insert_pages(conn, pages: Iterable[PageRecord]) -> None:
    rows = [
        (
            page.doc_id,
            page.page_number,
            Json(page.triage_metrics.__dict__),
            page.triage_decision,
            page.reason_codes,
            page.di_json_path,
        )
        for page in pages
    ]
    if not rows:
        return
    with conn.cursor() as cursor:
        execute_values(
            cursor,
            """
            INSERT INTO pages (
                doc_id,
                page_number,
                triage_metrics,
                triage_decision,
                reason_codes,
                di_json_path
            )
            VALUES %s
            ON CONFLICT (doc_id, page_number) DO UPDATE
            SET triage_metrics = EXCLUDED.triage_metrics,
                triage_decision = EXCLUDED.triage_decision,
                reason_codes = EXCLUDED.reason_codes,
                di_json_path = EXCLUDED.di_json_path
            """,
            rows,
        )


def fetch_pages(conn, doc_id: str) -> List[PageRecord]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT doc_id, page_number, triage_metrics, triage_decision,
                   reason_codes, di_json_path
            FROM pages
            WHERE doc_id = %s
            ORDER BY page_number
            """,
            (doc_id,),
        )
        rows = cursor.fetchall()
    pages: List[PageRecord] = []
    for row in rows:
        metrics_dict = row[2] or {}
        metrics = TriageMetrics(
            text_length=int(metrics_dict.get("text_length", 0)),
            text_density=float(metrics_dict.get("text_density", 0.0)),
            image_coverage_ratio=float(metrics_dict.get("image_coverage_ratio", 0.0)),
            layout_complexity_score=float(metrics_dict.get("layout_complexity_score", 0.0)),
        )
        pages.append(
            PageRecord(
                doc_id=str(row[0]),
                page_number=int(row[1]),
                triage_metrics=metrics,
                triage_decision=row[3],
                reason_codes=list(row[4] or []),
                di_json_path=row[5],
            )
        )
    return pages


def count_chunks(conn, doc_id: str) -> int:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT COUNT(*) FROM chunks WHERE doc_id = %s
            """,
            (doc_id,),
        )
        row = cursor.fetchone()
    return int(row[0] if row else 0)


def insert_chunks(conn, chunks: Iterable[ChunkRecord]) -> None:
    register_vector(conn)
    rows: List[tuple] = [
        (
            chunk.chunk_id,
            chunk.doc_id,
            chunk.page_numbers,
            chunk.macro_id,
            chunk.child_id,
            chunk.chunk_type,
            chunk.text_content,
            chunk.char_start,
            chunk.char_end,
            Json(chunk.polygons),
            chunk.heading_path,
            chunk.section_id,
            chunk.source_type,
            chunk.embedding,
            chunk.embedding_model,
            chunk.embedding_dim,
            chunk.document_type,
            chunk.classification_label,
        )
        for chunk in chunks
    ]
    if not rows:
        return
    with conn.cursor() as cursor:
        execute_values(
            cursor,
            """
            INSERT INTO chunks (
                chunk_id,
                doc_id,
                page_numbers,
                macro_id,
                child_id,
                chunk_type,
                text_content,
                char_start,
                char_end,
                polygons,
                heading_path,
                section_id,
                source_type,
                embedding,
                embedding_model,
                embedding_dim,
                document_type,
                classification_label
            )
            VALUES %s
            ON CONFLICT (doc_id, macro_id, child_id) DO NOTHING
            """,
            rows,
        )


def upsert_document_facts(conn, facts: Iterable[DocumentFact]) -> None:
    rows = [
        (
            fact.doc_id,
            fact.fact_name,
            fact.value,
            fact.status,
            fact.confidence,
            fact.source_chunk_id,
            fact.page_numbers,
            Json(fact.polygons),
            fact.evidence_excerpt,
        )
        for fact in facts
    ]
    if not rows:
        return
    with conn.cursor() as cursor:
        execute_values(
            cursor,
            """
            INSERT INTO document_facts (
                doc_id,
                fact_name,
                value,
                status,
                confidence,
                source_chunk_id,
                page_numbers,
                polygons,
                evidence_excerpt
            )
            VALUES %s
            ON CONFLICT (doc_id, fact_name) DO UPDATE
            SET value = EXCLUDED.value,
                status = EXCLUDED.status,
                confidence = EXCLUDED.confidence,
                source_chunk_id = EXCLUDED.source_chunk_id,
                page_numbers = EXCLUDED.page_numbers,
                polygons = EXCLUDED.polygons,
                evidence_excerpt = EXCLUDED.evidence_excerpt
            """,
            rows,
        )


def update_document_classification(
    conn, doc_id: str, document_type: str, classification_label: str
) -> None:
    """Update the classification fields on a document after classifier agent runs."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE documents
            SET document_type = %s, classification_label = %s, updated_at = now()
            WHERE doc_id = %s
            """,
            (document_type, classification_label, doc_id),
        )


def update_chunks_classification(
    conn, doc_id: str, document_type: str, classification_label: str
) -> None:
    """Propagate classification to all chunks for a document."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE chunks
            SET document_type = %s, classification_label = %s
            WHERE doc_id = %s
            """,
            (document_type, classification_label, doc_id),
        )


def insert_classification_embedding(
    conn,
    embedding_id: str,
    document_type: str,
    classification_label: str,
    embedding: list,
    source_doc_id: Optional[str] = None,
) -> None:
    """Insert a classification embedding into pgvector-backed storage."""
    register_vector(conn)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO classification_embeddings
                (embedding_id, document_type, classification_label, embedding, source_doc_id)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (embedding_id) DO NOTHING
            """,
            (embedding_id, document_type, classification_label, embedding, source_doc_id),
        )


def search_classification_embeddings(
    conn,
    query_embedding: list,
    top_k: int = 1,
    threshold: float = 0.85,
) -> List[dict]:
    """Find most similar classification embeddings using pgvector cosine similarity.

    Returns list of dicts with: embedding_id, document_type, classification_label, similarity.
    """
    register_vector(conn)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT embedding_id, document_type, classification_label,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM classification_embeddings
            WHERE 1 - (embedding <=> %s::vector) >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, threshold, query_embedding, top_k),
        )
        rows = cursor.fetchall()
    return [
        {
            "embedding_id": str(row[0]),
            "document_type": row[1],
            "classification_label": row[2],
            "similarity": float(row[3]),
        }
        for row in rows
    ]


def fetch_all_classification_embeddings(conn) -> List[dict]:
    """Fetch all classification embeddings for SGD retraining.

    Returns list of dicts with: embedding_id, document_type, classification_label, embedding.
    """
    register_vector(conn)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT embedding_id, document_type, classification_label, embedding
            FROM classification_embeddings
            ORDER BY created_at
            """
        )
        rows = cursor.fetchall()
    return [
        {
            "embedding_id": str(row[0]),
            "document_type": row[1],
            "classification_label": row[2],
            "embedding": [float(x) for x in row[3]],
        }
        for row in rows
    ]


def count_classification_embeddings(conn) -> int:
    """Count total classification embeddings in pgvector store."""
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM classification_embeddings")
        row = cursor.fetchone()
    return int(row[0] if row else 0)


def fetch_document_fact(conn, doc_id: str, fact_name: str) -> Optional[DocumentFact]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT doc_id, fact_name, value, status, confidence, source_chunk_id,
                   page_numbers, polygons, evidence_excerpt
            FROM document_facts
            WHERE doc_id = %s AND fact_name = %s
            """,
            (doc_id, fact_name),
        )
        row = cursor.fetchone()
    if not row:
        return None
    return DocumentFact(
        doc_id=str(row[0]),
        fact_name=row[1],
        value=row[2],
        status=row[3],
        confidence=float(row[4] or 0.0),
        source_chunk_id=str(row[5]) if row[5] else None,
        page_numbers=list(row[6] or []),
        polygons=list(row[7] or []),
        evidence_excerpt=row[8],
    )
