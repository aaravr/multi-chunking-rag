from typing import Iterable, List, Optional

from psycopg2.extras import Json
from pgvector.psycopg2 import register_vector

from core.contracts import ChunkRecord, DocumentRecord, PageRecord, TriageMetrics


def insert_document(conn, document: DocumentRecord) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO documents (doc_id, filename, sha256, page_count)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (doc_id) DO NOTHING
            """,
            (document.doc_id, document.filename, document.sha256, document.page_count),
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
        cursor.executemany(
            """
            INSERT INTO pages (
                doc_id,
                page_number,
                triage_metrics,
                triage_decision,
                reason_codes,
                di_json_path
            )
            VALUES (%s, %s, %s, %s, %s, %s)
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
        )
        for chunk in chunks
    ]
    if not rows:
        return
    with conn.cursor() as cursor:
        cursor.executemany(
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
                embedding_dim
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
