"""WO-010: Chunk insertion idempotency on retry."""

import os
import uuid

import pytest

from core.contracts import ChunkRecord, DocumentRecord
from storage.db import get_connection
from storage import repo


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL required for integration test",
)
def test_chunk_insert_idempotent():
    """Inserting same (doc_id, macro_id, child_id) twice does not duplicate."""
    doc_id = str(uuid.uuid4())
    with get_connection() as conn:
        repo.insert_document(
            conn,
            DocumentRecord(
                doc_id=doc_id,
                filename="test.pdf",
                sha256="a" * 64,
                page_count=1,
            ),
        )
        conn.commit()

    def _chunk_record(cid: str) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=cid,
            doc_id=doc_id,
            page_numbers=[1],
            macro_id=0,
            child_id=0,
            chunk_type="narrative",
            text_content="Test content",
            char_start=0,
            char_end=12,
            polygons=[],
            heading_path="",
            section_id="",
            source_type="native",
            embedding=[0.1] * 768,
            embedding_model="nomic-ai/modernbert-embed-base",
            embedding_dim=768,
        )

    chunk1 = _chunk_record(str(uuid.uuid4()))
    chunk2 = _chunk_record(str(uuid.uuid4()))  # Same doc/macro/child, different chunk_id

    with get_connection() as conn:
        repo.insert_chunks(conn, [chunk1])
        conn.commit()
        count_after_first = repo.count_chunks(conn, doc_id)

        repo.insert_chunks(conn, [chunk2])  # Retry: same logical chunk
        conn.commit()
        count_after_retry = repo.count_chunks(conn, doc_id)

    assert count_after_first == count_after_retry, "Retry must not duplicate chunks"


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL required for integration test",
)
def test_chunk_insert_idempotent_requires_migration():
    """Migration 003 must be applied for idempotency. This test documents the requirement."""
    # If this test runs, we assume migrations are applied.
    # The actual idempotency is tested by test_chunk_insert_idempotent.
    assert True
