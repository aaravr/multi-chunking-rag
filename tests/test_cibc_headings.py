import os

import pytest

from ingestion.ingest_pipeline import ingest_and_chunk
from storage.db import get_connection


@pytest.mark.skipif(
    not os.getenv("CIBC_PDF_PATH"),
    reason="CIBC_PDF_PATH not set; skipping heavy ingest test.",
)
def test_cibc_headings_present():
    pdf_path = os.environ["CIBC_PDF_PATH"]
    doc_id = ingest_and_chunk(pdf_path, force_reprocess=False)
    headings = [
        "%Significant events%",
        "%Items of note%",
        "%Note 21%",
        "%Significant legal proceedings%",
    ]
    with get_connection() as conn:
        with conn.cursor() as cursor:
            for heading in headings:
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM chunks
                    WHERE doc_id = %s
                      AND heading_path ILIKE %s
                    """,
                    (doc_id, heading),
                )
                count = cursor.fetchone()[0]
                assert count > 0, f"Missing heading_path for {heading}"
