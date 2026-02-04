import os

import fitz
import pytest

from core import config
from ingestion.ingest_pipeline import ingest_and_chunk
from storage.db import get_connection
from storage.schema_contract import check_schema_contract
from storage.setup_db import run_setup


@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL not set; skipping ingestion smoke test.",
)
def test_ingestion_smoke(tmp_path):
    test_db = os.environ["TEST_DATABASE_URL"]
    config.settings.database_url = test_db
    config.settings.disable_di = True
    config.settings.data_dir = str(tmp_path / "data")
    run_setup()
    check_schema_contract()

    pdf_path = tmp_path / "tiny.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Items of note: FDIC special assessment.")
    doc.save(str(pdf_path))
    doc.close()

    doc_id = ingest_and_chunk(str(pdf_path), force_reprocess=True)
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM chunks WHERE doc_id = %s AND chunk_type IS NOT NULL",
                (doc_id,),
            )
            count = cursor.fetchone()[0]
    assert count > 0
