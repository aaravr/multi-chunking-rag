from typing import Dict, List

from storage.db import get_connection

REQUIRED_SCHEMA: Dict[str, List[str]] = {
    "chunks": [
        "chunk_id",
        "doc_id",
        "page_numbers",
        "macro_id",
        "child_id",
        "text_content",
        "char_start",
        "char_end",
        "polygons",
        "source_type",
        "embedding_model",
        "embedding_dim",
        "embedding",
        "chunk_type",
    ],
    "pages": [
        "doc_id",
        "page_number",
        "triage_metrics",
        "triage_decision",
        "reason_codes",
        "di_json_path",
        "created_at",
    ],
    "documents": [
        "doc_id",
        "filename",
        "sha256",
        "page_count",
        "created_at",
    ],
    "document_facts": [
        "doc_id",
        "fact_name",
        "value",
        "status",
        "confidence",
        "source_chunk_id",
        "page_numbers",
        "polygons",
        "evidence_excerpt",
        "created_at",
    ],
}


def check_schema_contract() -> None:
    missing = {}
    with get_connection() as conn:
        with conn.cursor() as cursor:
            for table, columns in REQUIRED_SCHEMA.items():
                cursor.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = %s
                    """,
                    (table,),
                )
                existing = {row[0] for row in cursor.fetchall()}
                missing_cols = [col for col in columns if col not in existing]
                if missing_cols:
                    missing[table] = missing_cols
    if missing:
        details = "; ".join(
            f"{table}: {', '.join(cols)}" for table, cols in missing.items()
        )
        raise RuntimeError(
            "Schema contract check failed. Missing columns: "
            f"{details}. Remediation: run `python storage/setup_db.py` "
            "to apply schema and migrations."
        )
