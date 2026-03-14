from typing import Dict, List

from storage.db import get_connection

REQUIRED_SCHEMA: Dict[str, List[str]] = {
    # ── Core PoC Tables ──────────────────────────────────────────────
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
        # Lineage fields — required by §2.1 (Deterministic Lineage invariant)
        "heading_path",
        "section_id",
        # Enterprise classification fields — added by migration 004
        "document_type",
        "classification_label",
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
        # Enterprise fields — added by migration 004
        "document_type",
        "classification_label",
        "updated_at",
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
    # ── Enterprise Tables (migration 004) ────────────────────────────
    "audit_log": [
        "log_id",
        "agent_id",
        "event_type",
        "model_id",
        "full_prompt",
        "full_response",
        "input_tokens",
        "output_tokens",
        "temperature",
        "timestamp",
    ],
    # ── Feedback Loop Tables (migration 009) — canonical path ────────
    "prediction_traces": [
        "trace_id",
        "query_id",
        "boundary_client",
        "boundary_division",
        "boundary_jurisdiction",
        "created_at",
    ],
    "feedback_events": [
        "feedback_id",
        "boundary_client",
        "boundary_division",
        "boundary_jurisdiction",
        "rating",
        "created_at",
    ],
    "feedback_attributions": [
        "attribution_id",
        "feedback_id",
        "impacted_layers",
        "attribution_method",
    ],
    "retraining_jobs": [
        "job_id",
        "layer",
        "boundary_client",
        "trigger_type",
        "status",
    ],
    "model_candidates": [
        "candidate_id",
        "layer",
        "boundary_client",
        "stage",
    ],
}


# Unique constraints that MUST exist (table → list of column-sets).
# Each entry is a frozenset of column names that must form a unique constraint.
REQUIRED_UNIQUE_CONSTRAINTS: Dict[str, List[frozenset]] = {
    "documents": [frozenset({"sha256"})],
    "chunks": [frozenset({"doc_id", "macro_id", "child_id"})],
}

# Required indexes (table → list of column-sets).
REQUIRED_INDEXES: Dict[str, List[frozenset]] = {
    "prediction_traces": [frozenset({"query_id"})],
    "feedback_events": [frozenset({"boundary_client"})],
}


def check_schema_contract() -> None:
    """Validate the database schema against the contract.

    Checks:
    1. All required columns exist in each table
    2. Required unique constraints exist
    3. Required indexes exist
    """
    errors: List[str] = []
    with get_connection() as conn:
        with conn.cursor() as cursor:
            # 1. Column presence
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
                    errors.append(
                        f"Missing columns in {table}: {', '.join(missing_cols)}"
                    )

            # 2. Unique constraints
            for table, constraint_sets in REQUIRED_UNIQUE_CONSTRAINTS.items():
                cursor.execute(
                    """
                    SELECT
                        tc.constraint_name,
                        array_agg(kcu.column_name ORDER BY kcu.ordinal_position)
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    WHERE tc.table_schema = 'public'
                      AND tc.table_name = %s
                      AND tc.constraint_type IN ('UNIQUE', 'PRIMARY KEY')
                    GROUP BY tc.constraint_name
                    """,
                    (table,),
                )
                existing_constraints = [
                    frozenset(row[1]) for row in cursor.fetchall()
                ]
                for required_cols in constraint_sets:
                    if required_cols not in existing_constraints:
                        errors.append(
                            f"Missing UNIQUE constraint on {table}"
                            f"({', '.join(sorted(required_cols))})"
                        )

            # 3. Indexes
            for table, index_sets in REQUIRED_INDEXES.items():
                cursor.execute(
                    """
                    SELECT indexdef
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                      AND tablename = %s
                    """,
                    (table,),
                )
                index_defs = [row[0].lower() for row in cursor.fetchall()]
                for required_cols in index_sets:
                    # Check if any index covers the required columns
                    found = False
                    for idx_def in index_defs:
                        if all(col in idx_def for col in required_cols):
                            found = True
                            break
                    if not found:
                        errors.append(
                            f"Missing index on {table}"
                            f"({', '.join(sorted(required_cols))})"
                        )

    if errors:
        details = "; ".join(errors)
        raise RuntimeError(
            f"Schema contract check failed. {details}. "
            "Remediation: run `python storage/setup_db.py` "
            "to apply schema and migrations."
        )
