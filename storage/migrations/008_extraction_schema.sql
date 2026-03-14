-- Migration 008: Schema-Driven Extraction, Transformer, MCP Reference Data
-- Implements §10 of MASTER_PROMPT: Extraction pipeline tables

-- ── Extraction Schemas ──────────────────────────────────────────────
-- Stores declarative field extraction definitions per document type.
CREATE TABLE IF NOT EXISTS extraction_schemas (
    schema_id       TEXT PRIMARY KEY,
    schema_name     TEXT NOT NULL,
    document_type   TEXT NOT NULL,
    classification_label TEXT NOT NULL,
    version         TEXT NOT NULL DEFAULT '1.0',
    description     TEXT DEFAULT '',
    fields_json     JSONB NOT NULL DEFAULT '[]'::jsonb,  -- List[FieldDefinition] as JSON
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(document_type, classification_label, version)
);

CREATE INDEX IF NOT EXISTS idx_extraction_schemas_doctype
    ON extraction_schemas(document_type, classification_label);

-- ── Extracted Fields ────────────────────────────────────────────────
-- Stores per-field extraction results with confidence and provenance.
CREATE TABLE IF NOT EXISTS extracted_fields (
    extraction_id       TEXT NOT NULL DEFAULT gen_random_uuid()::text,
    doc_id              UUID NOT NULL,
    schema_id           TEXT NOT NULL,
    query_id            TEXT DEFAULT '',
    field_name          TEXT NOT NULL,
    raw_value           TEXT DEFAULT '',
    normalized_value    TEXT DEFAULT '',
    confidence          REAL DEFAULT 0.0,
    source_chunk_ids    TEXT[] DEFAULT '{}',
    page_numbers        INTEGER[] DEFAULT '{}',
    extraction_method   TEXT DEFAULT 'llm'
        CHECK (extraction_method IN ('llm', 'regex', 'deterministic', 'mcp_normalized')),
    validation_passed   BOOLEAN DEFAULT TRUE,
    validation_errors   TEXT[] DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (extraction_id)
);

CREATE INDEX IF NOT EXISTS idx_extracted_fields_doc
    ON extracted_fields(doc_id);
CREATE INDEX IF NOT EXISTS idx_extracted_fields_schema
    ON extracted_fields(schema_id);
CREATE INDEX IF NOT EXISTS idx_extracted_fields_field
    ON extracted_fields(doc_id, field_name);

-- ── Extraction Results (aggregate per doc) ──────────────────────────
CREATE TABLE IF NOT EXISTS extraction_results (
    result_id           TEXT NOT NULL DEFAULT gen_random_uuid()::text,
    doc_id              UUID NOT NULL,
    schema_id           TEXT NOT NULL,
    query_id            TEXT DEFAULT '',
    overall_confidence  REAL DEFAULT 0.0,
    extraction_model    TEXT DEFAULT '',
    input_tokens        INTEGER DEFAULT 0,
    output_tokens       INTEGER DEFAULT 0,
    warnings            TEXT[] DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (result_id)
);

CREATE INDEX IF NOT EXISTS idx_extraction_results_doc
    ON extraction_results(doc_id);

-- ── MCP Lookup Audit Log ────────────────────────────────────────────
-- Tracks every MCP reference data lookup for auditability.
CREATE TABLE IF NOT EXISTS mcp_lookup_log (
    lookup_id           TEXT NOT NULL DEFAULT gen_random_uuid()::text,
    doc_id              UUID,
    query_id            TEXT DEFAULT '',
    field_name          TEXT NOT NULL,
    lookup_key          TEXT NOT NULL,
    lookup_value        TEXT NOT NULL,
    canonical_value     TEXT DEFAULT '',
    confidence          REAL DEFAULT 0.0,
    matched             BOOLEAN DEFAULT FALSE,
    server_url          TEXT DEFAULT '',
    source              TEXT DEFAULT '',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (lookup_id)
);

CREATE INDEX IF NOT EXISTS idx_mcp_lookup_doc
    ON mcp_lookup_log(doc_id);

-- ── Transformation Rules ────────────────────────────────────────────
-- Stores transformation rules for field normalization.
CREATE TABLE IF NOT EXISTS transformation_rules (
    rule_id             TEXT NOT NULL DEFAULT gen_random_uuid()::text,
    schema_id           TEXT NOT NULL,
    field_name          TEXT NOT NULL,
    transform_type      TEXT NOT NULL
        CHECK (transform_type IN ('mcp_lookup', 'regex_replace', 'date_format', 'currency_convert', 'uppercase', 'lowercase')),
    mcp_server_url      TEXT DEFAULT '',
    mcp_lookup_key      TEXT DEFAULT '',
    regex_pattern       TEXT DEFAULT '',
    regex_replacement   TEXT DEFAULT '',
    date_input_format   TEXT DEFAULT '',
    date_output_format  TEXT DEFAULT '',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (rule_id),
    UNIQUE(schema_id, field_name)
);
