CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS documents (
    doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    page_count INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pages (
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page_number INT NOT NULL,
    triage_metrics JSONB NOT NULL,
    triage_decision TEXT NOT NULL,
    reason_codes TEXT[] NOT NULL,
    di_json_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, page_number),
    CONSTRAINT pages_triage_decision_check
        CHECK (triage_decision IN ('native_only', 'di_required'))
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page_numbers INT[] NOT NULL,
    macro_id INT NOT NULL,
    child_id INT NOT NULL,
    chunk_type TEXT NOT NULL DEFAULT 'narrative',
    text_content TEXT NOT NULL,
    char_start INT NOT NULL,
    char_end INT NOT NULL,
    polygons JSONB NOT NULL,
    heading_path TEXT NOT NULL,
    section_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dim INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chunks_source_type_check
        CHECK (source_type IN ('di', 'native'))
);

CREATE TABLE IF NOT EXISTS document_facts (
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    fact_name TEXT NOT NULL,
    value TEXT,
    status TEXT NOT NULL,
    confidence FLOAT8 NOT NULL DEFAULT 0.0,
    source_chunk_id UUID,
    page_numbers INT[],
    polygons JSONB,
    evidence_excerpt TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, fact_name)
);

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS heading_path TEXT NOT NULL DEFAULT '';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS section_id TEXT NOT NULL DEFAULT '';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_type TEXT NOT NULL DEFAULT 'narrative';
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chunks_chunk_type_check'
    ) THEN
        ALTER TABLE chunks
            ADD CONSTRAINT chunks_chunk_type_check
            CHECK (chunk_type IN ('narrative', 'table', 'heading', 'boilerplate'));
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS pages_doc_id_idx ON pages (doc_id);
CREATE INDEX IF NOT EXISTS document_facts_doc_id_idx ON document_facts (doc_id);
