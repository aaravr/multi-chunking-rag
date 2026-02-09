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

CREATE INDEX IF NOT EXISTS document_facts_doc_id_idx ON document_facts (doc_id);
