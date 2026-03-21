-- Migration 005: Classification memory table (MASTER_PROMPT §4.8)
-- Persists learned classification patterns for the self-learning classifier agent.

CREATE TABLE IF NOT EXISTS classification_memory (
    pattern_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_type TEXT NOT NULL,
    classification_label TEXT NOT NULL,
    filename_pattern TEXT,
    title_keywords TEXT[] NOT NULL DEFAULT '{}',
    structural_signals JSONB NOT NULL DEFAULT '{}',
    success_count INT NOT NULL DEFAULT 0,
    total_count INT NOT NULL DEFAULT 1,
    last_used TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS classification_memory_doc_type_idx
    ON classification_memory (document_type);
CREATE INDEX IF NOT EXISTS classification_memory_label_idx
    ON classification_memory (classification_label);
CREATE INDEX IF NOT EXISTS classification_memory_filename_idx
    ON classification_memory (filename_pattern);
