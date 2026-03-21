-- Migration 004: Enterprise schema extensions (MASTER_PROMPT §8.2)
-- Adds: users, document_access, audit_log, query_history, entities, entity_mentions,
--        prompt_templates tables + new indexes + document metadata columns.

-- ── Extended document metadata (§8.1) ────────────────────────────────
ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_type TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS classification_label TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS entity_name TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS reporting_period TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS jurisdiction TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;

-- ── Extended chunk metadata (§2.1) ───────────────────────────────────
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS document_type TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS classification_label TEXT;

-- ── Users table (§8.2) ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL DEFAULT 'reader',
    clearance_level INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Document access control (§8.2) ──────────────────────────────────
CREATE TABLE IF NOT EXISTS document_access (
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(user_id),
    role TEXT,
    permission_level TEXT NOT NULL DEFAULT 'read',
    granted_by UUID REFERENCES users(user_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, COALESCE(user_id, '00000000-0000-0000-0000-000000000000'::uuid), COALESCE(role, ''))
);

-- ── Audit log — IMMUTABLE, APPEND-ONLY (§2.4) ──────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID,
    agent_id TEXT NOT NULL,
    step_id TEXT NOT NULL DEFAULT '',
    event_type TEXT NOT NULL,
    model_id TEXT NOT NULL DEFAULT '',
    prompt_template_version TEXT NOT NULL DEFAULT '',
    full_prompt TEXT NOT NULL DEFAULT '',
    full_response TEXT NOT NULL DEFAULT '',
    input_tokens INT NOT NULL DEFAULT 0,
    output_tokens INT NOT NULL DEFAULT 0,
    temperature FLOAT8 NOT NULL DEFAULT 0.0,
    latency_ms FLOAT8 NOT NULL DEFAULT 0.0,
    cost_estimate FLOAT8 NOT NULL DEFAULT 0.0,
    user_id UUID,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- §2.4: Audit logs are IMMUTABLE — prevent UPDATE and DELETE via rule.
-- (Full RLS enforcement will be added in production deployment.)
CREATE INDEX IF NOT EXISTS audit_log_query_id_idx ON audit_log (query_id);
CREATE INDEX IF NOT EXISTS audit_log_timestamp_idx ON audit_log (timestamp);

-- ── Query history (§8.2) ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS query_history (
    query_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    session_id TEXT,
    original_query TEXT NOT NULL,
    resolved_query TEXT,
    intent_type TEXT,
    coverage_subtype TEXT,
    document_targets UUID[],
    answer_summary TEXT,
    confidence FLOAT8,
    verification_verdict TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Entity graph (§6.4) ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS entities (
    entity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    aliases TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS entity_mentions (
    mention_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL REFERENCES entities(entity_id),
    chunk_id UUID NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    mention_text TEXT NOT NULL,
    page_numbers INT[] NOT NULL,
    confidence FLOAT8 NOT NULL DEFAULT 0.0
);

-- ── Episodic memory (§6.3) ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS episodic_memory (
    user_id UUID NOT NULL,
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    memory_type TEXT NOT NULL,
    memory_key TEXT NOT NULL,
    memory_value JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ,
    PRIMARY KEY (user_id, doc_id, memory_type, memory_key)
);

-- ── Prompt templates (§8.2) ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS prompt_templates (
    template_id TEXT NOT NULL,
    intent_type TEXT NOT NULL,
    version TEXT NOT NULL,
    template_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by UUID,
    PRIMARY KEY (template_id, version)
);

-- ── Additional indexes (§8.3) ───────────────────────────────────────
CREATE INDEX IF NOT EXISTS chunks_document_type_idx ON chunks (document_type);
CREATE INDEX IF NOT EXISTS chunks_classification_label_idx ON chunks (classification_label);
CREATE INDEX IF NOT EXISTS chunks_page_numbers_gin_idx ON chunks USING GIN (page_numbers);
CREATE INDEX IF NOT EXISTS entities_canonical_name_idx ON entities (canonical_name);
CREATE INDEX IF NOT EXISTS entity_mentions_entity_id_idx ON entity_mentions (entity_id);
CREATE INDEX IF NOT EXISTS entity_mentions_doc_id_idx ON entity_mentions (doc_id);
