-- Migration 007: Feedback and Retraining tables (§4.10, §4.11)
--
-- Stores user feedback on query answers and retraining event logs.
-- Used by FeedbackAgent and RetrainingAgent for the self-learning loop.

CREATE TABLE IF NOT EXISTS feedback_entries (
    feedback_id    UUID PRIMARY KEY,
    query_id       TEXT NOT NULL,
    doc_id         UUID,
    rating         TEXT NOT NULL CHECK (rating IN ('positive', 'negative', 'correction')),
    comment        TEXT DEFAULT '',
    correct_answer TEXT DEFAULT '',
    cited_chunk_ids TEXT[] DEFAULT '{}',
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_doc_id ON feedback_entries(doc_id);
CREATE INDEX IF NOT EXISTS idx_feedback_query_id ON feedback_entries(query_id);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback_entries(rating);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback_entries(created_at);

CREATE TABLE IF NOT EXISTS retraining_events (
    event_id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trigger_type           TEXT NOT NULL CHECK (trigger_type IN ('scheduled', 'threshold', 'manual')),
    retrained_components   TEXT[] DEFAULT '{}',
    metrics_before         JSONB DEFAULT '{}',
    metrics_after          JSONB DEFAULT '{}',
    feedback_entries_used  INT DEFAULT 0,
    patterns_pruned        INT DEFAULT 0,
    duration_ms            FLOAT DEFAULT 0.0,
    skipped_reason         TEXT DEFAULT '',
    created_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_retraining_created_at ON retraining_events(created_at);

-- Chunking outcomes table (referenced by preprocessor_agent but not yet created)
CREATE TABLE IF NOT EXISTS chunking_outcomes (
    outcome_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id               UUID NOT NULL,
    strategy_name        TEXT NOT NULL,
    document_type        TEXT NOT NULL,
    classification_label TEXT NOT NULL,
    page_count           INT NOT NULL,
    chunk_count          INT NOT NULL,
    avg_chunk_tokens     FLOAT NOT NULL,
    table_chunk_ratio    FLOAT DEFAULT 0.0,
    heading_chunk_ratio  FLOAT DEFAULT 0.0,
    boilerplate_ratio    FLOAT DEFAULT 0.0,
    processing_time_ms   FLOAT DEFAULT 0.0,
    quality_score        FLOAT DEFAULT 0.0,
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunking_outcomes_doc_type
    ON chunking_outcomes(document_type, classification_label);
