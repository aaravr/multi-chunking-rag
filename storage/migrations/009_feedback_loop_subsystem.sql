-- Migration 009: Feedback Loop Subsystem — Production-Grade Tables
-- Supports the full feedback-to-retraining pipeline:
--   Ingest → Join → Normalize → Attribute → Build → Guard → Retrain → Evaluate → Promote
--
-- All tables carry boundary_key for training isolation: B = (client, division, jurisdiction)
-- All training rows carry source_feedback_ids for lineage preservation.

-- ── Prediction Traces (runtime decision records) ────────────────────

CREATE TABLE IF NOT EXISTS prediction_traces (
    trace_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id        TEXT NOT NULL,
    doc_id          UUID,
    user_id         TEXT,
    boundary_client TEXT NOT NULL,
    boundary_division TEXT NOT NULL DEFAULT '',
    boundary_jurisdiction TEXT NOT NULL DEFAULT '',

    -- Per-layer decisions (JSONB for flexibility, validated at app layer)
    planner_decision    JSONB NOT NULL DEFAULT '{}',
    classifier_decision JSONB NOT NULL DEFAULT '{}',
    chunking_decision   JSONB NOT NULL DEFAULT '{}',
    extraction_decision JSONB NOT NULL DEFAULT '{}',
    transformation_decision JSONB NOT NULL DEFAULT '{}',

    -- Final output
    final_answer        TEXT,
    final_confidence    FLOAT NOT NULL DEFAULT 0.0,
    citations           TEXT[],
    model_versions      JSONB NOT NULL DEFAULT '{}',
    total_latency_ms    FLOAT NOT NULL DEFAULT 0.0,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prediction_traces_query ON prediction_traces(query_id);
CREATE INDEX IF NOT EXISTS idx_prediction_traces_doc ON prediction_traces(doc_id);
CREATE INDEX IF NOT EXISTS idx_prediction_traces_boundary ON prediction_traces(boundary_client, boundary_division);
CREATE INDEX IF NOT EXISTS idx_prediction_traces_created ON prediction_traces(created_at);


-- ── Feedback Events (enhanced from migration 007) ───────────────────

CREATE TABLE IF NOT EXISTS feedback_events (
    feedback_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id        UUID REFERENCES prediction_traces(trace_id),
    query_id        TEXT,
    doc_id          UUID,
    user_id         TEXT,
    boundary_client TEXT NOT NULL,
    boundary_division TEXT NOT NULL DEFAULT '',
    boundary_jurisdiction TEXT NOT NULL DEFAULT '',

    -- Feedback content
    rating          TEXT NOT NULL CHECK (rating IN ('positive', 'negative', 'correction')),
    comment         TEXT,
    correct_answer  TEXT,
    correct_document_type TEXT,
    correct_classification_label TEXT,
    correct_evidence_spans TEXT[],
    correct_field_values JSONB DEFAULT '{}',
    processing_path_override TEXT,

    -- Metadata
    channel         TEXT NOT NULL DEFAULT 'api',
    cited_chunk_ids TEXT[],
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_events_trace ON feedback_events(trace_id);
CREATE INDEX IF NOT EXISTS idx_feedback_events_query ON feedback_events(query_id);
CREATE INDEX IF NOT EXISTS idx_feedback_events_doc ON feedback_events(doc_id);
CREATE INDEX IF NOT EXISTS idx_feedback_events_boundary ON feedback_events(boundary_client, boundary_division);
CREATE INDEX IF NOT EXISTS idx_feedback_events_rating ON feedback_events(rating);
CREATE INDEX IF NOT EXISTS idx_feedback_events_created ON feedback_events(created_at);


-- ── Feedback Attributions ───────────────────────────────────────────

CREATE TABLE IF NOT EXISTS feedback_attributions (
    attribution_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feedback_id     UUID NOT NULL REFERENCES feedback_events(feedback_id),
    trace_id        UUID REFERENCES prediction_traces(trace_id),
    boundary_client TEXT NOT NULL,
    boundary_division TEXT NOT NULL DEFAULT '',
    boundary_jurisdiction TEXT NOT NULL DEFAULT '',

    -- Attribution result (JSONB array of impacted layers)
    impacted_layers JSONB NOT NULL DEFAULT '[]',
    attribution_method TEXT NOT NULL DEFAULT 'rule_based',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_attributions_feedback ON feedback_attributions(feedback_id);
CREATE INDEX IF NOT EXISTS idx_attributions_boundary ON feedback_attributions(boundary_client, boundary_division);


-- ── Training Rows (one table per layer for schema clarity) ──────────

-- Planner training rows: π*(z) = argmax_a E[R(a, z)]
CREATE TABLE IF NOT EXISTS training_rows_planner (
    row_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_feedback_ids UUID[] NOT NULL,
    boundary_client     TEXT NOT NULL,
    boundary_division   TEXT NOT NULL DEFAULT '',
    boundary_jurisdiction TEXT NOT NULL DEFAULT '',

    query_text          TEXT,
    document_type       TEXT,
    classification_label TEXT,
    page_count          INT DEFAULT 0,
    chosen_action       TEXT,
    chosen_processing_path TEXT,
    query_decomposition TEXT[],
    correct_action      TEXT,
    correct_processing_path TEXT,
    accuracy            FLOAT NOT NULL DEFAULT 0.0,
    review_cost         FLOAT NOT NULL DEFAULT 0.0,
    latency_ms          FLOAT NOT NULL DEFAULT 0.0,
    error_penalty       FLOAT NOT NULL DEFAULT 0.0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_planner_rows_boundary ON training_rows_planner(boundary_client, boundary_division);

-- Classifier training rows: P(c|z) = softmax(Wz + b)
CREATE TABLE IF NOT EXISTS training_rows_classifier (
    row_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_feedback_ids UUID[] NOT NULL,
    boundary_client     TEXT NOT NULL,
    boundary_division   TEXT NOT NULL DEFAULT '',
    boundary_jurisdiction TEXT NOT NULL DEFAULT '',

    filename            TEXT,
    front_matter_text   TEXT,
    structural_signals  JSONB DEFAULT '{}',
    predicted_document_type TEXT,
    predicted_classification_label TEXT,
    predicted_confidence FLOAT DEFAULT 0.0,
    correct_document_type TEXT,
    correct_classification_label TEXT,
    is_correct          BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_classifier_rows_boundary ON training_rows_classifier(boundary_client, boundary_division);
CREATE INDEX IF NOT EXISTS idx_classifier_rows_correct ON training_rows_classifier(is_correct);

-- Chunking training rows: Q(k,x) = α·EvidenceRecall + β·FieldAccuracy - γ·ContextLoss
CREATE TABLE IF NOT EXISTS training_rows_chunking (
    row_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_feedback_ids UUID[] NOT NULL,
    boundary_client     TEXT NOT NULL,
    boundary_division   TEXT NOT NULL DEFAULT '',
    boundary_jurisdiction TEXT NOT NULL DEFAULT '',

    document_type       TEXT,
    classification_label TEXT,
    page_count          INT DEFAULT 0,
    chosen_strategy     TEXT,
    processing_level    TEXT,
    chunk_count         INT DEFAULT 0,
    evidence_recall     FLOAT NOT NULL DEFAULT 0.0,
    field_accuracy      FLOAT NOT NULL DEFAULT 0.0,
    context_loss        FLOAT NOT NULL DEFAULT 0.0,
    missing_evidence_spans TEXT[],
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunking_rows_boundary ON training_rows_chunking(boundary_client, boundary_division);

-- Extraction training rows: v̂ = E(x, c, S)
CREATE TABLE IF NOT EXISTS training_rows_extraction (
    row_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_feedback_ids UUID[] NOT NULL,
    boundary_client     TEXT NOT NULL,
    boundary_division   TEXT NOT NULL DEFAULT '',
    boundary_jurisdiction TEXT NOT NULL DEFAULT '',

    document_type       TEXT,
    classification_label TEXT,
    field_name          TEXT NOT NULL,
    predicted_value     TEXT,
    predicted_confidence FLOAT DEFAULT 0.0,
    extraction_method   TEXT,
    source_chunk_ids    UUID[],
    source_text_snippet TEXT,
    correct_value       TEXT,
    is_correct          BOOLEAN NOT NULL DEFAULT FALSE,
    is_unresolved_reference BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_extraction_rows_boundary ON training_rows_extraction(boundary_client, boundary_division);
CREATE INDEX IF NOT EXISTS idx_extraction_rows_field ON training_rows_extraction(field_name);
CREATE INDEX IF NOT EXISTS idx_extraction_rows_unresolved ON training_rows_extraction(is_unresolved_reference) WHERE is_unresolved_reference = TRUE;

-- Calibration training rows: P(correct|z) = σ(wᵀz)
CREATE TABLE IF NOT EXISTS training_rows_calibration (
    row_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_feedback_ids UUID[] NOT NULL,
    boundary_client     TEXT NOT NULL,
    boundary_division   TEXT NOT NULL DEFAULT '',
    boundary_jurisdiction TEXT NOT NULL DEFAULT '',

    planner_confidence  FLOAT DEFAULT 0.0,
    classifier_confidence FLOAT DEFAULT 0.0,
    chunking_quality_score FLOAT DEFAULT 0.0,
    extraction_confidence FLOAT DEFAULT 0.0,
    final_confidence    FLOAT DEFAULT 0.0,
    document_type       TEXT,
    query_intent        TEXT,
    chunk_count         INT DEFAULT 0,
    model_id            TEXT,
    is_correct          BOOLEAN NOT NULL DEFAULT FALSE,
    confidence_bucket   TEXT,
    is_overconfident    BOOLEAN NOT NULL DEFAULT FALSE,
    is_underconfident   BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_calibration_rows_boundary ON training_rows_calibration(boundary_client, boundary_division);
CREATE INDEX IF NOT EXISTS idx_calibration_rows_bucket ON training_rows_calibration(confidence_bucket);
CREATE INDEX IF NOT EXISTS idx_calibration_rows_overconfident ON training_rows_calibration(is_overconfident) WHERE is_overconfident = TRUE;


-- ── Model Candidates ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS model_candidates (
    candidate_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    layer               TEXT NOT NULL CHECK (layer IN ('planning', 'classification', 'chunking', 'extraction', 'calibration')),
    boundary_client     TEXT NOT NULL,
    boundary_division   TEXT NOT NULL DEFAULT '',
    boundary_jurisdiction TEXT NOT NULL DEFAULT '',

    model_version       TEXT,
    parent_model_version TEXT,
    training_row_count  INT NOT NULL DEFAULT 0,
    source_feedback_ids UUID[],

    -- Hierarchical sharing: θ_B = θ_shared + δ_B
    shared_base_version TEXT,
    boundary_delta_version TEXT,

    stage               TEXT NOT NULL DEFAULT 'shadow' CHECK (stage IN ('shadow', 'canary', 'approved', 'rejected', 'rollback_ready')),
    training_started_at TIMESTAMPTZ,
    training_completed_at TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_candidates_layer_boundary ON model_candidates(layer, boundary_client, boundary_division);
CREATE INDEX IF NOT EXISTS idx_candidates_stage ON model_candidates(stage);


-- ── Evaluation Reports ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS evaluation_reports (
    report_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id        UUID NOT NULL REFERENCES model_candidates(candidate_id),
    layer               TEXT NOT NULL,
    boundary_client     TEXT NOT NULL,
    boundary_division   TEXT NOT NULL DEFAULT '',
    boundary_jurisdiction TEXT NOT NULL DEFAULT '',

    baseline_metrics    JSONB NOT NULL DEFAULT '{}',
    candidate_metrics   JSONB NOT NULL DEFAULT '{}',
    improvement_delta   FLOAT NOT NULL DEFAULT 0.0,
    is_statistically_significant BOOLEAN NOT NULL DEFAULT FALSE,
    p_value             FLOAT,
    recommendation      TEXT,
    explanation         TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_reports_candidate ON evaluation_reports(candidate_id);


-- ── Boundary Sharing Approvals ──────────────────────────────────────

CREATE TABLE IF NOT EXISTS boundary_sharing_approvals (
    approval_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_client       TEXT NOT NULL,
    source_division     TEXT NOT NULL DEFAULT '',
    target_client       TEXT NOT NULL,
    target_division     TEXT NOT NULL DEFAULT '',
    approved_by         TEXT NOT NULL,
    approved_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at          TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_sharing_source ON boundary_sharing_approvals(source_client, source_division);
CREATE INDEX IF NOT EXISTS idx_sharing_target ON boundary_sharing_approvals(target_client, target_division);


-- ── Retraining Jobs ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS retraining_jobs (
    job_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    layer               TEXT NOT NULL,
    boundary_client     TEXT NOT NULL,
    boundary_division   TEXT NOT NULL DEFAULT '',
    boundary_jurisdiction TEXT NOT NULL DEFAULT '',

    trigger_type        TEXT NOT NULL CHECK (trigger_type IN ('scheduled', 'threshold', 'manual', 'event_driven')),
    row_count           INT NOT NULL DEFAULT 0,
    status              TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    candidate_id        UUID REFERENCES model_candidates(candidate_id),

    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    error_message       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_retraining_jobs_layer ON retraining_jobs(layer, boundary_client);
CREATE INDEX IF NOT EXISTS idx_retraining_jobs_status ON retraining_jobs(status);
