-- Migration 006: pgvector-backed classification embeddings (MASTER_PROMPT §4.8)
-- Stores document front-matter embeddings for the self-learning classifier agent.
-- Replaces in-memory LlamaIndex SimpleVectorStore with persistent pgvector storage.

CREATE TABLE IF NOT EXISTS classification_embeddings (
    embedding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_type TEXT NOT NULL,
    classification_label TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    source_doc_id UUID REFERENCES documents(doc_id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW index for fast cosine similarity search on classification embeddings
CREATE INDEX IF NOT EXISTS classification_embeddings_hnsw_idx
    ON classification_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS classification_embeddings_doc_type_idx
    ON classification_embeddings (document_type);
CREATE INDEX IF NOT EXISTS classification_embeddings_label_idx
    ON classification_embeddings (classification_label);
