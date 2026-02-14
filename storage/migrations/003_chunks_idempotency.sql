-- WO-010: Add unique constraint for idempotent chunk insertion.
-- Ensures ingestion retries do NOT duplicate chunks.

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chunks_doc_macro_child_unique'
    ) THEN
        -- Remove duplicates first (keep one row per doc_id, macro_id, child_id).
        DELETE FROM chunks a
        USING chunks b
        WHERE a.ctid < b.ctid
          AND a.doc_id = b.doc_id
          AND a.macro_id = b.macro_id
          AND a.child_id = b.child_id;

        ALTER TABLE chunks
            ADD CONSTRAINT chunks_doc_macro_child_unique
            UNIQUE (doc_id, macro_id, child_id);
    END IF;
END
$$;
