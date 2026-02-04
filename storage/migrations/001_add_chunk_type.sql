ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS chunk_type TEXT;

UPDATE chunks
SET chunk_type = 'table'
WHERE chunk_type IS NULL
  AND text_content LIKE '[TABLE]%';

UPDATE chunks
SET chunk_type = 'narrative'
WHERE chunk_type IS NULL;

ALTER TABLE chunks
    ALTER COLUMN chunk_type SET NOT NULL;

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
