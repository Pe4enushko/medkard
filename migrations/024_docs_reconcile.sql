-- 024_docs_reconcile.sql
-- Reconcile the docs schema to a single contextual embedding.
-- Converges BOTH origins to one clean shape:
--   * fresh DB (fact_q_* columns/indexes created by 002)
--   * drifted stand DB (untracked chunk_embedding + hyde_reembedded + docs_chunk_embedding_hnsw)
-- Forward-only, idempotent. VECTOR dim stays 1024 (Qwen3-Embedding-0.6B).

-- Data-loss guard: only meaningful when both columns exist (live stand). On a fresh DB
-- chunk_embedding is absent -> skipped; docs is empty anyway.
DO $$
DECLARE
    unmigrated int;
BEGIN
    IF (SELECT count(*) FROM information_schema.columns
        WHERE table_name = 'docs'
          AND column_name IN ('chunk_embedding', 'embedding')) = 2 THEN
        SELECT count(*) INTO unmigrated
        FROM docs
        WHERE embedding IS NULL AND chunk_embedding IS NOT NULL;
        IF unmigrated > 0 THEN
            RAISE EXCEPTION
                '% строк docs держат вектор только в chunk_embedding — прогони reingest --force-all до этой миграции',
                unmigrated;
        END IF;
    END IF;
END$$;

DROP INDEX IF EXISTS docs_fact_q_embedding_idx;
DROP INDEX IF EXISTS docs_procedure_q_embedding_idx;
DROP INDEX IF EXISTS docs_constraint_q_embedding_idx;
DROP INDEX IF EXISTS docs_chunk_embedding_hnsw;

ALTER TABLE docs
    DROP COLUMN IF EXISTS fact_q,
    DROP COLUMN IF EXISTS procedure_q,
    DROP COLUMN IF EXISTS constraint_q,
    DROP COLUMN IF EXISTS fact_q_embedding,
    DROP COLUMN IF EXISTS procedure_q_embedding,
    DROP COLUMN IF EXISTS constraint_q_embedding,
    DROP COLUMN IF EXISTS chunk_embedding,
    DROP COLUMN IF EXISTS hyde_reembedded,
    ADD COLUMN IF NOT EXISTS embedding VECTOR(1024);

CREATE INDEX IF NOT EXISTS docs_embedding_idx
    ON docs USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
