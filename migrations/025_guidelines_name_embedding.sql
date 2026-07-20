-- migrations/025_guidelines_name_embedding.sql
-- Add a vector column for guideline *names* to the registry, so semantic search
-- over guideline titles is possible. Embeds title + age category (passage mode,
-- bare embed, no instruct prefix). Populated by GuidelinesStorage.upsert_many.
-- Forward-only, idempotent. Dim 1024 (Qwen3-Embedding-0.6B).

ALTER TABLE guidelines
    ADD COLUMN IF NOT EXISTS name_embedding VECTOR(1024);

CREATE INDEX IF NOT EXISTS guidelines_name_embedding_idx
    ON guidelines USING hnsw (name_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE name_embedding IS NOT NULL;
