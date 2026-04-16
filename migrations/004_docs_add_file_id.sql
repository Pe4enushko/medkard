-- 004_docs_add_file_id.sql
-- Adds the file_id column to an existing docs table.
-- Safe to run even if the column already exists (IF NOT EXISTS guard).

ALTER TABLE docs
    ADD COLUMN IF NOT EXISTS file_id TEXT NOT NULL DEFAULT '';
