-- Migration 006: add raw checker source text to audit results
ALTER TABLE results ADD COLUMN IF NOT EXISTS sources TEXT;
