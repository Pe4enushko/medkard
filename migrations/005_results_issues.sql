-- Migration 005: replace clinical_sources JSONB with issues JSONB
ALTER TABLE results ADD COLUMN IF NOT EXISTS issues JSONB NOT NULL DEFAULT '[]';
ALTER TABLE results DROP COLUMN IF EXISTS clinical_sources;
