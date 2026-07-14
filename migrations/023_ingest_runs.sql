-- 023_ingest_runs.sql — resume-state for scripts/reingest-pdfs.py.
-- content_hash = sha256 of the PDF at the last successful ('done') reingest.
CREATE TABLE IF NOT EXISTS ingest_runs (
    file_id      TEXT PRIMARY KEY,
    status       TEXT NOT NULL DEFAULT 'pending',   -- 'pending' | 'done' | 'failed'
    content_hash TEXT,
    error        TEXT,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
