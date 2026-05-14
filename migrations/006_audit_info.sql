-- Migration 006: audit_info table — one row per pipeline run summary.
--
-- Columns:
--   date                  TIMESTAMPTZ  when the run was recorded
--   total_tokens          INTEGER      total LLM tokens consumed
--   total_time_sec        NUMERIC      total wall-clock seconds for the run
--   avg_time_per_card_sec NUMERIC      total_time_sec / total_cards
--   avg_tokens_per_card   NUMERIC      total_tokens / total_cards
--   total_cards           INTEGER      all appointments seen by the pipeline
--   filtered_cards        INTEGER      appointments skipped due to icd_ignore_codes
--   icd_ignore_codes      TEXT[]       ICD codes that triggered filtering
--   organization          TEXT         optional clinic / organization name

CREATE TABLE IF NOT EXISTS audit_info (
    id                    UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    date                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    total_tokens          INTEGER     NOT NULL,
    total_time_sec        NUMERIC     NOT NULL,
    avg_time_per_card_sec NUMERIC     NOT NULL,
    avg_tokens_per_card   NUMERIC     NOT NULL,
    total_cards           INTEGER     NOT NULL,
    filtered_cards        INTEGER     NOT NULL DEFAULT 0,
    icd_ignore_codes      TEXT[]      NOT NULL DEFAULT '{}',
    organization          TEXT
);
