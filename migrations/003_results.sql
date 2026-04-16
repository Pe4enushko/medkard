-- 003_results.sql
-- Stores audit results for processed ambulatory cards.
--
-- clinical_sources JSON shape:
-- [
--   {
--     "flag": "<flag_text>",
--     "sources": [
--       {
--         "file":          "<manifest ID, e.g. 340_2>",
--         "file_metadata": { <full manifest.csv row for that file> },
--         "section":       "<TOC section title>",   -- optional, omit if absent
--         "page":          <0-based page index>
--       }
--     ]
--   }
-- ]

CREATE TABLE IF NOT EXISTS results (
    id               UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    input            JSONB   NOT NULL,            -- raw JSON payload from 1C
    flags            TEXT[]  NOT NULL DEFAULT '{}',
    clinical_sources JSONB   NOT NULL DEFAULT '[]'
);

-- GIN indexes for array containment queries on flags and JSONB path queries on sources.
CREATE INDEX IF NOT EXISTS results_flags_idx           ON results USING gin (flags);
CREATE INDEX IF NOT EXISTS results_clinical_sources_idx ON results USING gin (clinical_sources);
