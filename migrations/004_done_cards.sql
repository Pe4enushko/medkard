-- Migration 007: done_cards table — one row per audited card.
--
-- ignored = TRUE means the card was skipped by the ICD ignore-list; in that
-- case only card_guid is meaningful — all audit columns may be NULL.

CREATE TABLE IF NOT EXISTS done_cards (
    id            UUID     PRIMARY KEY DEFAULT uuid_generate_v4(),
    card_guid     TEXT     UNIQUE,
    card_data     JSONB,
    formal_result JSONB,
    diag_result   JSONB,
    token_count   INTEGER,
    time_ms       INTEGER,
    ignored       BOOLEAN  NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS done_cards_card_guid_idx ON done_cards (card_guid);
