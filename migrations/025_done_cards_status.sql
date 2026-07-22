-- migrations/025_done_cards_status.sql
-- Migration 025: status column on done_cards — tracks whether a card's
-- stored data has been through the audit pipeline yet.
--
-- 'pending' = card_data is raw/unaudited (freshly pushed or never audited);
-- 'done'    = card has a terminal outcome (audited, ignored, or broken).
--
-- Existing rows are backfilled to 'done': every row already in the table
-- came from a completed pipeline run (upsert / upsert_ignored / upsert_broken),
-- so none of them are "still pending" under the new column's meaning.

ALTER TABLE done_cards
    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'done'));

UPDATE done_cards SET status = 'done' WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS done_cards_status_idx ON done_cards (status);
