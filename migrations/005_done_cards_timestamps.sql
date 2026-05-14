-- Migration 007: add per-card audit timestamps to done_cards.

ALTER TABLE done_cards
    ADD COLUMN IF NOT EXISTS started_at  TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS finished_at TIMESTAMPTZ;
