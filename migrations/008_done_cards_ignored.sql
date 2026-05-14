-- Migration 008: add ignored flag to done_cards.
--
-- ignored = TRUE means the card was skipped by the ICD ignore-list; in that
-- case only card_guid is meaningful — all audit columns may be NULL.

ALTER TABLE done_cards
    ADD COLUMN IF NOT EXISTS ignored BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE done_cards
    ALTER COLUMN card_data     DROP NOT NULL,
    ALTER COLUMN formal_result DROP NOT NULL,
    ALTER COLUMN diag_result   DROP NOT NULL,
    ALTER COLUMN token_count   DROP NOT NULL,
    ALTER COLUMN time_ms       DROP NOT NULL;
