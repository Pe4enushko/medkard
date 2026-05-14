-- Migration 009: convert done_cards text columns to JSONB.
--
-- card_data was already stored as valid JSON.
-- formal_result and diag_result were stored as pretty-formatted text in older
-- rows; those rows must be cleared before running this migration on an existing
-- DB (TRUNCATE done_cards), or this migration should only be applied to a
-- fresh DB.

ALTER TABLE done_cards
    ALTER COLUMN card_data     TYPE JSONB USING card_data::jsonb,
    ALTER COLUMN formal_result TYPE JSONB USING formal_result::jsonb,
    ALTER COLUMN diag_result   TYPE JSONB USING diag_result::jsonb;
