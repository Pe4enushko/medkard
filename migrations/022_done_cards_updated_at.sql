-- Migration 019: change-tracking column for incremental export to the engine's
-- analyst replica. updated_at is bumped to transaction now() on every insert
-- and update, regardless of card type (audited / ignored / broken).

ALTER TABLE done_cards
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();

CREATE OR REPLACE FUNCTION done_cards_touch_updated_at()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS done_cards_set_updated_at ON done_cards;
CREATE TRIGGER done_cards_set_updated_at
    BEFORE INSERT OR UPDATE ON done_cards
    FOR EACH ROW EXECUTE FUNCTION done_cards_touch_updated_at();

CREATE INDEX IF NOT EXISTS done_cards_updated_at_idx ON done_cards (updated_at);
