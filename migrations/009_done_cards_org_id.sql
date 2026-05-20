-- Migration 009: wire organization_id into done_cards.
--
-- Nullable FK — rows created before organizations existed stay NULL.

ALTER TABLE done_cards
    ADD COLUMN IF NOT EXISTS organization_id UUID REFERENCES organizations(id);

CREATE INDEX IF NOT EXISTS done_cards_organization_id_idx ON done_cards (organization_id);
