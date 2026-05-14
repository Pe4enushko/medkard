-- Migration 007: done_cards table — one row per audited card.
--
-- card_data     raw text content of the card
-- card_guid     1C appointment GUID
-- formal_result formal structure audit output as text
-- diag_result   diagnosis audit output as text
-- token_count   total LLM tokens consumed for this card
-- time_ms       total time spent auditing this card in milliseconds

CREATE TABLE IF NOT EXISTS done_cards (
    id           UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    card_guid    TEXT        UNIQUE,
    card_data    TEXT        NOT NULL,
    formal_result TEXT       NOT NULL,
    diag_result  TEXT        NOT NULL,
    token_count  INTEGER     NOT NULL,
    time_ms      INTEGER     NOT NULL
);

CREATE INDEX IF NOT EXISTS done_cards_card_guid_idx ON done_cards (card_guid);
