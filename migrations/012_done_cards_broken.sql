ALTER TABLE done_cards
    ADD COLUMN IF NOT EXISTS broken    BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS stacktrace TEXT;

CREATE OR REPLACE VIEW broken_cards AS
SELECT
    done_cards.id,
    done_cards.card_guid,
    organizations.name                                         AS organization_name,
    to_date(done_cards.card_data -> 'Прием' ->> 'DATE', 'DD.MM.YYYY') AS visit_date,
    done_cards.started_at,
    done_cards.stacktrace
FROM done_cards
LEFT JOIN organizations ON organizations.id = done_cards.organization_id
WHERE done_cards.broken = TRUE
ORDER BY done_cards.started_at DESC;
