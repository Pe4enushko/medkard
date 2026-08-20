-- Migration 027: push_log — one row per push that overwrites an existing
-- done_cards row, dated, so per-organization-per-date push metrics can be
-- computed directly. Every push counts; overrode_audit distinguishes the
-- ones that destroyed a completed audit result from harmless re-pushes.
--
-- POST /visits/push funnels into DoneCardsStorage.upsert_pending, whose
-- ON CONFLICT branch resets card_data and clears every audit-derived column
-- (formal_result / diag_result / icd_check_result / ignored / broken /
-- stacktrace). If the row had already been audited, that output is gone
-- with no trace it ever existed — push_log is that trace.

CREATE TABLE IF NOT EXISTS push_log (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    card_guid        TEXT,
    organization_id  UUID REFERENCES organizations(id),
    pushed_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    overrode_audit   BOOLEAN NOT NULL,
    card_data        JSONB
);

COMMENT ON COLUMN push_log.card_data IS
    'Резерв под снимок card_data, которую этот пуш затёр. Сейчас всегда NULL — '
    'колонка существует заранее, чтобы начать её заполнять без новой миграции.';

CREATE INDEX IF NOT EXISTS push_log_org_date_idx ON push_log (organization_id, pushed_at);
CREATE INDEX IF NOT EXISTS push_log_card_guid_idx ON push_log (card_guid);

-- --------------------------------------------------------------------------
-- The trigger: log every push, dated, with whether it destroyed audit output
-- --------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION done_cards_log_push()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO push_log (card_guid, organization_id, overrode_audit)
    VALUES (
        OLD.card_guid,
        OLD.organization_id,
        OLD.formal_result IS NOT NULL
            OR OLD.diag_result IS NOT NULL
            OR OLD.icd_check_result IS NOT NULL
    );
    RETURN NEW;
END;
$$;

-- NEW.status = 'pending' is the push signature: upsert_pending is the only
-- writer that sets a row to 'pending'. OLD.status is deliberately unconstrained
-- so a re-push over an already-pending card is logged too (overrode_audit =
-- FALSE — it had no results to lose, but it still overwrote stored data).
--
-- A done -> done update (a re-audit via upsert()) replaces results with fresh
-- ones rather than losing them, and does not touch status, so it never fires
-- this trigger at all.
--
-- BEFORE UPDATE, so the log write happens as part of the same transaction as
-- the wipe it is recording — matching the existing done_cards_set_updated_at
-- trigger's approach.
DROP TRIGGER IF EXISTS done_cards_journal_overwrite ON done_cards;
DROP TRIGGER IF EXISTS done_cards_log_push ON done_cards;
CREATE TRIGGER done_cards_log_push
    BEFORE UPDATE ON done_cards
    FOR EACH ROW
    WHEN (NEW.status = 'pending')
    EXECUTE FUNCTION done_cards_log_push();

-- Drop artifacts from an earlier version of this migration, if this database
-- ever had it applied (old journal table + counter column + org flag).
DROP FUNCTION IF EXISTS done_cards_journal_overwrite();
DROP TABLE IF EXISTS audit_overwrite_journal;
ALTER TABLE done_cards DROP COLUMN IF EXISTS push_count;
ALTER TABLE organizations DROP COLUMN IF EXISTS audit_overwrite_journal_enabled;
DROP VIEW IF EXISTS audit_overwrite_metrics;

-- --------------------------------------------------------------------------
-- Metrics view — pushes per organization per date
-- --------------------------------------------------------------------------
CREATE OR REPLACE VIEW push_metrics_by_date AS
SELECT
    organizations.name                                   AS organization_name,
    (push_log.pushed_at AT TIME ZONE 'UTC')::date         AS push_date,
    count(*)                                              AS pushes_total,
    count(*) FILTER (WHERE push_log.overrode_audit)       AS pushes_overrode_audit,
    count(*) FILTER (WHERE NOT push_log.overrode_audit)   AS pushes_no_override
FROM push_log
LEFT JOIN organizations ON organizations.id = push_log.organization_id
GROUP BY organizations.name, (push_log.pushed_at AT TIME ZONE 'UTC')::date
ORDER BY organization_name, push_date;
