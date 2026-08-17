-- Migration 027: journal + counters for pushes that overwrite stored card data.
--
-- POST /visits/push funnels into DoneCardsStorage.upsert_pending, whose
-- ON CONFLICT branch resets card_data and clears every audit-derived column
-- (formal_result / diag_result / icd_check_result / ignored / broken /
-- stacktrace). When the row had already been audited, that audit output is
-- gone with no trace of it ever existing.
--
-- Two distinct things are recorded here, because they answer different
-- questions and have very different storage costs:
--
--   1. push_count on done_cards — how many times a row was overwritten at all,
--      including harmless re-pushes over a not-yet-audited card. A counter,
--      not a history: cheap, one integer per row.
--   2. audit_overwrite_journal — a row per overwrite that actually destroyed
--      audit results, holding the old card_data and results verbatim so the
--      lost work can be inspected (or recovered) after the fact.
--
-- Both are driven by ONE trigger — the same one that performs the wipe — so a
-- clobber cannot happen without being accounted for.

-- --------------------------------------------------------------------------
-- Per-organization opt-out
-- --------------------------------------------------------------------------
-- Default TRUE: journalling is on unless an org explicitly turns it off.
-- The flag gates ONLY the journal (which stores whole cards and can grow
-- large), never push_count — otherwise disabling it would silently stop the
-- statistics too, and a zero would be unreadable: "nothing was overwritten"
-- and "we stopped looking" must not look the same.
ALTER TABLE organizations
    ADD COLUMN IF NOT EXISTS audit_overwrite_journal_enabled BOOLEAN NOT NULL DEFAULT TRUE;

-- --------------------------------------------------------------------------
-- Overwrite counter
-- --------------------------------------------------------------------------
-- Starts at 0 for existing rows rather than being backfilled: how many times
-- a card was pushed before this migration is unknowable, and a made-up number
-- is worse than an honest zero. Counting starts at rollout.
ALTER TABLE done_cards
    ADD COLUMN IF NOT EXISTS push_count INTEGER NOT NULL DEFAULT 0;

-- --------------------------------------------------------------------------
-- The journal
-- --------------------------------------------------------------------------
-- Columns mirror the done_cards columns the wipe clears, and hold the OLD
-- values — what was lost, not what replaced it. card_data is the old card too,
-- so a card can be reconstructed as it stood when the audit ran.
CREATE TABLE IF NOT EXISTS audit_overwrite_journal (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    card_guid         TEXT,
    organization_id   UUID REFERENCES organizations(id),
    overwritten_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- the overwritten card and its lost audit output
    card_data         JSONB,
    formal_result     JSONB,
    diag_result       JSONB,
    icd_check_result  JSONB,
    token_count       INTEGER,
    time_ms           INTEGER,
    started_at        TIMESTAMPTZ,
    finished_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS audit_overwrite_journal_org_time_idx
    ON audit_overwrite_journal (organization_id, overwritten_at DESC);

CREATE INDEX IF NOT EXISTS audit_overwrite_journal_card_guid_idx
    ON audit_overwrite_journal (card_guid);

-- --------------------------------------------------------------------------
-- The trigger: count every overwrite, journal the destructive ones
-- --------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION done_cards_journal_overwrite()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    journal_enabled BOOLEAN;
BEGIN
    -- Every push over an existing row counts, whatever its previous state and
    -- regardless of the org's journal flag.
    NEW.push_count := COALESCE(OLD.push_count, 0) + 1;

    -- Only a push over genuine audit output destroys anything worth keeping.
    -- Rows that were still 'pending' have no results yet; 'ignored' rows carry
    -- none at all, and 'broken' rows carry only a stacktrace. All three fall
    -- out here as counted-but-not-journalled.
    IF OLD.formal_result IS NULL
       AND OLD.diag_result IS NULL
       AND OLD.icd_check_result IS NULL
    THEN
        RETURN NEW;
    END IF;

    -- A card with no organization is journalled: absence of an org is not an
    -- opt-out, and dropping those rows would lose exactly the cards whose
    -- ownership is already unclear.
    IF OLD.organization_id IS NULL THEN
        journal_enabled := TRUE;
    ELSE
        SELECT organizations.audit_overwrite_journal_enabled
          INTO journal_enabled
          FROM organizations
         WHERE organizations.id = OLD.organization_id;
        journal_enabled := COALESCE(journal_enabled, TRUE);
    END IF;

    IF NOT journal_enabled THEN
        RETURN NEW;
    END IF;

    INSERT INTO audit_overwrite_journal (
        card_guid, organization_id, card_data,
        formal_result, diag_result, icd_check_result,
        token_count, time_ms, started_at, finished_at
    ) VALUES (
        OLD.card_guid, OLD.organization_id, OLD.card_data,
        OLD.formal_result, OLD.diag_result, OLD.icd_check_result,
        OLD.token_count, OLD.time_ms, OLD.started_at, OLD.finished_at
    );

    RETURN NEW;
END;
$$;

-- NEW.status = 'pending' is the push signature: upsert_pending is the only
-- writer that sets a row to 'pending'. OLD.status is deliberately unconstrained
-- so that a re-push over an already-pending card counts too — it overwrote
-- stored data just the same, it simply had no results to destroy.
--
-- A done -> done update (a re-audit via upsert) replaces results with fresh
-- ones rather than losing them, and is not a push, so it is excluded.
--
-- BEFORE UPDATE, so NEW.push_count is assigned as part of the same write —
-- matching the existing done_cards_set_updated_at trigger's approach.
DROP TRIGGER IF EXISTS done_cards_journal_overwrite ON done_cards;
CREATE TRIGGER done_cards_journal_overwrite
    BEFORE UPDATE ON done_cards
    FOR EACH ROW
    WHEN (NEW.status = 'pending')
    EXECUTE FUNCTION done_cards_journal_overwrite();

-- --------------------------------------------------------------------------
-- Metrics view
-- --------------------------------------------------------------------------
-- Per organization:
--   overwrites_with_results — pushes that destroyed audit output (journal rows)
--   overwrites_total        — every push over an existing row (sum of push_count)
--   overwrites_no_results   — the difference: re-pushes that cost nothing
--
-- CAVEAT: with journalling disabled, overwrites_with_results is undercounted
-- and overwrites_no_results is inflated by the same amount, since the former
-- is derived from journal rows that were never written. Read the split only
-- where journal_enabled is true; overwrites_total is always accurate.
CREATE OR REPLACE VIEW audit_overwrite_metrics AS
WITH journal_stats AS (
    SELECT
        organization_id,
        count(*)                        AS overwrites_with_results,
        count(DISTINCT card_guid)       AS cards_affected,
        min(overwritten_at)             AS first_overwrite_at,
        max(overwritten_at)             AS last_overwrite_at,
        count(*) FILTER (WHERE overwritten_at >= now() - INTERVAL '7 days')  AS overwrites_last_7d,
        count(*) FILTER (WHERE overwritten_at >= now() - INTERVAL '30 days') AS overwrites_last_30d
    FROM audit_overwrite_journal
    GROUP BY organization_id
),
push_stats AS (
    SELECT
        organization_id,
        sum(push_count)::bigint AS overwrites_total
    FROM done_cards
    GROUP BY organization_id
)
SELECT
    organizations.name                                          AS organization_name,
    organizations.audit_overwrite_journal_enabled               AS journal_enabled,
    COALESCE(journal_stats.overwrites_with_results, 0)          AS overwrites_with_results,
    COALESCE(push_stats.overwrites_total, 0)                    AS overwrites_total,
    GREATEST(
        COALESCE(push_stats.overwrites_total, 0)
            - COALESCE(journal_stats.overwrites_with_results, 0),
        0
    )                                                           AS overwrites_no_results,
    COALESCE(journal_stats.cards_affected, 0)                   AS cards_affected,
    journal_stats.first_overwrite_at,
    journal_stats.last_overwrite_at,
    COALESCE(journal_stats.overwrites_last_7d, 0)               AS overwrites_last_7d,
    COALESCE(journal_stats.overwrites_last_30d, 0)              AS overwrites_last_30d
FROM organizations
LEFT JOIN journal_stats ON journal_stats.organization_id = organizations.id
LEFT JOIN push_stats    ON push_stats.organization_id    = organizations.id
ORDER BY organizations.name;
