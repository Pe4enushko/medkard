-- Migration 016: replace per-organization API key columns with a standalone
-- api_keys table, and drop the now-unused columns from organizations.
--
-- Only one integrating service exists today, so keys authenticate "this is
-- our trusted app" rather than "this is org X" — the caller names the org
-- it wants per request (?org=...). A standalone table means adding
-- per-key org scoping later (if a second integration ever needs it) is a
-- new column/join, not a schema migration off of organizations again.

CREATE TABLE IF NOT EXISTS api_keys (
    id               UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    label            TEXT        NOT NULL,
    key_hash         TEXT        NOT NULL,
    key_prefix       TEXT        NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at       TIMESTAMPTZ
);

-- Only active (non-revoked) keys need unique hashes — otherwise a revoked
-- key's hash could never be reissued.
CREATE UNIQUE INDEX IF NOT EXISTS api_keys_key_hash_idx
    ON api_keys (key_hash) WHERE revoked_at IS NULL;

ALTER TABLE organizations
    DROP COLUMN IF EXISTS api_key_hash,
    DROP COLUMN IF EXISTS api_key_prefix,
    DROP COLUMN IF EXISTS api_key_created_at,
    DROP COLUMN IF EXISTS api_key_revoked_at;
