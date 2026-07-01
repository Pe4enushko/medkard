-- Migration 015: API keys for organizations (pull API auth).
--
-- Only the SHA-256 hash of the raw key is stored; api_key_prefix keeps a
-- short, non-secret identifier for logs/admin output. api_key_revoked_at
-- disables a key without deleting its row.

ALTER TABLE organizations
    ADD COLUMN IF NOT EXISTS api_key_hash        TEXT,
    ADD COLUMN IF NOT EXISTS api_key_prefix      TEXT,
    ADD COLUMN IF NOT EXISTS api_key_created_at  TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS api_key_revoked_at  TIMESTAMPTZ;

CREATE UNIQUE INDEX IF NOT EXISTS organizations_api_key_hash_idx
    ON organizations (api_key_hash) WHERE api_key_hash IS NOT NULL;
