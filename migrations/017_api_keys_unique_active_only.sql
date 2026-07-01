-- Migration 017: api_keys.key_hash uniqueness should only apply to active
-- (non-revoked) keys — otherwise a revoked key's hash can never be reissued,
-- even though it no longer authenticates anything.

DROP INDEX IF EXISTS api_keys_key_hash_idx;

CREATE UNIQUE INDEX IF NOT EXISTS api_keys_key_hash_idx
    ON api_keys (key_hash) WHERE revoked_at IS NULL;
