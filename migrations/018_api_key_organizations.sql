-- Migration 018: scope api_keys to specific organizations via a join table.
--
-- A key now only authorizes the orgs listed here (M:M) instead of every
-- org unconditionally — every key must have at least one row, enforced at
-- the application layer (scripts/create-api-key.py requires org names).

CREATE TABLE IF NOT EXISTS api_key_organizations (
    api_key_id      UUID NOT NULL REFERENCES api_keys(id)     ON DELETE CASCADE,
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    PRIMARY KEY (api_key_id, organization_id)
);
