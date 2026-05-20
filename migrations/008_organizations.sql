-- Migration 008: organizations table.

CREATE TABLE IF NOT EXISTS organizations (
    id         UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    name       TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
