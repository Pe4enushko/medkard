-- Migration 010: drugs and dietary supplements lookup tables with trigram search.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS drugs (
    id              SERIAL      PRIMARY KEY,
    trade_name      TEXT        NOT NULL,
    inn_name        TEXT,
    dosage_form     TEXT,
    dosage          TEXT,
    patient_exclusions TEXT
);

CREATE INDEX IF NOT EXISTS drugs_trade_name_trgm_idx
    ON drugs USING GIN (trade_name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS drugs_inn_name_trgm_idx
    ON drugs USING GIN (inn_name gin_trgm_ops);

CREATE TABLE IF NOT EXISTS dietary_supplements (
    id                      SERIAL      PRIMARY KEY,
    registration_number     TEXT,
    status                  TEXT,
    product_name            TEXT        NOT NULL,
    manufacturer_name       TEXT,
    country_of_manufacture  TEXT,
    scope_of_application    TEXT,
    label_info              TEXT,
    registered_at           DATE,
    product_name_tsv        TSVECTOR GENERATED ALWAYS AS (to_tsvector('russian', product_name)) STORED
);

CREATE INDEX IF NOT EXISTS dietary_supplements_product_name_tsv_idx
    ON dietary_supplements USING GIN (product_name_tsv);
