#!/usr/bin/env bash
# Truncates and reloads drugs and dietary_supplements tables from CSV source files.
# Usage: ./scripts/seed-reference-lists.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$ROOT_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env not found at $ENV_FILE" >&2
    exit 1
fi

while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// /}" ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    key="${key// /}"
    export "$key=$value"
done < "$ENV_FILE"

: "${POSTGRES_HOST:?POSTGRES_HOST not set in .env}"
: "${POSTGRES_PORT:=5432}"
: "${POSTGRES_DB:?POSTGRES_DB not set in .env}"
: "${POSTGRES_USER:?POSTGRES_USER not set in .env}"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD not set in .env}"

export PGPASSWORD="$POSTGRES_PASSWORD"

DRUGS_CSV="$ROOT_DIR/Drugs list.csv"
SUPPLEMENTS_CSV="$ROOT_DIR/Dietary supplements.csv"

for f in "$DRUGS_CSV" "$SUPPLEMENTS_CSV"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: source file not found: $f" >&2
        exit 1
    fi
done

psql_cmd() {
    psql \
        --host="$POSTGRES_HOST" \
        --port="$POSTGRES_PORT" \
        --dbname="$POSTGRES_DB" \
        --username="$POSTGRES_USER" \
        --set=ON_ERROR_STOP=1 \
        "$@"
}

DRUGS_SQL=$(mktemp /tmp/seed-drugs-XXXXXX.sql)
SUPPS_SQL=$(mktemp /tmp/seed-supps-XXXXXX.sql)
trap 'rm -f "$DRUGS_SQL" "$SUPPS_SQL"' EXIT

# ── drugs ─────────────────────────────────────────────────────────────────────
# drugs CSV: comma-separated, UTF-8, has header row.
# Columns: Торговое наименование, МНН, Лекарственная форма, Дозировка, Исключение отдельных групп пациентов

echo "Seeding drugs from '$DRUGS_CSV' ..."

cat > "$DRUGS_SQL" <<SQL
TRUNCATE TABLE drugs RESTART IDENTITY;

CREATE TEMP TABLE drugs_staging (
    trade_name         TEXT,
    inn_name           TEXT,
    dosage_form        TEXT,
    dosage             TEXT,
    patient_exclusions TEXT
);

\COPY drugs_staging FROM '$DRUGS_CSV' WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'UTF8', QUOTE '"')

INSERT INTO drugs (trade_name, inn_name, dosage_form, dosage, patient_exclusions)
SELECT
    NULLIF(TRIM(trade_name), ''),
    NULLIF(TRIM(inn_name), ''),
    NULLIF(TRIM(dosage_form), ''),
    NULLIF(TRIM(dosage), ''),
    NULLIF(TRIM(patient_exclusions), '')
FROM drugs_staging
WHERE TRIM(trade_name) <> '';
SQL

psql_cmd -f "$DRUGS_SQL"

# ── dietary_supplements ───────────────────────────────────────────────────────
# Dietary supplements CSV: semicolon-separated, UTF-8, has header row.
# Source column positions (1-based):
#   1  Номер свидетельства       → registration_number
#   2  Статус                    → status
#   4  Дата оформления документа → registered_at
#   5  Наименование продукции    → product_name
#   6  Наименование изготовителя → manufacturer_name
#   8  Страна изготовителя       → country_of_manufacture
#   13 Область применения        → scope_of_application
#   16 Информация на этикетке    → label_info

echo "Seeding dietary_supplements from '$SUPPLEMENTS_CSV' ..."

cat > "$SUPPS_SQL" <<SQL
TRUNCATE TABLE dietary_supplements RESTART IDENTITY;

CREATE TEMP TABLE supplements_staging (
    col01 TEXT, col02 TEXT, col03 TEXT, col04 TEXT, col05 TEXT,
    col06 TEXT, col07 TEXT, col08 TEXT, col09 TEXT, col10 TEXT,
    col11 TEXT, col12 TEXT, col13 TEXT, col14 TEXT, col15 TEXT,
    col16 TEXT, col17 TEXT, col18 TEXT, col19 TEXT, col20 TEXT,
    col21 TEXT, col22 TEXT, col23 TEXT, col24 TEXT, col25 TEXT,
    col26 TEXT, col27 TEXT, col28 TEXT, col29 TEXT, col30 TEXT,
    col31 TEXT, col32 TEXT, col33 TEXT, col34 TEXT, col35 TEXT,
    col36 TEXT, col37 TEXT, col38 TEXT, col39 TEXT
);

\COPY supplements_staging FROM '$SUPPLEMENTS_CSV' WITH (FORMAT csv, HEADER true, DELIMITER ';', ENCODING 'UTF8', QUOTE '"')

INSERT INTO dietary_supplements (
    registration_number, status, registered_at,
    product_name, manufacturer_name, country_of_manufacture,
    scope_of_application, label_info
)
SELECT
    NULLIF(TRIM(col01), ''),
    NULLIF(TRIM(col02), ''),
    NULLIF(TRIM(col04), '')::DATE,
    NULLIF(TRIM(col05), ''),
    NULLIF(TRIM(col06), ''),
    NULLIF(TRIM(col08), ''),
    NULLIF(TRIM(col13), ''),
    NULLIF(TRIM(col16), '')
FROM supplements_staging
WHERE TRIM(col05) <> '';
SQL

psql_cmd -f "$SUPPS_SQL"

echo "Done. Rows loaded:"
psql_cmd -c "SELECT 'drugs' AS tbl, COUNT(*) FROM drugs UNION ALL SELECT 'dietary_supplements', COUNT(*) FROM dietary_supplements;"
