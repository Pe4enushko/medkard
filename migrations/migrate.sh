#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$(dirname "$SCRIPT_DIR")/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env not found at $ENV_FILE" >&2
    exit 1
fi

# Parse key=value pairs from .env, skipping comments and blank lines
while IFS='=' read -r key value; do
    [[ "$key" =~ ^[[:space:]]*# ]] && continue
    [[ -z "$key" ]] && continue
    key="${key// /}"
    export "$key=$value"
done < "$ENV_FILE"

: "${POSTGRES_HOST:?POSTGRES_HOST not set in .env}"
: "${POSTGRES_PORT:=5432}"
: "${POSTGRES_DB:?POSTGRES_DB not set in .env}"
: "${POSTGRES_USER:?POSTGRES_USER not set in .env}"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD not set in .env}"

export PGPASSWORD="$POSTGRES_PASSWORD"

SKIP_UNTIL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-until) SKIP_UNTIL="${2:?--skip-until needs a filename}"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

echo "Running migrations against $POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB ..."

run_psql() {
    psql \
        --host="$POSTGRES_HOST" \
        --port="$POSTGRES_PORT" \
        --dbname="$POSTGRES_DB" \
        --username="$POSTGRES_USER" \
        --set=ON_ERROR_STOP=1 \
        "$@"
}

# Ledger of applied migrations. Bootstrapped here because it cannot record itself.
run_psql -c "CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);"

is_applied() {  # $1 = basename
    [[ "$(run_psql -tA -c "SELECT 1 FROM schema_migrations WHERE filename = '$1'")" == "1" ]]
}

record_only() {  # $1 = basename — mark applied without running the file
    run_psql -c "INSERT INTO schema_migrations(filename) VALUES ('$1') ON CONFLICT (filename) DO NOTHING;"
}

apply_and_record() {  # $1 = path, $2 = basename — file + ledger row in ONE transaction
    run_psql --single-transaction -f "$1" \
        -c "INSERT INTO schema_migrations(filename) VALUES ('$2');"
}

for sql_file in "$SCRIPT_DIR"/[0-9]*.sql; do
    name="$(basename "$sql_file")"
    if [[ -n "$SKIP_UNTIL" && "$name" < "$SKIP_UNTIL" ]]; then
        echo "  baseline (record only) $name"
        record_only "$name"
        continue
    fi
    if is_applied "$name"; then
        echo "  skip (already applied) $name"
        continue
    fi
    echo "  Applying $name ..."
    apply_and_record "$sql_file" "$name"
done

echo "All migrations applied."
