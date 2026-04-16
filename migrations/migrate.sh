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

echo "Running migrations against $POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB ..."

for sql_file in "$SCRIPT_DIR"/[0-9]*.sql; do
    echo "  Applying $(basename "$sql_file") ..."
    psql \
        --host="$POSTGRES_HOST" \
        --port="$POSTGRES_PORT" \
        --dbname="$POSTGRES_DB" \
        --username="$POSTGRES_USER" \
        --file="$sql_file" \
        --set=ON_ERROR_STOP=1
done

echo "All migrations applied."
