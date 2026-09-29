#!/usr/bin/env bash
# Stage 1 of the dev stand: start the container, create the database, migrate it
# and remember the credentials in .env.dev. Idempotent — safe to re-run.
#
# Never touches .env: switching the checkout over is stage 3 (switch_env.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$ROOT/docker-compose.dev.yml"
DEV_ENV="$ROOT/.env.dev"

DB_NAME="${MEDKARD_DEV_DB:-medkard_dev}"
DB_USER="${MEDKARD_DEV_USER:-medkard_dev}"
DB_PORT="${MEDKARD_DEV_PORT:-55432}"
HEALTH_TIMEOUT_SECONDS="${MEDKARD_DEV_HEALTH_TIMEOUT:-120}"

usage() {
    cat <<'EOF'
Usage: deploy_dev.sh

Starts the local dev Postgres (docker-compose.dev.yml), creates the role and
database, enables the extensions, applies migrations/*.sql and writes the
connection parameters to .env.dev.

Environment overrides: MEDKARD_DEV_DB, MEDKARD_DEV_USER, MEDKARD_DEV_PORT,
MEDKARD_DEV_HEALTH_TIMEOUT.
EOF
}

for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

command -v docker >/dev/null || { echo "ERROR: docker not found" >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "ERROR: 'docker compose' not available" >&2; exit 1; }
command -v psql >/dev/null || { echo "ERROR: psql not found (install postgresql-client)" >&2; exit 1; }
command -v pg_isready >/dev/null || { echo "ERROR: pg_isready not found (install postgresql-client)" >&2; exit 1; }

echo "==> starting container"
MEDKARD_DEV_PORT="$DB_PORT" docker compose -f "$COMPOSE_FILE" up -d

# Superuser connection. Auth is trust (loopback only), so no password is needed
# here; the generated one below matters only to the application.
#
# Two flags keep this from hanging, and both were learned the hard way:
#   -w                  — psql otherwise waits for a password on a container
#                         whose data directory was initialised under another
#                         auth method, and waits without printing anything;
#   PGCONNECT_TIMEOUT   — docker's port proxy accepts the TCP connection before
#                         Postgres listens, so a client can sit in an
#                         established connection waiting for a handshake that
#                         nobody will send.
export PGCONNECT_TIMEOUT="${PGCONNECT_TIMEOUT:-5}"
admin() { psql -w -h 127.0.0.1 -p "$DB_PORT" -U postgres -d postgres -v ON_ERROR_STOP=1 "$@"; }

echo "==> waiting for the server to answer (up to ${HEALTH_TIMEOUT_SECONDS}s)"
# The probe is a real query, not pg_isready: an accepted TCP connection says
# nothing about a server that can answer, and the healthcheck says nothing about
# a container reused from an earlier compose file, where it does not exist.
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_SECONDS ))
until admin -tA -c 'SELECT 1' >/dev/null 2>&1; do
    if (( $(date +%s) > deadline )); then
        state="$(docker inspect -f '{{.State.Status}} health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' medkard-dev-db 2>/dev/null || echo unknown)"
        cat >&2 <<MSG

ERROR: no usable answer from 127.0.0.1:$DB_PORT after ${HEALTH_TIMEOUT_SECONDS}s
       (container: $state)

If the container is running, its data directory was most likely initialised
before POSTGRES_HOST_AUTH_METHOD=trust was in docker-compose.dev.yml — that
setting only applies when the directory is created, so postgres still wants a
password nobody has. There is nothing to lose in a fresh stand:

    docker compose -f docker-compose.dev.yml down -v
    bash scripts/dev/deploy_dev.sh

--- last 20 log lines ---
MSG
        docker logs --tail 20 medkard-dev-db >&2 2>&1 || true
        exit 1
    fi
    printf '.'
    sleep 2
done
echo " ready"

# A password is generated once and then reused: the role already exists on a
# re-run, and rewriting it would invalidate whatever .env already carries.
# `head` closing the pipe early makes the writer exit 141, and under
# `set -o pipefail` that kills the script without a word — hence awk and python,
# each a single process reading to its own end.
DB_PASSWORD="$(awk -F= '$1 == "POSTGRES_PASSWORD" { sub(/^[^=]*=/, ""); print; exit }' "$DEV_ENV" 2>/dev/null || true)"
if [[ -z "$DB_PASSWORD" ]]; then
    DB_PASSWORD="$(python3 -c 'import secrets, string; print("".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(24)))')"
fi

echo "==> role and database"
admin -q <<SQL
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${DB_USER}') THEN
        CREATE ROLE ${DB_USER} LOGIN;
    END IF;
END
\$\$;
ALTER ROLE ${DB_USER} WITH PASSWORD '${DB_PASSWORD}' CREATEDB;
SQL

if [[ "$(admin -tA -c "SELECT 1 FROM pg_database WHERE datname = '${DB_NAME}'")" != "1" ]]; then
    admin -q -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"
fi

# Extensions are created here, as superuser, because migration 001 would run as
# the application role: "uuid-ossp" is not trusted and CREATE EXTENSION would be
# refused. GRANT is for PG15+, where public has no CREATE on schema public.
psql -w -h 127.0.0.1 -p "$DB_PORT" -U postgres -d "$DB_NAME" -v ON_ERROR_STOP=1 -q \
    -c "CREATE EXTENSION IF NOT EXISTS vector;" \
    -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;" \
    -c 'CREATE EXTENSION IF NOT EXISTS "uuid-ossp";' \
    -c "GRANT ALL ON SCHEMA public TO ${DB_USER};"

cat > "$DEV_ENV" <<ENV
# Written by scripts/dev/deploy_dev.sh — the local dev stand, not a secret store.
# Stage 2 (fill_dev.sh) reads its target from here; stage 3 (switch_env.sh)
# copies these lines into .env.
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=${DB_PORT}
POSTGRES_DB=${DB_NAME}
POSTGRES_USER=${DB_USER}
POSTGRES_PASSWORD=${DB_PASSWORD}
ENV

echo "==> migrations"
MEDKARD_ENV_FILE="$DEV_ENV" bash "$ROOT/migrations/migrate.sh"

echo
echo "dev stand ready: postgresql://${DB_USER}@127.0.0.1:${DB_PORT}/${DB_NAME}"
echo "credentials written to .env.dev (.env untouched)"
