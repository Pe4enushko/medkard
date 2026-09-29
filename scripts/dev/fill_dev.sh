#!/usr/bin/env bash
# Stage 2 of the dev stand: copy data from the source database into the dev one.
#
# Source — POSTGRES_* from .env (production, unless you point .env elsewhere).
# Target — POSTGRES_* from .env.dev, written by deploy_dev.sh.
#
# Reference tables are copied whole; done_cards and push_log only for the last N
# days, because the whole table is tens of thousands of audited cards and the
# stand needs a sample, not the archive.
#
# The copy runs through `psql \copy` rather than pg_dump on purpose: it needs no
# version match between the client and the two servers, and it is the only way
# to filter rows by date in one pass.
#
# NO ANONYMISATION HAPPENS HERE. done_cards.card_data carries ДанныеОсмотра as
# 1C sent it, which in practice includes the representative's ФИО and СНИЛС.
# Copying cards puts that on this machine; that is a deliberate choice, made
# when the стенд was designed, not an accident of this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# MEDKARD_SRC_ENV points the source elsewhere — a replica whose credentials you
# keep in a second file, or a local database when testing this script itself.
SRC_ENV="${MEDKARD_SRC_ENV:-$ROOT/.env}"
DST_ENV="$ROOT/.env.dev"

usage() {
    cat <<'EOF'
Usage: fill_dev.sh [DAYS]

DAYS — how many days of audited cards to copy (default 7). Reference tables are
copied in full regardless.

Reads the source from .env and the target from .env.dev. Refuses to run if the
two point at the same database.
EOF
}

DAYS="${1:-${DAYS:-7}}"
case "$DAYS" in
    -h|--help) usage; exit 0 ;;
    ''|*[!0-9]*) echo "ERROR: DAYS must be a whole number, got '$DAYS'" >&2; exit 2 ;;
esac

read_env() {  # $1 = file, $2 = key
    grep -E "^[[:space:]]*$2=" "$1" | head -1 | cut -d= -f2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/"
}

[[ -f "$SRC_ENV" ]] || { echo "ERROR: $SRC_ENV not found — the source database is read from it" >&2; exit 1; }
[[ -f "$DST_ENV" ]] || { echo "ERROR: .env.dev not found — run scripts/dev/deploy_dev.sh first" >&2; exit 1; }

SRC_HOST="$(read_env "$SRC_ENV" POSTGRES_HOST)"
SRC_PORT="$(read_env "$SRC_ENV" POSTGRES_PORT)"
SRC_DB="$(read_env "$SRC_ENV" POSTGRES_DB)"
SRC_USER="$(read_env "$SRC_ENV" POSTGRES_USER)"
SRC_PASSWORD="$(read_env "$SRC_ENV" POSTGRES_PASSWORD)"

DST_HOST="$(read_env "$DST_ENV" POSTGRES_HOST)"
DST_PORT="$(read_env "$DST_ENV" POSTGRES_PORT)"
DST_DB="$(read_env "$DST_ENV" POSTGRES_DB)"
DST_USER="$(read_env "$DST_ENV" POSTGRES_USER)"
DST_PASSWORD="$(read_env "$DST_ENV" POSTGRES_PASSWORD)"

: "${SRC_HOST:?POSTGRES_HOST missing in the source env file}"
: "${DST_HOST:?POSTGRES_HOST missing in .env.dev}"

if [[ "$SRC_HOST:$SRC_PORT/$SRC_DB" == "$DST_HOST:$DST_PORT/$DST_DB" ]]; then
    echo "ERROR: source and target are the same database ($SRC_HOST:$SRC_PORT/$SRC_DB)" >&2
    exit 1
fi

src() { PGPASSWORD="$SRC_PASSWORD" psql -h "$SRC_HOST" -p "${SRC_PORT:-5432}" -U "$SRC_USER" -d "$SRC_DB" -v ON_ERROR_STOP=1 "$@"; }
dst() { PGPASSWORD="$DST_PASSWORD" psql -h "$DST_HOST" -p "${DST_PORT:-5432}" -U "$DST_USER" -d "$DST_DB" -v ON_ERROR_STOP=1 "$@"; }

echo "source: $SRC_USER@$SRC_HOST:${SRC_PORT:-5432}/$SRC_DB  (read only)"
echo "target: $DST_USER@$DST_HOST:${DST_PORT:-5432}/$DST_DB"
echo "cards:  last $DAYS day(s)"
echo

# Order matters: a table comes after the ones it references.
REFERENCE_TABLES=(
    organizations
    api_keys
    api_key_organizations
    guidelines
    docs
    ingest_runs
    drugs
    dietary_supplements
    grls_imports
    grls_registry
)

copy_table() {  # $1 = table, $2 = SELECT statement
    local table="$1" query="$2" rows
    dst -q -c "TRUNCATE TABLE ${table} CASCADE;"
    src -q -c "\\copy (${query}) TO STDOUT" | dst -q -c "\\copy ${table} FROM STDIN"
    rows="$(dst -tA -c "SELECT count(*) FROM ${table}")"
    printf '  %-22s %10s rows\n' "$table" "$rows"
}

for table in "${REFERENCE_TABLES[@]}"; do
    if [[ "$(src -tA -c "SELECT to_regclass('public.${table}') IS NOT NULL")" != "t" ]]; then
        printf '  %-22s %10s\n' "$table" "absent in source, skipped"
        continue
    fi
    copy_table "$table" "SELECT * FROM ${table}"
done

# Audited cards and the push journal — windowed. COALESCE because an unfinished
# card has no finished_at, and a card can be re-audited (updated_at moves).
copy_table done_cards \
    "SELECT * FROM done_cards WHERE COALESCE(finished_at, updated_at, started_at) >= now() - interval '${DAYS} days'"

if [[ "$(src -tA -c "SELECT to_regclass('public.push_log') IS NOT NULL")" == "t" ]]; then
    copy_table push_log \
        "SELECT * FROM push_log WHERE pushed_at >= now() - interval '${DAYS} days'"
fi

echo
echo "dev stand filled:"
dst -tA -c "SELECT 'cards ' || count(*) FROM done_cards"
dst -tA -c "SELECT 'guideline chunks ' || count(*) FROM docs"
dst -tA -c "SELECT 'drug registry rows ' || count(*) FROM grls_registry"
