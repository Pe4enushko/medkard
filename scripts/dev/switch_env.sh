#!/usr/bin/env bash
# Stage 3 of the dev stand: point .env at the dev database.
#
# Only the POSTGRES_* lines are rewritten — LLM keys, embedding settings and 1C
# credentials stay as they are. The previous .env is kept as a timestamped
# backup, so going back is a copy, not a re-fill.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$ROOT/.env"
DEV_ENV="$ROOT/.env.dev"

ASSUME_YES=0

usage() {
    cat <<'EOF'
Usage: switch_env.sh [-y|--yes]

Backs .env up and replaces its POSTGRES_* lines with the ones from .env.dev.
Asks first unless -y is given (the orchestrator asks on its behalf).
EOF
}

for arg in "$@"; do
    case "$arg" in
        -y|--yes) ASSUME_YES=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

[[ -f "$DEV_ENV" ]] || { echo "ERROR: .env.dev not found — run scripts/dev/deploy_dev.sh first" >&2; exit 1; }
[[ -f "$ENV_FILE" ]] || { echo "ERROR: .env not found — copy .env.example first" >&2; exit 1; }

read_dev() {  # one awk, no pipeline: see deploy_dev.sh
    awk -F= -v key="$1" '
        /^[[:space:]]*#/ { next }
        { name = $1; gsub(/[[:space:]]/, "", name) }
        name == key { sub(/^[^=]*=/, ""); gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print; exit }
    ' "$DEV_ENV"
}

DEV_HOST="$(read_dev POSTGRES_HOST)"
DEV_PORT="$(read_dev POSTGRES_PORT)"
DEV_DB="$(read_dev POSTGRES_DB)"
DEV_USER="$(read_dev POSTGRES_USER)"
DEV_PASSWORD="$(read_dev POSTGRES_PASSWORD)"
: "${DEV_HOST:?POSTGRES_HOST missing in .env.dev}"

CURRENT="$(grep -E '^[[:space:]]*POSTGRES_(HOST|PORT|DB)=' "$ENV_FILE" | tr '\n' ' ' || true)"
echo "current .env: ${CURRENT:-no POSTGRES_* lines}"
echo "dev stand:    POSTGRES_HOST=$DEV_HOST POSTGRES_PORT=$DEV_PORT POSTGRES_DB=$DEV_DB"

if (( ! ASSUME_YES )); then
    read -r -p "Back .env up and switch it to the dev stand? [Y/n] " answer
    case "${answer:-Y}" in
        [Yy]|[Yy][Ee][Ss]|"") ;;
        *) echo "left .env alone; the dev credentials stay in .env.dev"; exit 0 ;;
    esac
fi

BACKUP="$ENV_FILE.bak.$(date +%Y-%m-%d-%H%M%S)"
cp -p "$ENV_FILE" "$BACKUP"

# Rewrite in place: replace the five keys, append the ones that were absent.
python3 - "$ENV_FILE" "$DEV_HOST" "$DEV_PORT" "$DEV_DB" "$DEV_USER" "$DEV_PASSWORD" <<'PY'
import re
import sys

path, host, port, db, user, password = sys.argv[1:7]
wanted = {
    "POSTGRES_HOST": host,
    "POSTGRES_PORT": port,
    "POSTGRES_DB": db,
    "POSTGRES_USER": user,
    "POSTGRES_PASSWORD": password,
}
lines = open(path, encoding="utf-8").read().splitlines()
seen = set()
out = []
for line in lines:
    match = re.match(r"\s*([A-Z0-9_]+)\s*=", line)
    key = match.group(1) if match else None
    if key in wanted:
        out.append(f"{key}={wanted[key]}")
        seen.add(key)
    else:
        out.append(line)
for key, value in wanted.items():
    if key not in seen:
        out.append(f"{key}={value}")
open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
PY

echo ".env now points at the dev stand; previous copy: ${BACKUP##*/}"
echo "to go back: cp ${BACKUP##*/} .env"
