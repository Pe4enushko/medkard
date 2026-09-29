#!/usr/bin/env bash
# The one command to run: drives the three stages of the dev stand and owns
# every question, so the stage scripts stay argument-driven and usable alone.
#
#   1. deploy_dev.sh   — container, database, migrations, credentials in .env.dev
#   2. fill_dev.sh N   — data from the source database; N is asked here
#   3. switch_env.sh   — back .env up and point it at the stand
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_DAYS="${DAYS:-7}"

usage() {
    cat <<'EOF'
Usage: dev_stand.sh [--days N] [--no-switch]

Runs deploy → fill → switch. Without --days the number of days of audited cards
is asked interactively (default 7). --no-switch stops after filling and leaves
.env alone.
EOF
}

DAYS=""
SWITCH=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --days) DAYS="${2:?--days needs a number}"; shift 2 ;;
        --no-switch) SWITCH=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

echo "=== stage 1/3: deploy ==="
bash "$SCRIPT_DIR/deploy_dev.sh"

if [[ -z "$DAYS" ]]; then
    echo
    read -r -p "How many days of audited cards to copy? [$DEFAULT_DAYS] " DAYS
    DAYS="${DAYS:-$DEFAULT_DAYS}"
fi

echo
echo "=== stage 2/3: fill ==="
bash "$SCRIPT_DIR/fill_dev.sh" "$DAYS"

if (( ! SWITCH )); then
    echo
    echo "stopping before the switch; dev credentials are in .env.dev"
    exit 0
fi

echo
echo "=== stage 3/3: switch .env ==="
bash "$SCRIPT_DIR/switch_env.sh"

echo
echo "next: e2e/run-diagnosis-graph-tests.sh deterministic"
