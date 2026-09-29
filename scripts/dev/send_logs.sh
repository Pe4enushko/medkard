#!/usr/bin/env bash
# Send e2e run logs from this machine to another one over scp.
#
# Run it where the tests ran; the alias is an ssh host from ~/.ssh/config, so no
# credentials live here. Files keep their names — they already carry a timestamp
# (e2e-YYYY-MM-DD_HH-MM-SS-PID.log), so nothing is overwritten on the far side.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOGS_DIR="$ROOT/logs"

ALIAS="remoteclaude"
DEST="projects/logs"
COUNT=3
ALL=0
WITH_TRACES=0

usage() {
    cat <<'EOF'
Usage: send_logs.sh [SSH_ALIAS] [--count N | --all] [--traces] [--dest PATH]

Copies the newest e2e logs from logs/ to SSH_ALIAS (default: remoteclaude).

  SSH_ALIAS     host from ~/.ssh/config, first positional argument
  --count N     how many newest e2e-*.log files to send (default 3)
  --all         send every e2e-*.log instead of the newest N
  --traces      also send logs/graphtraces.jsonl (it can be large)
  --dest PATH   target directory, relative to the remote home (default projects/logs)

Examples:
  scripts/dev/send_logs.sh                      # 3 newest logs to remoteclaude
  scripts/dev/send_logs.sh remoteclaude --all
  scripts/dev/send_logs.sh myhost --count 1 --traces
EOF
}

# First positional argument is the alias; flags may come in any order after it.
if [[ $# -gt 0 && "$1" != -* ]]; then
    ALIAS="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --count) COUNT="${2:?--count needs a number}"; shift 2 ;;
        --all) ALL=1; shift ;;
        --traces) WITH_TRACES=1; shift ;;
        --dest) DEST="${2:?--dest needs a path}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$COUNT" in
    ''|*[!0-9]*) echo "ERROR: --count must be a whole number, got '$COUNT'" >&2; exit 2 ;;
esac

[[ -d "$LOGS_DIR" ]] || { echo "ERROR: $LOGS_DIR does not exist — nothing to send" >&2; exit 1; }
command -v scp >/dev/null || { echo "ERROR: scp not found" >&2; exit 1; }

# `sort | awk` rather than `sort | head`: head closing the pipe makes the writer
# exit 141, and under `set -o pipefail` that would kill the script silently.
mapfile -t FILES < <(
    find "$LOGS_DIR" -maxdepth 1 -type f -name 'e2e-*.log' -printf '%T@\t%p\n' 2>/dev/null \
        | sort -nr \
        | awk -F'\t' -v limit="$COUNT" -v all="$ALL" 'all == 1 || NR <= limit { print $2 }'
)

if (( WITH_TRACES )); then
    [[ -f "$LOGS_DIR/graphtraces.jsonl" ]] && FILES+=("$LOGS_DIR/graphtraces.jsonl")
fi

if (( ${#FILES[@]} == 0 )); then
    echo "no e2e-*.log files in $LOGS_DIR — run e2e/run-diagnosis-graph-tests.sh first" >&2
    exit 1
fi

echo "to:    ${ALIAS}:${DEST}/"
for file in "${FILES[@]}"; do
    printf '  %s  %s\n' "$(du -h "$file" | cut -f1)" "$(basename "$file")"
done

# BatchMode: a wrong alias must fail with a message, not sit on a password prompt.
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$ALIAS" true 2>/dev/null; then
    echo "ERROR: cannot reach '$ALIAS' without interaction — check ~/.ssh/config and your key" >&2
    exit 1
fi

ssh -o BatchMode=yes "$ALIAS" "mkdir -p '$DEST'"
scp -p -q "${FILES[@]}" "${ALIAS}:${DEST}/"

echo "sent ${#FILES[@]} file(s)"
