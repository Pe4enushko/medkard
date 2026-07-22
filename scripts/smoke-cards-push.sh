#!/usr/bin/env bash
# End-to-end smoke test for the cards-push feature (POST /cards/push → nightly audit).
#
# Unlike scripts/test-cards-route.sh (which fires one request for you to eyeball),
# this drives the whole push→audit cycle against a live API + the configured
# Postgres and prints a PASS/FAIL verdict per check. Exit code 0 = all passed.
#
# Checks, in order:
#   1. push without auth            → 401/403
#   2. push without Прием.GUID      → 422
#   3. push of an empty shell       → 422   (none of Пациент/Услуги/Диагнозы)
#   4. push of a real card          → 200, row lands status='pending', results NULL
#   5. re-push with new data        → card_data replaced, still exactly one row
#   6. push over an audited card    → status back to 'pending', audit columns wiped
#   7. --pending-only audit run     → card becomes status='done'   [--with-audit]
#
# Usage:
#   ./scripts/smoke-cards-push.sh URL API_KEY [ORG] [--with-audit]
#
#   URL           base URL of the API, e.g. http://localhost:8000
#   API_KEY       bearer token (see scripts/create-api-key.py)
#   ORG           org name for ?org=   (default: MDS)
#   --with-audit  also run step 7, which invokes
#                 `python scripts/audit-one-c-period.py ORG --pending-only`.
#                 That burns real LLM tokens, so it is opt-in.
#
# The test card uses a dedicated GUID (see TEST_GUID) and is deleted from
# done_cards on exit, including on failure.

set -euo pipefail

usage() {
  sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

URL="${1:?Usage: $0 URL API_KEY [ORG] [--with-audit]}"
API_KEY="${2:?Usage: $0 URL API_KEY [ORG] [--with-audit]}"
ORG="MDS"
WITH_AUDIT=0
shift 2
for arg in "$@"; do
  case "$arg" in
    --with-audit) WITH_AUDIT=1 ;;
    *) ORG="$arg" ;;
  esac
done

BASE_URL="${URL%/}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Fixed, obviously-synthetic GUID so a leftover row is recognisable and the
# cleanup below can always find it. Valid hex — Прием.GUID is stored as text,
# but a malformed one would be indistinguishable from corrupt 1C data.
TEST_GUID="5406c6de-0000-4000-8000-000000000001"

PASS=0
FAIL=0

ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; PASS=$((PASS + 1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAIL=$((FAIL + 1)); }
step() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# --- DB access ---------------------------------------------------------------
# Reuses the app's own .env + psycopg rather than requiring a psql binary, so
# this runs anywhere the audit scripts already run.
psql_json() {
  python3 - "$1" <<'PY'
import json, os, sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg

root = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
load_dotenv(Path(os.environ["SMOKE_ROOT"]) / ".env")
conninfo = (
    f"host={os.environ['POSTGRES_HOST']} "
    f"port={os.environ.get('POSTGRES_PORT', '5432')} "
    f"dbname={os.environ['POSTGRES_DB']} "
    f"user={os.environ['POSTGRES_USER']} "
    f"password={os.environ['POSTGRES_PASSWORD']}"
)
with psycopg.connect(conninfo, autocommit=True, row_factory=psycopg.rows.dict_row) as conn:
    cur = conn.execute(sys.argv[1], {"g": os.environ["SMOKE_GUID"]})
    rows = cur.fetchall() if cur.description else []
print(json.dumps(rows, default=str, ensure_ascii=False))
PY
}

export SMOKE_ROOT="$ROOT"
export SMOKE_GUID="$TEST_GUID"

cleanup() {
  psql_json "DELETE FROM done_cards WHERE card_guid = %(g)s" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Prints "[]" if the DB is unreachable, so the assertions below fail loudly
# instead of set -e aborting the run before any verdict is printed.
card_row() {
  psql_json "SELECT status, card_data, formal_result, diag_result, icd_check_result, ignored, broken FROM done_cards WHERE card_guid = %(g)s" 2>/tmp/.smoke_db.err || {
    printf '\033[31m  DB query failed:\033[0m %s\n' "$(tail -1 /tmp/.smoke_db.err)" >&2
    echo '[]'
  }
}

# --- HTTP helper -------------------------------------------------------------
# Prints "STATUS<TAB>BODY". Auth header omitted entirely when $2 is "noauth".
# A connection failure must surface as a visible FAIL, not as an empty status
# that every expect_status then reports as a mismatch against a blank value —
# so curl's own failure is caught here and turned into status "000".
push() {
  local body="$1" auth="${2:-auth}" code="" bodyfile
  local -a hdr=(-H "Content-Type: application/json")
  [[ "$auth" == "auth" ]] && hdr+=(-H "Authorization: Bearer ${API_KEY}")
  bodyfile="$(mktemp)"
  if ! code="$(curl -sS --max-time 20 -o "$bodyfile" -w '%{http_code}' \
      -X POST "${hdr[@]}" --data-raw "$body" \
      "${BASE_URL}/cards/push?org=${ORG}" 2>"${bodyfile}.err")"; then
    code="000"
    printf '000\tconnection failed: %s' "$(tr -d '\n' <"${bodyfile}.err" | head -c 200)"
    rm -f "$bodyfile" "${bodyfile}.err"
    return 0
  fi
  printf '%s\t%s' "$code" "$(cat "$bodyfile")"
  rm -f "$bodyfile" "${bodyfile}.err"
}

status_of() { cut -f1 <<<"$1"; }
body_of()   { cut -f2- <<<"$1"; }

# Runs a python assertion snippet with the JSON row set on stdin.
# Kept as a function because `python3 <<'PY' <<<"$row"` silently feeds the
# script the row instead of the code — two stdin redirects, last one wins.
assert_row() {
  local rows="$1" label="$2" code="$3"
  if printf '%s' "$rows" | python3 -c "$code" 2>/tmp/.smoke_assert.err; then
    ok "$label"
  else
    bad "$label — $(tail -1 /tmp/.smoke_assert.err)"
  fi
}

expect_status() {
  local got want label
  got="$(status_of "$1")"; want="$2"; label="$3"
  if [[ "$got" == "$want" ]]; then
    ok "$label (HTTP $got)"
  else
    bad "$label — expected HTTP $want, got $got: $(body_of "$1" | head -c 200)"
  fi
}

CARD_V1=$(cat <<EOF
{"Прием": {"GUID": "$TEST_GUID", "DATE": "$(date +%d.%m.%Y)"},
 "Пациент": {"ФИО": "Смоук Тестович", "ДатаРождения": "01.01.1990"},
 "Услуги": [], "Диагнозы": [], "smoke_version": 1}
EOF
)
CARD_V2=$(sed 's/"smoke_version": 1/"smoke_version": 2/' <<<"$CARD_V1")

echo "API:  ${BASE_URL}   org=${ORG}"
echo "GUID: ${TEST_GUID}"
cleanup

step "1-3. Rejections"
# Either 401 or 403 is acceptable here — which one depends on whether the
# dependency rejects the missing header or the empty credential.
r="$(push "$CARD_V1" noauth)"; s="$(status_of "$r")"
if [[ "$s" == "401" || "$s" == "403" ]]; then
  ok "unauthenticated push is rejected (HTTP $s)"
else
  bad "unauthenticated push — expected 401/403, got $s"
fi

expect_status "$(push '{"Прием": {}, "Пациент": {}}')" 422 "push without Прием.GUID"
expect_status "$(push "{\"Прием\": {\"GUID\": \"$TEST_GUID\"}}")" 422 "push of an empty shell"

step "4. Push a real card"
r="$(push "$CARD_V1")"
expect_status "$r" 200 "push accepted"
if [[ "$(body_of "$r")" == *'"status":"pending"'* ]]; then
  ok "response says status=pending"
else
  bad "response body lacks status=pending: $(body_of "$r" | head -c 200)"
fi

row="$(card_row)"
n="$(python3 -c 'import json,sys; print(len(json.load(sys.stdin)))' <<<"$row" 2>/dev/null || echo 0)"
if [[ "$n" == "1" ]]; then ok "exactly one done_cards row"; else bad "expected 1 row, found $n"; fi
assert_row "$row" "row is pending with audit columns NULL" '
import json, sys
rows = json.load(sys.stdin)
assert rows, "no row in done_cards"
r = rows[0]
assert r["status"] == "pending", "status=" + str(r["status"])
assert r["formal_result"] is None and r["diag_result"] is None, "audit results not NULL"
assert r["ignored"] is False and r["broken"] is False, "flags not reset"
'

step "5. Re-push replaces card_data"
expect_status "$(push "$CARD_V2")" 200 "second push accepted"
row="$(card_row)"
assert_row "$row" "card_data replaced (smoke_version=2), still one row" '
import json, sys
rows = json.load(sys.stdin)
assert len(rows) == 1, str(len(rows)) + " rows, expected 1"
got = rows[0]["card_data"].get("smoke_version")
assert got == 2, "smoke_version=" + str(got) + ", expected 2"
'

step "6. Push over an audited card resets it to pending"
psql_json "UPDATE done_cards SET status='done', formal_result='{\"findings\": []}'::jsonb, ignored=TRUE WHERE card_guid = %(g)s" >/dev/null 2>&1 \
  || bad "could not stage an audited row (DB unreachable?) — step 6 result is meaningless"
expect_status "$(push "$CARD_V1")" 200 "push over a done card accepted"
row="$(card_row)"
assert_row "$row" "status back to pending, audit columns wiped" '
import json, sys
rows = json.load(sys.stdin)
assert rows, "no row in done_cards"
r = rows[0]
assert r["status"] == "pending", "status=" + str(r["status"])
assert r["formal_result"] is None, "formal_result survived the push"
assert r["ignored"] is False, "ignored flag survived the push"
'

if [[ "$WITH_AUDIT" == "1" ]]; then
  step "7. Nightly audit picks the pending card up (--pending-only)"
  echo "  running audit-one-c-period.py ${ORG} --pending-only (real LLM calls)..."
  if (cd "$ROOT" && python3 scripts/audit-one-c-period.py "$ORG" --pending-only -y >/tmp/.smoke_audit.log 2>&1); then
    row="$(card_row)"
    assert_row "$row" "card audited: status=done" '
import json, sys
rows = json.load(sys.stdin)
assert rows, "row vanished from done_cards"
r = rows[0]
assert r["status"] == "done", "status=" + str(r["status"]) + " (see /tmp/.smoke_audit.log)"
'
  else
    bad "audit run failed — see /tmp/.smoke_audit.log"
    tail -15 /tmp/.smoke_audit.log | sed 's/^/    /'
  fi
else
  step "7. Nightly audit — SKIPPED (pass --with-audit to run it; costs LLM tokens)"
fi

step "Result: ${PASS} passed, ${FAIL} failed"
[[ "$FAIL" == "0" ]]
