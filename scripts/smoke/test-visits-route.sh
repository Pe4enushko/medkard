#!/usr/bin/env bash
# Ad-hoc smoke test for the pull API's /visits routes (src/api/routes/visits.py).
#
# Hits one route with mock query params so you can eyeball status code/body
# against a running `api.app:create_app` instance, without writing a client.
#
# Usage:
#   ./scripts/smoke/test-visits-route.sh ROUTE URL API_KEY [ORG] [DATE] [DOCTOR_CODE]
#
#   ROUTE   check | pull | export | push | check_updates | doctors
#   URL     base URL of the API, e.g. http://localhost:8000
#   API_KEY bearer token (see scripts/operator/create-api-key.py)
#   ORG     org name for ?org=          (default: MDS)
#   DATE    date for check/pull, DD.MM.YYYY not required — ISO YYYY-MM-DD
#           (default: today; ignored by push, which sends a mock card body,
#           and by doctors, which takes org only;
#           for check_updates it becomes ?since=DATE"T00:00:00")
#   DOCTOR_CODE  pull only — Прием.Врач_код, narrows the report to one doctor.
#           Codes come from the doctors route. With no cards for that doctor
#           the API returns 200 and a one-cell placeholder workbook, not 404;
#           a code outside [\w-]{1,64} is rejected with 422.
#
# Examples:
#   ./scripts/smoke/test-visits-route.sh check  http://localhost:8000 $API_KEY
#   ./scripts/smoke/test-visits-route.sh pull   http://localhost:8000 $API_KEY Alenka 2026-07-01
#   ./scripts/smoke/test-visits-route.sh pull   http://localhost:8000 $API_KEY MDS 2026-07-01 00012
#   ./scripts/smoke/test-visits-route.sh export http://localhost:8000 $API_KEY MDS
#   ./scripts/smoke/test-visits-route.sh push   http://localhost:8000 $API_KEY MDS
#   ./scripts/smoke/test-visits-route.sh check_updates http://localhost:8000 $API_KEY MDS 2026-07-20
#   ./scripts/smoke/test-visits-route.sh doctors http://localhost:8000 $API_KEY MDS

set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 ROUTE URL API_KEY [ORG] [DATE] [DOCTOR_CODE]

  ROUTE   check | pull | export | push | check_updates | doctors
  URL     base URL of the API, e.g. http://localhost:8000
  API_KEY bearer token (see scripts/operator/create-api-key.py)
  ORG     org name for ?org=          (default: MDS)
  DATE    date for check/pull, ISO YYYY-MM-DD (default: today)
          (push ignores DATE — it sends a mock card body instead;
          doctors ignores DATE — it takes org only;
          check_updates turns DATE into ?since=DATE"T00:00:00")
  DOCTOR_CODE  pull only — Прием.Врач_код from the doctors route. Narrows the
          report to one doctor; no cards for them still yields 200 with a
          one-cell placeholder workbook, and a malformed code yields 422.

Examples:
  $0 check  http://localhost:8000 \$API_KEY
  $0 pull   http://localhost:8000 \$API_KEY Alenka 2026-07-01
  $0 pull   http://localhost:8000 \$API_KEY MDS 2026-07-01 00012
  $0 export http://localhost:8000 \$API_KEY MDS
  $0 push   http://localhost:8000 \$API_KEY MDS
  $0 check_updates http://localhost:8000 \$API_KEY MDS 2026-07-20
  $0 doctors http://localhost:8000 \$API_KEY MDS
EOF
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

ROUTE="${1:?Usage: $0 ROUTE URL API_KEY [ORG] [DATE] [DOCTOR_CODE]   (ROUTE = check|pull|export|push|check_updates|doctors)}"
URL="${2:?Usage: $0 ROUTE URL API_KEY [ORG] [DATE] [DOCTOR_CODE]}"
API_KEY="${3:?Usage: $0 ROUTE URL API_KEY [ORG] [DATE] [DOCTOR_CODE]}"
ORG="${4:-MDS}"
DATE="${5:-$(date +%Y-%m-%d)}"
DOCTOR_CODE="${6:-}"

BASE_URL="${URL%/}"

# Mock card body for push: minimum viable shape per visits.py's
# _extract_card_guid + _looks_like_a_visit (needs Прием.GUID and at least
# one of Пациент/Услуги/Диагнозы).
_MOCK_CARD_JSON='{
  "Прием": {"GUID": "00000000-0000-0000-0000-000000000001"},
  "Пациент": {"ФИО": "Тестов Тест Тестович", "ДатаРождения": "01.01.1990"},
  "Услуги": [],
  "Диагнозы": []
}'

# curl -G would also pull --data-raw's JSON body into the query string, so
# push (a POST with a body) has to urlencode `org` itself instead of relying
# on -G. Plain-ASCII org names need no escaping beyond spaces; this covers
# the mock orgs (MDS, Alenka) without pulling in a urlencode dependency.
urlencode_org() {
  local out="" c
  for (( i=0; i<${#1}; i++ )); do
    c="${1:$i:1}"
    case "$c" in
      [a-zA-Z0-9.~_-]) out+="$c" ;;
      *) out+=$(printf '%%%02X' "'$c") ;;
    esac
  done
  printf '%s' "$out"
}

case "$ROUTE" in
  check)
    REQUEST_URL="${BASE_URL}/visits/check"
    ARGS=(-G --data-urlencode "org=${ORG}" --data-urlencode "date=${DATE}")
    ;;
  pull)
    REQUEST_URL="${BASE_URL}/visits/pull"
    ARGS=(-G --data-urlencode "org=${ORG}" --data-urlencode "date=${DATE}")
    # Omit the param entirely when unset: an empty doctor_code is a 422, not
    # "no filter".
    if [[ -n "$DOCTOR_CODE" ]]; then
      ARGS+=(--data-urlencode "doctor_code=${DOCTOR_CODE}")
    fi
    ;;
  export)
    REQUEST_URL="${BASE_URL}/visits/export"
    ARGS=(-G --data-urlencode "org=${ORG}" --data-urlencode "since=${DATE}T00:00:00" --data-urlencode "limit=10")
    ;;
  push)
    REQUEST_URL="${BASE_URL}/visits/push?org=$(urlencode_org "$ORG")"
    ARGS=(-X POST -H "Content-Type: application/json" --data-raw "$_MOCK_CARD_JSON")
    ;;
  check_updates)
    # since is optional server-side (defaults to the last week) — passed here
    # so DATE stays meaningful; statuses returned: done|pending only.
    REQUEST_URL="${BASE_URL}/visits/check_updates"
    ARGS=(-G --data-urlencode "org=${ORG}" --data-urlencode "since=${DATE}T00:00:00")
    ;;
  doctors)
    # No date: the listing spans the org's whole card history.
    REQUEST_URL="${BASE_URL}/visits/doctors"
    ARGS=(-G --data-urlencode "org=${ORG}")
    ;;
  *)
    echo "ROUTE must be check, pull, export, push, check_updates or doctors, got: $ROUTE" >&2
    exit 2
    ;;
esac

echo "Route:   $ROUTE"
echo "Request: $REQUEST_URL"
# -G folds the params into the URL only inside curl, so the filter would be
# invisible in the line above — the one thing worth eyeballing on this route.
if [[ "$ROUTE" == "pull" && -n "$DOCTOR_CODE" ]]; then
  echo "Filter:  doctor_code=${DOCTOR_CODE}"
fi
echo "---"

curl -sS -i \
  --max-time 15 \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Accept: application/json" \
  "${ARGS[@]}" \
  "$REQUEST_URL"
echo
