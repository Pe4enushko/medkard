#!/usr/bin/env bash
# Ad-hoc smoke test for the pull API's /cards routes (src/api/routes/cards.py).
#
# Hits one route with mock query params so you can eyeball status code/body
# against a running `api.app:create_app` instance, without writing a client.
#
# Usage:
#   ./scripts/test-cards-route.sh ROUTE URL API_KEY [ORG] [DATE]
#
#   ROUTE   check | pull | export | push
#   URL     base URL of the API, e.g. http://localhost:8000
#   API_KEY bearer token (see scripts/create-api-key.py)
#   ORG     org name for ?org=          (default: MDS)
#   DATE    date for check/pull, DD.MM.YYYY not required — ISO YYYY-MM-DD
#           (default: today; ignored by push, which sends a mock card body)
#
# Examples:
#   ./scripts/test-cards-route.sh check  http://localhost:8000 $API_KEY
#   ./scripts/test-cards-route.sh pull   http://localhost:8000 $API_KEY Alenka 2026-07-01
#   ./scripts/test-cards-route.sh export http://localhost:8000 $API_KEY MDS
#   ./scripts/test-cards-route.sh push   http://localhost:8000 $API_KEY MDS

set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 ROUTE URL API_KEY [ORG] [DATE]

  ROUTE   check | pull | export | push
  URL     base URL of the API, e.g. http://localhost:8000
  API_KEY bearer token (see scripts/create-api-key.py)
  ORG     org name for ?org=          (default: MDS)
  DATE    date for check/pull, ISO YYYY-MM-DD (default: today)
          (push ignores DATE — it sends a mock card body instead)

Examples:
  $0 check  http://localhost:8000 \$API_KEY
  $0 pull   http://localhost:8000 \$API_KEY Alenka 2026-07-01
  $0 export http://localhost:8000 \$API_KEY MDS
  $0 push   http://localhost:8000 \$API_KEY MDS
EOF
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

ROUTE="${1:?Usage: $0 ROUTE URL API_KEY [ORG] [DATE]   (ROUTE = check|pull|export|push)}"
URL="${2:?Usage: $0 ROUTE URL API_KEY [ORG] [DATE]}"
API_KEY="${3:?Usage: $0 ROUTE URL API_KEY [ORG] [DATE]}"
ORG="${4:-MDS}"
DATE="${5:-$(date +%Y-%m-%d)}"

BASE_URL="${URL%/}"

# Mock card body for push: minimum viable shape per cards.py's
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
    REQUEST_URL="${BASE_URL}/cards/check"
    ARGS=(-G --data-urlencode "org=${ORG}" --data-urlencode "date=${DATE}")
    ;;
  pull)
    REQUEST_URL="${BASE_URL}/cards/pull"
    ARGS=(-G --data-urlencode "org=${ORG}" --data-urlencode "date=${DATE}")
    ;;
  export)
    REQUEST_URL="${BASE_URL}/cards/export"
    ARGS=(-G --data-urlencode "org=${ORG}" --data-urlencode "since=${DATE}T00:00:00" --data-urlencode "limit=10")
    ;;
  push)
    REQUEST_URL="${BASE_URL}/cards/push?org=$(urlencode_org "$ORG")"
    ARGS=(-X POST -H "Content-Type: application/json" --data-raw "$_MOCK_CARD_JSON")
    ;;
  *)
    echo "ROUTE must be check, pull, export or push, got: $ROUTE" >&2
    exit 2
    ;;
esac

echo "Route:   $ROUTE"
echo "Request: $REQUEST_URL"
echo "---"

curl -sS -i \
  --max-time 15 \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Accept: application/json" \
  "${ARGS[@]}" \
  "$REQUEST_URL"
echo
