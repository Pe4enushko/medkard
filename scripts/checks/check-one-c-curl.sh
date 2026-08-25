#!/usr/bin/env bash
# Reproduce, with curl, the exact request scripts/audit-one-c-period.py sends to 1C.
#
# Mirrors integrations/one_c.py:
#   - Alenka: GET, query params datebegin=DD.MM.YYYY & dateend=DD.MM.YYYY
#     (same separator logic)
#   - MDS: POST {"date": "YYYY-MM-DD"}, one request per day of the period
#   - Basic auth from <ORG>_ONE_C_LOGIN / <ORG>_ONE_C_PASSWORD
#   - Accept: application/json
#
# Usage:
#   ./scripts/checks/check-one-c-curl.sh Alenka 03.06.2026 03.06.2026
#   ./scripts/checks/check-one-c-curl.sh MDS    01.06.2026 03.06.2026
#
# Reads the same env vars the Python client does. Load your .env first, e.g.:
#   set -a; source .env; set +a
#   ./scripts/checks/check-one-c-curl.sh Alenka 03.06.2026 03.06.2026

set -euo pipefail

ORG="${1:?Usage: $0 ORG DATEBEGIN DATEEND   (ORG = Alenka|MDS)}"
DATEBEGIN="${2:?Usage: $0 ORG DATEBEGIN DATEEND}"
DATEEND="${3:?Usage: $0 ORG DATEBEGIN DATEEND}"

case "$ORG" in
  Alenka) PREFIX="ALENKA"; REQUIRES_PASSWORD=1 ;;
  MDS)    PREFIX="MDS";    REQUIRES_PASSWORD=0 ;;
  *) echo "ORG must be Alenka or MDS, got: $ORG" >&2; exit 2 ;;
esac

URL="$(eval echo "\${${PREFIX}_ONE_C_APPOINTMENTS_URL:-}")"
LOGIN="$(eval echo "\${${PREFIX}_ONE_C_LOGIN:-}")"
PASSWORD="$(eval echo "\${${PREFIX}_ONE_C_PASSWORD:-}")"
TIMEOUT="$(eval echo "\${${PREFIX}_ONE_C_TIMEOUT_SECONDS:-15}")"

[ -n "$URL" ]   || { echo "${PREFIX}_ONE_C_APPOINTMENTS_URL is not set (source your .env first)" >&2; exit 1; }
[ -n "$LOGIN" ] || { echo "${PREFIX}_ONE_C_LOGIN is not set" >&2; exit 1; }
if [ "$REQUIRES_PASSWORD" = "1" ] && [ -z "$PASSWORD" ]; then
  echo "${PREFIX}_ONE_C_PASSWORD is not set" >&2; exit 1
fi

# DD.MM.YYYY -> YYYY-MM-DD
to_iso() {
  local d="$1"
  IFS=. read -r dd mm yyyy <<< "$d"
  echo "${yyyy}-${mm}-${dd}"
}

echo "ORG:        $ORG"
echo "URL (base): $URL"
echo "Login:      $LOGIN"

if [ "$ORG" = "MDS" ]; then
  # One POST per day of the period, same as MdsOneCClient.fetch_json_for_period.
  ISO_BEGIN="$(to_iso "$DATEBEGIN")"
  ISO_END="$(to_iso "$DATEEND")"
  day="$ISO_BEGIN"
  while [ "$(date -d "$day" +%s)" -le "$(date -d "$ISO_END" +%s)" ]; do
    echo "---"
    echo "Request:    POST $URL  body: {\"date\": \"$day\"}"
    curl -sS -i \
      --max-time "$TIMEOUT" \
      -u "${LOGIN}:${PASSWORD}" \
      -H "Accept: application/json" \
      -H "Content-Type: application/json" \
      -X POST "$URL" \
      -d "{\"date\": \"$day\"}"
    echo
    day="$(date -d "$day + 1 day" +%Y-%m-%d)"
  done
  exit 0
fi

# Same separator logic as one_c.py: '&' if the URL already has a '?', else '?'.
case "$URL" in
  *\?*) SEP="&" ;;
  *)    SEP="?" ;;
esac

echo "Request:    ${URL}${SEP}datebegin=${DATEBEGIN}&dateend=${DATEEND}"
echo "---"

# -u sends Basic auth exactly like base64(login:password) in one_c.py.
# -G + --data-urlencode mirrors urllib.parse.urlencode (dots stay literal).
curl -sS -i \
  --max-time "$TIMEOUT" \
  -u "${LOGIN}:${PASSWORD}" \
  -H "Accept: application/json" \
  -G "$URL" \
  --data-urlencode "datebegin=${DATEBEGIN}" \
  --data-urlencode "dateend=${DATEEND}"
echo
