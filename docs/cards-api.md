# Cards API (`/cards`)

Pull API for organizations integrating with medkard: service-to-service
traffic over a private WireGuard tunnel, authenticated by a single bearer
API key scoped to specific organizations (`api/auth.py`). Every route takes
`?org=<name>` (case-insensitive) to name the target organization.

Routes live in `src/api/routes/cards.py`; smoke-test any of them with
`scripts/test-cards-route.sh` (mock data, no client needed).

## GET /cards/check

Cheap row-count check for a given date. Query params: `date` (`YYYY-MM-DD`),
`org`.

Returns JSON:

```json
{"date": "2026-07-01", "count": 42}
```

Integrating service typically compares `count` against how many rows it
last ingested, and calls `pull` again on a mismatch.

## GET /cards/pull

Returns the audited cards for a date as an xlsx workbook (not JSON) — the
integrating service stores the file and runs it through its own RAG
ingestion pipeline. Query params: `date`, `org`.

- `200` — `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`,
  `Content-Disposition: attachment; filename="report_<org>_<date>.xlsx"`
- `404` — no cards for that org/date

## GET /cards/export

Paginated JSON export of raw (not necessarily audited) card data. Query
params:

- `org` — required
- `since` — optional, only cards updated at/after this timestamp
- `limit` — max rows to return (`0` = no limit)
- `cursor` — pagination offset

Returns a JSON array of row dicts.

## POST /cards/push

*Not yet on `release`/`main` — implemented on `cards-push-endpoint` /
`dev`. Documented here for when it lands.*

Accepts a single updated card from a 1C org, to be merged into the next
nightly audit batch rather than waiting for the scheduled pull. Query
param: `org`. Body: the raw card JSON (same shape 1C sends elsewhere).

Validation:
- rejects (`422`) if `Прием.GUID` is missing
- rejects (`422`) if the card has none of `Пациент`/`Услуги`/`Диагнозы` —
  an empty shell rather than a real visit (empty values in those keys are
  still accepted; only total absence is rejected)

On success, upserts the card as pending (`storage/done_cards_storage.py`)
and returns:

```json
{"card_guid": "<guid>", "status": "pending"}
```
