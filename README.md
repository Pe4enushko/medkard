# medkard

LLM-based audit system for Russian ambulatory cards from 1C.

## Setup

Copy `.env.example` to `.env` and fill in Postgres, LLM, embedding, and 1C credentials.
Run migrations in `migrations/` against your Postgres DB.

## Scripts

### `audit-one-c-period.py`

Fetches visits from 1C for a period and audits them.

```
python scripts/audit-one-c-period.py {Alenka,MDS} [--days N] [--ignore-icd CODE ...] [--excel PATH]
```

- `{Alenka,MDS}` — 1C organization to fetch visits from
- `--days N` — start date is N days before today (default: today)
- `--ignore-icd Z00.0 J06.9 ...` — skip visits where every diagnosis is in this list
- `--excel PATH` — output xlsx file (default: `audit_results.xlsx`)

Set `DATEEND` at the top of the script to change the end date.
1C responses are cached in `data_snapshots/` and reused on re-runs.

### `audit-file.py`

Audits visits from a local JSON file.

```
python scripts/audit-file.py [--file PATH] [--excel PATH]
```

- `--file PATH` — input JSON file (default: `data.json`)
- `--excel PATH` — output xlsx file (default: `audit_results.xlsx`)

Input format: a dict with an `"appointments"` key, or `[{"appointments": [...]}]`.

## Restart safety

Done GUIDs are loaded from the DB before each run — interrupted runs resume without re-auditing completed cards.
