# Analyst Replica Export — Medkard Design

**Date:** 2026-07-04
**Status:** Approved design, pending implementation plan
**Companion spec:** `engine` repo — `docs/superpowers/specs/2026-07-04-medkard-sql-analyst-engine-design.md`

## 1. Goal

The engine's medkard analyst is moving from consuming daily xlsx reports to
querying a **per-clinic replica** of audited-card data (full history, many
dates). This spec covers the **medkard-side changes** needed to feed that
replica: reliable change-tracking and an incremental row-export contract.

Medkard's production database is **not** re-partitioned — it stays a single DB
with org-scoped `done_cards`. The per-clinic database split happens on the
engine's replica. Medkard only needs to (a) make changes reliably discoverable
and (b) expose changed rows over the existing authenticated tunnel.

## 2. Scope

In scope (medkard):
- A change-tracking column so the engine can pull only what changed.
- An incremental JSON export endpoint (org-scoped, api-key auth).
- A full `card_guid` reconcile endpoint for the rare hard-delete case.

Out of scope (engine, see companion spec): the replica databases, per-clinic
roles, the sandbox tool, the `internal` network.

## 3. Change-tracking: `updated_at` on `done_cards`

`done_cards` has no reliable "last changed" marker today: `id` is a random
`uuid_generate_v4()` (not time-ordered), and `started_at`/`finished_at` are
NULL for ignored/broken cards. So a watermark needs a new column.

**Migration (`migrations/019_done_cards_updated_at.sql`):**
```sql
ALTER TABLE done_cards
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();

CREATE OR REPLACE FUNCTION done_cards_touch_updated_at()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS done_cards_set_updated_at ON done_cards;
CREATE TRIGGER done_cards_set_updated_at
    BEFORE INSERT OR UPDATE ON done_cards
    FOR EACH ROW EXECUTE FUNCTION done_cards_touch_updated_at();

CREATE INDEX IF NOT EXISTS done_cards_updated_at_idx ON done_cards (updated_at);
```

This bumps `updated_at` on every insert **and** update, regardless of card type
(audited / ignored / broken). The existing upserts in `done_cards_storage`
(`upsert`, `upsert_ignored`, `upsert_broken`) all pass through the trigger — no
call-site changes required.

**Redo semantics:** a redo is delete + re-insert with the **same `card_guid`**
(guids come from 1C, stable). The re-insert bumps `updated_at`, so the engine's
next incremental pull picks it up and overwrites its replica row by `card_guid`.

## 4. Export endpoint

Add to `src/api/routes/cards.py`, same auth and org-scoping as `pull`
(`Depends(require_org_access)` → `(org_id, org_name)`), delegating to a new
formatter method (routes stay DB-free, per the module contract).

```
GET /cards/export?org=<name>&since=<iso8601>&limit=<n=0>&cursor=<offset=0>
```

**Params:**
- `org` — **required**. Api-key resolves it to `organization_id`; only that org's
  rows are returned. Identical trust boundary to the existing `pull`.
- `since` — **optional** ISO8601 watermark. When set, returns rows where
  `updated_at > :since`. Omitted → all history (used by full export).
- `limit` — **optional, default `0` = no limit** (return everything matching in
  one response). Used by the daily pull.
- `cursor` — **optional, default `0`**, treated as an **OFFSET**. Only meaningful
  together with `limit > 0`; the full-export script increments it by `limit` each
  loop.

**Selection:**
```sql
SELECT ... FROM done_cards
WHERE organization_id = :org_id
  AND (:since IS NULL OR updated_at > :since)
ORDER BY updated_at, card_guid          -- stable order → correct OFFSET paging
[ LIMIT :limit OFFSET :cursor ]         -- only when limit > 0
```
The stable `ORDER BY` is required so `LIMIT/OFFSET` paging can't skip or duplicate
rows across pages.

**Response:** a plain JSON array of `done_cards` rows (pages are bounded, so no
streaming/NDJSON needed). Each object:
```
card_guid, card_data, formal_result, diag_result, icd_check_result,
token_count, time_ms, started_at, finished_at, ignored, broken,
stacktrace, updated_at, organization_name
```
JSONB columns are emitted as **native JSON** (not stringified) so the engine keeps
nested structure. `organization_name` is included so the engine routes each row to
the `medkard_<slug>` database without a second lookup.

**Two usage modes (engine side):**
- **Daily pull:** `?org=<slug>&since=<watermark>` with `limit=0` → one request
  returns the whole day's delta (~150 rows max). Engine's next watermark =
  `max(updated_at)` over the returned rows. Empty delta → `[]`.
- **Full export / backfill / resync:** the engine's full-export script loops
  `?org=<slug>&limit=5000&cursor=0`, then `cursor=5000`, `cursor=10000`, … until a
  page returns fewer than `limit` rows. (`since` omitted for a full backfill.)

### Formatter method
Add `ApiFormatter.export(org_id, since, limit, cursor) -> rows` (or a dedicated
`ExportFormatter`) issuing the query above via `done_cards_storage`. Add a storage
method `done_cards_storage.fetch_export(org_id, since, limit, cursor)` where
`limit == 0` means no `LIMIT/OFFSET` clause.

## 5. Reconcile endpoint (hard deletes / DR)

Incremental `updated_at` cannot signal a **hard delete** (e.g.
`delete_chinese_done_cards()`), which is rare. Rather than daily full scans,
expose the full guid set on demand for reconciliation:

```
GET /cards/guids?org=<name>
```
- Returns the complete set of `card_guid` for the org (reuse existing
  `done_cards_storage.get_done_guids(organization_id)`) as a plain JSON list.
- The engine's manual `medkard_replica_resync.py <slug>` fetches this and
  deletes replica rows whose guid is absent. Not part of the nightly path.

## 6. Auth / exposure notes

- No new trust boundary: `export`/`guids` reuse the existing api-key + `?org=`
  scoping used by `pull`. The engine calls them over the same WireGuard tunnel
  and `MedkardClient` credentials.
- **Postgres is not exposed** to the engine — export stays over HTTP. (The
  simpler-but-more-exposed alternative, a direct read-only DB role reachable over
  the tunnel, was considered and set aside to keep the DB unexposed and the
  api-key/org boundary intact. Revisit only if HTTP export proves too slow.)

## 7. Testing
- Trigger test: insert and update a `done_cards` row (via each upsert path,
  including ignored/broken) → `updated_at` advances each time.
- Export test: `since` filters correctly; `limit=0` returns the full matching set
  in one response; `limit>0` + `cursor` offset paging returns every row exactly
  once with no skips/dups across pages (stable `ORDER BY`); rows carry native
  JSONB and `organization_name`; org-scoping never leaks another org's rows.
- Reconcile test: `guids` returns the full org set; matches `get_done_guids`.

## 8. Phasing
1. `updated_at` migration + trigger + index.
2. `fetch_export` storage + `export` endpoint + tests.
3. `guids` reconcile endpoint.

## 9. Open items
- Default `limit` page size for the full-export loop (5000 is comfortable at
  ~150 MB/page; can go higher — see companion spec's headroom note). Daily uses
  `limit=0`.
- Whether `card_data` should be trimmed in export (kept **whole** for now — the
  analyst needs card content per the mixed metrics+drill-down requirement).
