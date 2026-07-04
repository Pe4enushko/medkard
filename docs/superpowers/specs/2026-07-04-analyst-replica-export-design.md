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

## 4. Incremental export endpoint

Add to `src/api/routes/cards.py`, same auth and org-scoping as `pull`
(`Depends(require_org_access)` → `(org_id, org_name)`), delegating to a new
formatter method (routes stay DB-free, per the module contract).

```
GET /cards/export?org=<name>&since=<iso8601>&limit=<n>&after=<cursor>
```
- **Auth/scope:** api-key resolves `?org=` to `organization_id`; only that org's
  rows are returned. Identical trust boundary to the existing `pull`.
- **Selection:** rows where `updated_at > :since AND organization_id = :org_id`,
  ordered by `(updated_at, card_guid)` for a stable cursor.
- **Response:** NDJSON (one JSON object per line) so large deltas stream without
  buffering a giant array. Each line is one `done_cards` row:
  ```
  card_guid, card_data, formal_result, diag_result, icd_check_result,
  token_count, time_ms, started_at, finished_at, ignored, broken,
  stacktrace, updated_at, organization_name
  ```
  JSONB columns are emitted as native JSON (not stringified), so the engine
  keeps nested structure. `organization_name` is included so the engine can
  route the row to the `medkard_<slug>` database without a second lookup.
- **Pagination:** `limit` (default e.g. 5000) + `after=<last card_guid at last
  updated_at>` cursor. The engine loops until a short page. Response carries the
  max `updated_at` seen (header or final summary line) → the engine's next
  watermark.
- **Empty delta:** returns zero lines (normal on days with no changes for a
  clinic).

### Formatter method
Add `ApiFormatter.export_changed(org_id, since, limit, after) -> rows` (or a
dedicated `ExportFormatter`) issuing the windowed query above via
`done_cards_storage`. Add a storage method
`done_cards_storage.fetch_changed_since(org_id, since, limit, after)`.

## 5. Reconcile endpoint (hard deletes / DR)

Incremental `updated_at` cannot signal a **hard delete** (e.g.
`delete_chinese_done_cards()`), which is rare. Rather than daily full scans,
expose the full guid set on demand for reconciliation:

```
GET /cards/guids?org=<name>
```
- Returns the complete set of `card_guid` for the org (reuse existing
  `done_cards_storage.get_done_guids(organization_id)`), as NDJSON or a plain
  list.
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
- Export test: `since` filters correctly; NDJSON lines carry native JSONB and
  `organization_name`; cursor pagination returns every changed row exactly once;
  org-scoping never leaks another org's rows.
- Reconcile test: `guids` returns the full org set; matches `get_done_guids`.

## 8. Phasing
1. `updated_at` migration + trigger + index.
2. `fetch_changed_since` storage + `export` endpoint + tests.
3. `guids` reconcile endpoint.

## 9. Open items
- Page size / streaming defaults tuned to real delta sizes (one day's batch per
  clinic).
- Whether `card_data` should be trimmed in export (kept **whole** for now — the
  analyst needs card content per the mixed metrics+drill-down requirement).
