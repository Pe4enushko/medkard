# Medkard Analyst Replica Export — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the engine a way to pull medkard's audited-card rows incrementally (daily delta) and in full (backfill), so it can populate a per-clinic analyst replica.

**Architecture:** Add an `updated_at` change-tracking trigger to `done_cards` (the watermark) and a `GET /cards/export` endpoint that returns rows as JSON (org-scoped, api-key auth) for both the daily delta (`since` + `limit=0`) and full backfill/resync (`limit`+`cursor` paging). Reads flow through `ApiFormatter` so route handlers stay parsing/auth-only, matching the existing `check`/`pull` routes.

**Tech Stack:** Python 3, FastAPI, psycopg3 (`AsyncConnectionPool`, `dict_row`), PostgreSQL, pytest (`asyncio_mode=auto`), plain SQL migrations run by `migrations/migrate.sh`.

**Companion spec:** `docs/superpowers/specs/2026-07-04-analyst-replica-export-design.md`
**Consumer:** the engine's sync job (separate repo/plan) calls these endpoints.

## Global Constraints

- `pythonpath = src` (from `pytest.ini`); imports are `from api.app import create_app`, `from storage.base import ...`, etc. — no `src.` prefix.
- Storage uses **psycopg3** with `row_factory=dict_row`; SQL params are named `%(name)s`; rows are `dict`. Connections come from the shared pool via `async with self._pool.connection() as conn`.
- Migrations are numbered `NNN_description.sql` in `migrations/`, applied in order by `migrations/migrate.sh` (idempotent SQL — `IF NOT EXISTS` / `CREATE OR REPLACE`). Next number is **019** (last is `018_api_key_organizations.sql`).
- Tests hit the **real configured Postgres** (`.env` at repo root, `load_dotenv(ROOT/".env")`) via FastAPI `TestClient`. Org is selected per-request with `?org=<name>` plus a `Bearer` api key; api-key fixtures come from `ApiKeysStorage`/`OrganizationsStorage` (see `tests/test_cards_api.py`). Fixtures that create keys are **function-scoped**.
- Export returns **audited cards only** (`ignored = FALSE AND broken = FALSE`) and **only these six columns**: `card_guid, card_data, formal_result, diag_result, icd_check_result, updated_at`. JSONB columns are emitted as **native JSON** (select raw `jsonb`, not `::text`). No `token_count`/`time_ms`/`started_at`/`finished_at`/`ignored`/`broken`/`stacktrace`/org identifier. `limit == 0` means **no `LIMIT/OFFSET`** (one-shot daily). `ORDER BY updated_at, card_guid` is required for correct offset paging.
- Card content (`card_data`) is exported **whole** (not trimmed).

---

### Task 1: `updated_at` change-tracking on `done_cards`

**Files:**
- Create: `migrations/022_done_cards_updated_at.sql`
- Test: `tests/test_done_cards_updated_at.py`

**Interfaces:**
- Consumes: nothing.
- Produces: column `done_cards.updated_at TIMESTAMPTZ NOT NULL` that advances to transaction `now()` on every INSERT and UPDATE; index `done_cards_updated_at_idx`.

- [ ] **Step 1: Write the failing test**

`tests/test_done_cards_updated_at.py`:
```python
"""
Verifies the done_cards.updated_at trigger (migration 019): updated_at is set
on insert and advances on update. Hits the real configured Postgres.
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

import psycopg
import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")


def _conninfo() -> str:
    return (
        f"host={os.environ['POSTGRES_HOST']} "
        f"port={os.environ.get('POSTGRES_PORT', '5432')} "
        f"dbname={os.environ['POSTGRES_DB']} "
        f"user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


async def _fetch_updated_at(conn, guid: str):
    cur = await conn.execute(
        "SELECT updated_at FROM done_cards WHERE card_guid = %(g)s", {"g": guid}
    )
    row = await cur.fetchone()
    return row[0]


@pytest.mark.asyncio
async def test_updated_at_set_on_insert_and_advances_on_update():
    guid = f"pytest-{uuid.uuid4()}"
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        try:
            await conn.execute(
                "INSERT INTO done_cards (card_guid, ignored) VALUES (%(g)s, FALSE)",
                {"g": guid},
            )
            first = await _fetch_updated_at(conn, guid)
            assert first is not None

            await asyncio.sleep(0.01)
            await conn.execute(
                "UPDATE done_cards SET token_count = 1 WHERE card_guid = %(g)s",
                {"g": guid},
            )
            second = await _fetch_updated_at(conn, guid)
            assert second > first
        finally:
            await conn.execute("DELETE FROM done_cards WHERE card_guid = %(g)s", {"g": guid})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_done_cards_updated_at.py -v`
Expected: FAIL — `psycopg.errors.UndefinedColumn: column "updated_at" does not exist` (migration not applied yet).

- [ ] **Step 3: Write the migration**

`migrations/022_done_cards_updated_at.sql`:
```sql
-- Migration 019: change-tracking column for incremental export to the engine's
-- analyst replica. updated_at is bumped to transaction now() on every insert
-- and update, regardless of card type (audited / ignored / broken).

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

- [ ] **Step 4: Apply the migration**

Run: `bash migrations/migrate.sh`
Expected: prints `Applying 022_done_cards_updated_at.sql ...` then `All migrations applied.` (earlier migrations are idempotent, so re-running is safe).

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_done_cards_updated_at.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add migrations/022_done_cards_updated_at.sql tests/test_done_cards_updated_at.py
git commit -m "feat(export): add done_cards.updated_at change-tracking trigger"
```

---

### Task 2: `ApiFormatter.export` reader

**Files:**
- Modify: `src/reporting/api_formatter.py`
- Test: `tests/test_export_formatter.py`

**Interfaces:**
- Consumes: `done_cards.updated_at` (Task 1); `require_org_access` is not used here (this is the DB layer).
- Produces:
  - `ApiFormatter.export(organization_id: str, since: str | None, limit: int, cursor: int) -> list[dict]`
    — audited cards only (`ignored=FALSE AND broken=FALSE`), ordered by `(updated_at, card_guid)`; when `limit == 0` no `LIMIT/OFFSET` is applied; each dict has exactly the keys `card_guid, card_data, formal_result, diag_result, icd_check_result, updated_at`. JSONB columns are native Python objects; `updated_at` is a `datetime`.

- [ ] **Step 1: Write the failing test**

`tests/test_export_formatter.py`:
```python
"""
Tests ApiFormatter.export against the real Postgres: since-filtering, limit=0
(no paging), and limit/cursor offset paging. Seeds and cleans up its own rows.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import psycopg
import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from api.app import create_app  # noqa: E402  (ensures src on path)
from reporting.api_formatter import ApiFormatter  # noqa: E402
from storage.organizations_storage import OrganizationsStorage  # noqa: E402


def _conninfo() -> str:
    return (
        f"host={os.environ['POSTGRES_HOST']} port={os.environ.get('POSTGRES_PORT','5432')} "
        f"dbname={os.environ['POSTGRES_DB']} user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


@pytest.fixture
async def alenka_org_id() -> str:
    async with OrganizationsStorage() as orgs:
        return await orgs.get_id_by_name("Alenka")


@pytest.fixture
async def seeded_guids(alenka_org_id: str):
    guids = [f"pytest-export-{uuid.uuid4()}" for _ in range(3)]
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        for g in guids:
            await conn.execute(
                "INSERT INTO done_cards (card_guid, card_data, ignored, organization_id) "
                "VALUES (%(g)s, %(d)s::jsonb, FALSE, %(o)s)",
                {"g": g, "d": '{"Прием": {"DATE": "01.07.2026"}}', "o": alenka_org_id},
            )
    yield guids, alenka_org_id
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        await conn.execute(
            "DELETE FROM done_cards WHERE card_guid = ANY(%(gs)s)", {"gs": guids}
        )


async def test_export_limit_zero_returns_all_with_native_jsonb(seeded_guids):
    guids, org_id = seeded_guids
    async with ApiFormatter() as fmt:
        rows = await fmt.export(org_id, since=None, limit=0, cursor=0)
    got = {r["card_guid"] for r in rows}
    assert set(guids) <= got
    sample = next(r for r in rows if r["card_guid"] in guids)
    assert isinstance(sample["card_data"], dict)             # native JSONB, not str
    assert set(sample.keys()) == {
        "card_guid", "card_data", "formal_result",
        "diag_result", "icd_check_result", "updated_at",
    }                                                        # trimmed, audited-only columns


async def test_export_cursor_offset_paging_is_exhaustive(seeded_guids):
    guids, org_id = seeded_guids
    seen, cursor = [], 0
    while True:
        async with ApiFormatter() as fmt:
            page = await fmt.export(org_id, since=None, limit=2, cursor=cursor)
        seen.extend(r["card_guid"] for r in page)
        if len(page) < 2:
            break
        cursor += 2
    assert set(guids) <= set(seen)
    assert len(seen) == len(set(seen))                       # no dup across pages
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_export_formatter.py -v`
Expected: FAIL — `AttributeError: 'ApiFormatter' object has no attribute 'export'`.

- [ ] **Step 3: Add the reader method and `ApiFormatter.export`**

In `src/reporting/api_formatter.py`, add a method to `_ApiCardsReader` (after `fetch_by_date`):
```python
    async def fetch_export(
        self, organization_id: str, since: str | None, limit: int, cursor: int
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT card_guid, card_data, formal_result, diag_result, "
            "       icd_check_result, updated_at "
            "FROM done_cards "
            "WHERE organization_id = %(org_id)s::uuid "
            "  AND ignored = FALSE AND broken = FALSE "        # audited cards only
            "  AND (%(since)s::timestamptz IS NULL OR updated_at > %(since)s::timestamptz) "
            "ORDER BY updated_at, card_guid "
        )
        params: dict[str, Any] = {"org_id": organization_id, "since": since}
        if limit and limit > 0:
            query += "LIMIT %(limit)s OFFSET %(cursor)s"
            params["limit"] = limit
            params["cursor"] = cursor

        async with self._pool.connection() as conn:
            cur = await conn.execute(query, params)
            return await cur.fetchall()
```
And on `ApiFormatter` (after `make_xlsx`):
```python
    async def export(
        self, organization_id: str, since: str | None, limit: int, cursor: int
    ) -> list[dict[str, Any]]:
        """Return done_cards rows for one org as native dicts.

        since=None → all history; limit=0 → no LIMIT/OFFSET (one-shot daily).
        limit>0 uses cursor as an OFFSET for the backfill loop.
        """
        return await self._reader.fetch_export(organization_id, since, limit, cursor)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_export_formatter.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/reporting/api_formatter.py tests/test_export_formatter.py
git commit -m "feat(export): ApiFormatter.export reader (since + limit/cursor paging)"
```

---

### Task 3: `GET /cards/export` route

**Files:**
- Modify: `src/api/routes/cards.py`
- Test: `tests/test_export_api.py`

**Interfaces:**
- Consumes: `ApiFormatter.export(...)` (Task 2); `require_org_access` → `(org_id, org_name)`.
- Produces: `GET /cards/export?org=&since=&limit=&cursor=` returning a JSON array of row objects (datetimes serialized to ISO strings by FastAPI's default encoder). Defaults: `limit=0`, `cursor=0`, `since=None`.

- [ ] **Step 1: Write the failing test**

`tests/test_export_api.py`:
```python
"""
Integration tests for GET /cards/export — real Postgres via TestClient, using
an api key scoped to Alenka + MDS. Seeds a couple of Alenka rows and cleans up.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import psycopg
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from api.app import create_app  # noqa: E402
from storage.api_keys_storage import ApiKeysStorage  # noqa: E402
from storage.organizations_storage import OrganizationsStorage  # noqa: E402


def _conninfo() -> str:
    return (
        f"host={os.environ['POSTGRES_HOST']} port={os.environ.get('POSTGRES_PORT','5432')} "
        f"dbname={os.environ['POSTGRES_DB']} user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
async def alenka_org_id() -> str:
    async with OrganizationsStorage() as orgs:
        return await orgs.get_id_by_name("Alenka")


@pytest.fixture
async def mds_org_id() -> str:
    async with OrganizationsStorage() as orgs:
        return await orgs.get_id_by_name("MDS")


@pytest.fixture
async def test_key(alenka_org_id: str, mds_org_id: str) -> str:
    raw = f"medkard_test_{uuid.uuid4().hex}"
    async with ApiKeysStorage() as api_keys:
        key_id = await api_keys.create_key("pytest-export", raw, [alenka_org_id, mds_org_id])
    yield raw
    async with ApiKeysStorage() as api_keys:
        await api_keys.revoke_key(key_id)


@pytest.fixture
async def seeded(alenka_org_id: str):
    guids = [f"pytest-exapi-{uuid.uuid4()}" for _ in range(2)]
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        for g in guids:
            await conn.execute(
                "INSERT INTO done_cards (card_guid, card_data, ignored, organization_id) "
                "VALUES (%(g)s, %(d)s::jsonb, FALSE, %(o)s)",
                {"g": g, "d": '{"Прием": {"DATE": "01.07.2026"}}', "o": alenka_org_id},
            )
    yield guids
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        await conn.execute("DELETE FROM done_cards WHERE card_guid = ANY(%(gs)s)", {"gs": guids})


def test_export_requires_key(client: TestClient):
    resp = client.get("/cards/export?org=Alenka")
    assert resp.status_code in (401, 403)


def test_export_returns_seeded_rows_with_native_jsonb(client, test_key, seeded):
    resp = client.get("/cards/export?org=Alenka", headers=_auth(test_key))
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    by_guid = {r["card_guid"]: r for r in body}
    assert set(seeded) <= set(by_guid)
    sample = by_guid[seeded[0]]
    assert isinstance(sample["card_data"], dict)          # native JSONB
    assert "token_count" not in sample and "organization_name" not in sample  # trimmed


def test_export_cursor_offset_paging(client, test_key, seeded):
    seen, cursor = [], 0
    while True:
        resp = client.get(f"/cards/export?org=Alenka&limit=1&cursor={cursor}", headers=_auth(test_key))
        page = resp.json()
        seen.extend(r["card_guid"] for r in page)
        if len(page) < 1:
            break
        cursor += 1
    assert set(seeded) <= set(seen)
    assert len(seen) == len(set(seen))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_export_api.py -v`
Expected: FAIL — the seeded-rows/paging tests get `404 Not Found` for `/cards/export` (route missing).

- [ ] **Step 3: Add the route**

In `src/api/routes/cards.py`, add the import and route (after `pull`):
```python
from reporting.api_formatter import ApiFormatter  # add near the ApiFormatter usage / top imports


@router.get("/export")
async def export(
    since: str | None = Query(default=None),
    limit: int = Query(default=0, ge=0),
    cursor: int = Query(default=0, ge=0),
    org_access: tuple[str, str] = Depends(require_org_access),
) -> list[dict]:
    org_id, _ = org_access
    async with ApiFormatter() as formatter:
        return await formatter.export(org_id, since, limit, cursor)
```
(`ApiFormatter` is already imported by `pull`'s module — if it is imported inline there, hoist the import to module top; otherwise add the import line above.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_export_api.py -v`
Expected: PASS (all three tests).

- [ ] **Step 5: Run the full API suite to confirm no regressions**

Run: `pytest tests/test_cards_api.py tests/test_export_api.py -v`
Expected: PASS (existing pull/check tests still green).

- [ ] **Step 6: Commit**

```bash
git add src/api/routes/cards.py tests/test_export_api.py
git commit -m "feat(export): GET /cards/export endpoint (daily since + backfill paging)"
```

---

## Self-Review

**Spec coverage:**
- §3 `updated_at` trigger + index → Task 1. ✓
- §4 export endpoint (`org` required, `since`, `limit=0` default, `cursor`-as-offset, native JSONB, **audited-only** + trimmed to six columns, stable ORDER BY) → Tasks 2 (reader) + 3 (route). ✓
- Hard-delete reconcile is handled engine-side by a full resync (truncate the clinic replica + full re-export via `/cards/export` with no `since`), so **no dedicated guids endpoint is needed** — dropped from scope. ✓
- §6 auth/exposure (reuse `require_org_access`, no PG exposure) → Tasks 3 & 4 use the existing dependency. ✓
- §7 testing (trigger advances; since filter; limit=0 one-shot; offset paging exhaustive/no-dup; org-scoping) → covered across the three test files. ✓

**Placeholder scan:** No TBD/TODO; every code and test step is complete; commands have expected output.

**Type consistency:** `ApiFormatter.export(organization_id, since, limit, cursor)` and `_ApiCardsReader.fetch_export(...)` share the same signature and are called identically in Task 3. Row dict keys used in tests (`card_guid`, `card_data`, `updated_at`) match the trimmed SELECT list in Task 2.

**Notes for the engine-side consumer (out of scope here):** the daily pull calls `?org=&since=<watermark>` with `limit=0`; the backfill/resync loops `?org=&limit=5000&cursor=<offset>` (a resync omits `since` and full-replaces the clinic replica, which also clears any hard-deleted rows).
