# Cards Push Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a 1C organization push a single updated card to us at any time via a new
`POST /cards/push` endpoint; store it as raw/unaudited (`status='pending'`, audit results wiped);
have the nightly audit job pick up pending cards for that org alongside its normal 1C pull.

**Architecture:** One new `status` column on `done_cards` (`pending`/`done`), two new
`DoneCardsStorage` methods (`upsert_pending`, `get_pending`), one new authenticated FastAPI route
reusing the existing `require_org_access` dependency, and a small nightly-script change that merges
pending-card raw data into the batch handed to `AuditPipeline.run_batched`. No new container — the
route lives in the existing `src/api` FastAPI service already deployed via `docker-compose.yml`.

**Tech Stack:** FastAPI, Pydantic, psycopg3 (async), PostgreSQL, pytest + pytest-asyncio
(`asyncio_mode=auto`), FastAPI `TestClient`.

## Global Constraints

- Migrations live in `migrations/NNN_description.sql`, applied in order by `migrations/migrate.sh`;
  next free number is `025`.
- `pythonpath = src` (pytest.ini) — tests import via `from api.app import create_app`, etc., no
  installation needed.
- Tests hit the **real configured Postgres** from `.env` (no mocking DB access) — matches
  `tests/test_cards_api.py` / `tests/test_done_cards_updated_at.py` conventions. Clean up any rows
  a test creates.
- `card_guid` is always lower-cased before storage/comparison (see `_visit_guid()` in
  `src/audit/pipeline.py` and `CardFilter.filter()` in `src/audit/filters.py`).
- Route handlers in `src/api/routes/cards.py` do **not** contain business logic — they parse
  params, delegate to a storage/formatter class, return the result (see existing file header
  comment).
- No new Pydantic request-body model for the push payload — the visit JSON is accepted as an
  arbitrary `dict` and stored as-is in `card_data`.

---

### Task 1: `done_cards.status` column + backfill

**Files:**
- Create: `migrations/025_done_cards_status.sql`
- Test: `tests/test_done_cards_status_migration.py`

**Interfaces:**
- Produces: `done_cards.status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','done'))`,
  index `done_cards_status_idx`. All rows existing before this migration end up `status='done'`
  (they've already been through the pipeline in some terminal form).

- [ ] **Step 1: Write the migration**

```sql
-- migrations/025_done_cards_status.sql
-- Migration 025: status column on done_cards — tracks whether a card's
-- stored data has been through the audit pipeline yet.
--
-- 'pending' = card_data is raw/unaudited (freshly pushed or never audited);
-- 'done'    = card has a terminal outcome (audited, ignored, or broken).
--
-- Existing rows are backfilled to 'done': every row already in the table
-- came from a completed pipeline run (upsert / upsert_ignored / upsert_broken),
-- so none of them are "still pending" under the new column's meaning.

ALTER TABLE done_cards
    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'done'));

UPDATE done_cards SET status = 'done' WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS done_cards_status_idx ON done_cards (status);
```

Note: the `UPDATE ... WHERE status = 'pending'` runs once, right after the column is added with its
default — at that moment every existing row is `'pending'` (the just-applied default), so this
backfills all of them to `'done'` in one statement without needing a `NOT EXISTS` guard.

- [ ] **Step 2: Apply the migration against the configured DB**

Run: `bash migrations/migrate.sh`
Expected: output includes `Applying 025_done_cards_status.sql` (or equivalent success line) and no
errors. Confirm with:

```bash
psql "$(python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(f\"host={os.environ['POSTGRES_HOST']} port={os.environ.get('POSTGRES_PORT','5432')} dbname={os.environ['POSTGRES_DB']} user={os.environ['POSTGRES_USER']} password={os.environ['POSTGRES_PASSWORD']}\")")" -c "\d done_cards" | grep status
```
Expected: a `status` column of type `text`, `not null`, with a default.

- [ ] **Step 3: Write the verification test**

```python
# tests/test_done_cards_status_migration.py
"""
Verifies migration 025: done_cards.status exists, defaults sensibly, and its
CHECK constraint rejects invalid values. Hits the real configured Postgres.
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


def _conninfo() -> str:
    return (
        f"host={os.environ['POSTGRES_HOST']} "
        f"port={os.environ.get('POSTGRES_PORT', '5432')} "
        f"dbname={os.environ['POSTGRES_DB']} "
        f"user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


@pytest.mark.asyncio
async def test_existing_rows_backfilled_to_done():
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        cur = await conn.execute(
            "SELECT count(*) FROM done_cards WHERE status IS NULL OR status NOT IN ('pending','done')"
        )
        row = await cur.fetchone()
        assert row[0] == 0


@pytest.mark.asyncio
async def test_new_row_defaults_to_pending():
    guid = f"pytest-{uuid.uuid4()}"
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        try:
            await conn.execute(
                "INSERT INTO done_cards (card_guid, ignored) VALUES (%(g)s, FALSE)",
                {"g": guid},
            )
            cur = await conn.execute(
                "SELECT status FROM done_cards WHERE card_guid = %(g)s", {"g": guid}
            )
            row = await cur.fetchone()
            assert row[0] == "pending"
        finally:
            await conn.execute("DELETE FROM done_cards WHERE card_guid = %(g)s", {"g": guid})


@pytest.mark.asyncio
async def test_invalid_status_is_rejected():
    guid = f"pytest-{uuid.uuid4()}"
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        with pytest.raises(psycopg.errors.CheckViolation):
            await conn.execute(
                "INSERT INTO done_cards (card_guid, status) VALUES (%(g)s, 'bogus')",
                {"g": guid},
            )
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_done_cards_status_migration.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add migrations/025_done_cards_status.sql tests/test_done_cards_status_migration.py
git commit -m "feat(db): add done_cards.status column (pending/done)"
```

---

### Task 2: `DoneCardsStorage.upsert_pending()` + `get_pending()`

**Files:**
- Modify: `src/storage/done_cards_storage.py`
- Modify: `src/storage/done_cards_storage.py` (`upsert`, `upsert_ignored`, `upsert_broken`, `get_done_guids`)
- Test: `tests/test_done_cards_storage_pending.py`

**Interfaces:**
- Consumes: `BaseStorage` (`self._pool`), nothing else new.
- Produces:
  - `DoneCardsStorage.upsert_pending(*, card_guid: str, card_data: str, organization_id: str | None = None) -> str`
    — returns the row's `id` (text UUID).
  - `DoneCardsStorage.get_pending(organization_id: str | None = None) -> list[dict]` — each dict has
    keys `card_guid` and `card_data` (the latter a JSON string, matching how `run_batched`/
    `AppointmentParser.split` already expect raw payload elements).
  - `upsert()`, `upsert_ignored()`, `upsert_broken()` now also set `status = 'done'`.
  - `get_done_guids()` now only returns guids where `status = 'done'` (a `pending` row's guid must
    NOT count as "already audited" — see Task 4 for why this matters).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_done_cards_storage_pending.py
"""
Integration tests for DoneCardsStorage.upsert_pending / get_pending, and for
status transitions on the existing upsert*/get_done_guids methods. Hits the
real configured Postgres.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from storage.done_cards_storage import DoneCardsStorage
from storage.models.result import FormalStructureResult


async def _cleanup(guid: str) -> None:
    async with DoneCardsStorage() as storage:
        async with storage._pool.connection() as conn:
            await conn.execute("DELETE FROM done_cards WHERE card_guid = %(g)s", {"g": guid})


@pytest.mark.asyncio
async def test_upsert_pending_creates_row_with_pending_status():
    guid = f"pytest-push-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            row_id = await storage.upsert_pending(
                card_guid=guid,
                card_data=json.dumps({"Прием": {"GUID": guid}}, ensure_ascii=False),
                organization_id=None,
            )
            assert row_id

            async with storage._pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT status, card_data, formal_result FROM done_cards WHERE card_guid = %(g)s",
                    {"g": guid},
                )
                row = await cur.fetchone()
        assert row["status"] == "pending"
        assert row["formal_result"] is None
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_upsert_pending_on_existing_done_row_wipes_results_and_flags():
    guid = f"pytest-push-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            now = datetime.now(timezone.utc)
            await storage.upsert(
                card_guid=guid,
                card_data=json.dumps({"Прием": {"GUID": guid}, "v": 1}, ensure_ascii=False),
                formal=FormalStructureResult(findings=[]),
                diagnosis=[],
                icd_check=[],
                token_count=10,
                time_ms=5,
                started_at=now,
                finished_at=now,
                organization_id=None,
            )

            async with storage._pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT status FROM done_cards WHERE card_guid = %(g)s", {"g": guid}
                )
                assert (await cur.fetchone())["status"] == "done"

            await storage.upsert_pending(
                card_guid=guid,
                card_data=json.dumps({"Прием": {"GUID": guid}, "v": 2}, ensure_ascii=False),
                organization_id=None,
            )

            async with storage._pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT status, card_data, formal_result, diag_result, icd_check_result, "
                    "ignored, broken, stacktrace FROM done_cards WHERE card_guid = %(g)s",
                    {"g": guid},
                )
                row = await cur.fetchone()
        assert row["status"] == "pending"
        assert row["card_data"]["v"] == 2
        assert row["formal_result"] is None
        assert row["diag_result"] is None
        assert row["icd_check_result"] is None
        assert row["ignored"] is False
        assert row["broken"] is False
        assert row["stacktrace"] is None
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_get_pending_returns_only_pending_rows_for_org():
    org_id = None  # NULL-scoped org bucket, matches existing tests' style for org-less rows
    pending_guid = f"pytest-pending-{uuid.uuid4()}"
    done_guid = f"pytest-done-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            await storage.upsert_pending(
                card_guid=pending_guid,
                card_data=json.dumps({"Прием": {"GUID": pending_guid}}, ensure_ascii=False),
                organization_id=org_id,
            )
            now = datetime.now(timezone.utc)
            await storage.upsert(
                card_guid=done_guid,
                card_data=json.dumps({"Прием": {"GUID": done_guid}}, ensure_ascii=False),
                formal=FormalStructureResult(findings=[]),
                diagnosis=[],
                icd_check=[],
                token_count=1,
                time_ms=1,
                started_at=now,
                finished_at=now,
                organization_id=org_id,
            )

            pending_rows = await storage.get_pending(organization_id=org_id)
        pending_guids = {r["card_guid"] for r in pending_rows}
        assert pending_guid in pending_guids
        assert done_guid not in pending_guids
    finally:
        await _cleanup(pending_guid)
        await _cleanup(done_guid)


@pytest.mark.asyncio
async def test_get_done_guids_excludes_pending_rows():
    guid = f"pytest-pending-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            await storage.upsert_pending(
                card_guid=guid,
                card_data=json.dumps({"Прием": {"GUID": guid}}, ensure_ascii=False),
                organization_id=None,
            )
            done_guids = await storage.get_done_guids(organization_id=None)
        assert guid not in done_guids
    finally:
        await _cleanup(guid)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_done_cards_storage_pending.py -v`
Expected: FAIL — `AttributeError: 'DoneCardsStorage' object has no attribute 'upsert_pending'`.

- [ ] **Step 3: Add `upsert_pending` and `get_pending`, update existing methods**

Add to `src/storage/done_cards_storage.py`, after `upsert_broken` and before `get_done_guids`:

```python
    async def upsert_pending(
        self,
        *,
        card_guid: str,
        card_data: str,
        organization_id: str | None = None,
    ) -> str:
        """Insert or update a done_cards row with fresh raw data awaiting audit.

        Sets status='pending' and clears every audit-derived column (results,
        ignored, broken, stacktrace) — a pushed update means the previous
        audit outcome, if any, is stale and must be recomputed from scratch.
        """
        try:
            async with self._pool.connection() as conn:
                cur = await conn.execute(
                    """
                    INSERT INTO done_cards
                        (card_guid, card_data, status, organization_id)
                    VALUES
                        (%(guid)s, %(data)s::jsonb, 'pending', %(org_id)s)
                    ON CONFLICT (card_guid) DO UPDATE SET
                        card_data         = EXCLUDED.card_data,
                        status            = 'pending',
                        formal_result     = NULL,
                        diag_result       = NULL,
                        icd_check_result  = NULL,
                        ignored           = FALSE,
                        broken            = FALSE,
                        stacktrace        = NULL,
                        organization_id   = EXCLUDED.organization_id
                    RETURNING id::text
                    """,
                    {"guid": card_guid, "data": card_data, "org_id": organization_id},
                )
                row = await cur.fetchone()
            row_id: str = row["id"]
            logger.info("💾 done_cards UPSERT_PENDING OK id=%s guid=%s", row_id, card_guid)
            return row_id
        except Exception:
            logger.exception("💾 done_cards UPSERT_PENDING FAILED guid=%s", card_guid)
            raise

    async def get_pending(self, organization_id: str | None = None) -> list[dict]:
        """Return card_guid + card_data for pending rows in an organization."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT card_guid, card_data FROM done_cards "
                "WHERE status = 'pending' "
                "AND organization_id IS NOT DISTINCT FROM %(org_id)s",
                {"org_id": organization_id},
            )
            rows = await cur.fetchall()
        logger.info("💾 done_cards loaded %d pending card(s) for org_id=%s", len(rows), organization_id)
        return rows
```

Update `upsert()` — in both the `card_guid`-provided and no-`card_guid` `INSERT` statements, add
`status` to the column list and value `'done'`, and in the `ON CONFLICT DO UPDATE SET` clause add
`status = 'done',`. Concretely, change the first `INSERT` column list from:

```sql
                        INSERT INTO done_cards
                            (card_guid, card_data, formal_result, diag_result, icd_check_result, token_count, time_ms, started_at, finished_at, ignored, organization_id)
                        VALUES
                            (%(guid)s, %(data)s::jsonb, %(formal)s::jsonb, %(diag)s::jsonb, %(icd_check)s::jsonb, %(tokens)s, %(ms)s, %(started_at)s, %(finished_at)s, FALSE, %(org_id)s)
                        ON CONFLICT (card_guid) DO UPDATE SET
                            card_data         = EXCLUDED.card_data,
```

to:

```sql
                        INSERT INTO done_cards
                            (card_guid, card_data, formal_result, diag_result, icd_check_result, token_count, time_ms, started_at, finished_at, ignored, organization_id, status)
                        VALUES
                            (%(guid)s, %(data)s::jsonb, %(formal)s::jsonb, %(diag)s::jsonb, %(icd_check)s::jsonb, %(tokens)s, %(ms)s, %(started_at)s, %(finished_at)s, FALSE, %(org_id)s, 'done')
                        ON CONFLICT (card_guid) DO UPDATE SET
                            card_data         = EXCLUDED.card_data,
                            status            = 'done',
```

and the second (no-guid) `INSERT` column list from:

```sql
                        INSERT INTO done_cards
                            (card_guid, card_data, formal_result, diag_result, icd_check_result, token_count, time_ms, started_at, finished_at, ignored, organization_id)
                        VALUES
                            (NULL, %(data)s::jsonb, %(formal)s::jsonb, %(diag)s::jsonb, %(icd_check)s::jsonb, %(tokens)s, %(ms)s, %(started_at)s, %(finished_at)s, FALSE, %(org_id)s)
                        RETURNING id::text
```

to:

```sql
                        INSERT INTO done_cards
                            (card_guid, card_data, formal_result, diag_result, icd_check_result, token_count, time_ms, started_at, finished_at, ignored, organization_id, status)
                        VALUES
                            (NULL, %(data)s::jsonb, %(formal)s::jsonb, %(diag)s::jsonb, %(icd_check)s::jsonb, %(tokens)s, %(ms)s, %(started_at)s, %(finished_at)s, FALSE, %(org_id)s, 'done')
                        RETURNING id::text
```

Update `upsert_ignored()` similarly — column list `(card_guid, card_data, ignored, organization_id)`
→ `(card_guid, card_data, ignored, organization_id, status)`, values `(%(guid)s, %(data)s::jsonb,
TRUE, %(org_id)s)` → `(%(guid)s, %(data)s::jsonb, TRUE, %(org_id)s, 'done')`, and add
`status = 'done',` into its `ON CONFLICT DO UPDATE SET` clause.

Update `upsert_broken()` similarly for both its `card_guid`-provided and no-guid branches — add
`status` column + `'done'` value to each `INSERT`, and `status = 'done',` to the
`ON CONFLICT DO UPDATE SET` clause.

Update `get_done_guids()`:

```python
    async def get_done_guids(self, organization_id: str | None = None) -> set[str]:
        """Return non-null card GUIDs with a terminal (done) status for an organization.

        Pending rows (freshly pushed, not yet audited) are excluded: their
        guid must not count as "already handled", or the nightly pipeline's
        always-on dedup would skip them forever.
        """
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT card_guid FROM done_cards "
                "WHERE card_guid IS NOT NULL "
                "AND status = 'done' "
                "AND organization_id IS NOT DISTINCT FROM %(org_id)s",
                {"org_id": organization_id},
            )
            rows = await cur.fetchall()
        guids = {row["card_guid"] for row in rows}
        logger.info("💾 done_cards loaded %d done guid(s) for org_id=%s", len(guids), organization_id)
        return guids
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_done_cards_storage_pending.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run the full existing done_cards/pipeline test suite to check for regressions**

Run: `pytest tests/test_done_cards_updated_at.py tests/test_cards_api.py tests/test_export_api.py -v`
Expected: all passed (no behavior change for already-done rows; `get_done_guids` still returns them
since they're all `status='done'`).

- [ ] **Step 6: Commit**

```bash
git add src/storage/done_cards_storage.py tests/test_done_cards_storage_pending.py
git commit -m "feat(storage): add upsert_pending/get_pending, scope get_done_guids to status=done"
```

---

### Task 3: `POST /cards/push` route

**Files:**
- Modify: `src/api/routes/cards.py`
- Test: `tests/test_cards_push_api.py`

**Interfaces:**
- Consumes: `api.auth.require_org_access` (existing dependency, returns `(org_id, org_name)`),
  `storage.done_cards_storage.DoneCardsStorage.upsert_pending` (Task 2).
- Produces: `POST /cards/push?org=<name>` → `200 OK` `{"card_guid": str, "status": "pending"}` on
  success, `422` if the body has no extractable `Прием.GUID`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cards_push_api.py
"""
Integration tests for POST /cards/push — hits the real configured Postgres
via FastAPI's TestClient, same fixture pattern as tests/test_cards_api.py.
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from api.app import create_app
from storage.api_keys_storage import ApiKeysStorage
from storage.done_cards_storage import DoneCardsStorage
from storage.organizations_storage import OrganizationsStorage


def _unique_key() -> str:
    return f"medkard_test_{uuid.uuid4().hex}"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
async def alenka_org_id() -> str:
    async with OrganizationsStorage() as organizations:
        return await organizations.get_id_by_name("Alenka")


@pytest.fixture
async def test_key(alenka_org_id: str) -> str:
    raw_key = _unique_key()
    async with ApiKeysStorage() as api_keys:
        key_id = await api_keys.create_key("pytest-push", raw_key, [alenka_org_id])
    yield raw_key
    async with ApiKeysStorage() as api_keys:
        await api_keys.revoke_key(key_id)


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _cleanup(guid: str) -> None:
    async def _delete():
        async with DoneCardsStorage() as storage:
            async with storage._pool.connection() as conn:
                await conn.execute("DELETE FROM done_cards WHERE card_guid = %(g)s", {"g": guid})

    asyncio.get_event_loop().run_until_complete(_delete())


def _get_pending(organization_id: str | None) -> list[dict]:
    async def _fetch():
        async with DoneCardsStorage() as storage:
            return await storage.get_pending(organization_id=organization_id)

    return asyncio.get_event_loop().run_until_complete(_fetch())


def test_push_missing_key_is_rejected(client: TestClient):
    guid = str(uuid.uuid4())
    resp = client.post("/cards/push?org=Alenka", json={"Прием": {"GUID": guid}})
    assert resp.status_code in (401, 403)


def test_push_without_guid_is_422(client: TestClient, test_key: str):
    resp = client.post("/cards/push?org=Alenka", json={"Прием": {}}, headers=_auth(test_key))
    assert resp.status_code == 422


def test_push_new_card_creates_pending_row(client: TestClient, test_key: str, alenka_org_id: str):
    guid = str(uuid.uuid4())
    try:
        resp = client.post(
            "/cards/push?org=Alenka",
            json={"Прием": {"GUID": guid}, "Пациент": {"ФИО": "Тест Тестов"}},
            headers=_auth(test_key),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["card_guid"] == guid.lower()
        assert body["status"] == "pending"

        rows = _get_pending(organization_id=alenka_org_id)
        matching = [r for r in rows if r["card_guid"] == guid.lower()]
        assert len(matching) == 1
    finally:
        _cleanup(guid)


def test_push_updates_existing_card_and_resets_to_pending(client: TestClient, test_key: str, alenka_org_id: str):
    guid = str(uuid.uuid4())
    try:
        first = client.post(
            "/cards/push?org=Alenka",
            json={"Прием": {"GUID": guid}, "v": 1},
            headers=_auth(test_key),
        )
        assert first.status_code == 200

        second = client.post(
            "/cards/push?org=Alenka",
            json={"Прием": {"GUID": guid}, "v": 2},
            headers=_auth(test_key),
        )
        assert second.status_code == 200
        assert second.json()["status"] == "pending"

        rows = _get_pending(organization_id=alenka_org_id)
        matching = [r for r in rows if r["card_guid"] == guid.lower()]
        assert len(matching) == 1
        assert matching[0]["card_data"]["v"] == 2
    finally:
        _cleanup(guid)
```

Note on the sync helpers above: `TestClient` drives the FastAPI app through its own internal event
loop, so these tests use plain `def test_...` signatures (matching the existing style in
`tests/test_cards_api.py`) rather than `@pytest.mark.asyncio`. `_cleanup` and `_get_pending` wrap
their async DB calls in `asyncio.get_event_loop().run_until_complete(...)` so they can be called
from those sync test bodies.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cards_push_api.py -v`
Expected: FAIL with `404 Not Found` on the `/cards/push` calls (route doesn't exist yet).

- [ ] **Step 3: Add the `PushResponse` model**

Add to `src/api/models.py`:

```python
class PushResponse(BaseModel):
    card_guid: str
    status: str
```

- [ ] **Step 4: Add the route**

Add to `src/api/routes/cards.py`, after the `export` route, and add the two needed imports at the
top of the file:

```python
from api.models import CheckResponse, PushResponse
from storage.done_cards_storage import DoneCardsStorage
```

(replacing the existing `from api.models import CheckResponse` line).

```python
def _extract_card_guid(card: dict) -> str | None:
    priem = card.get("Прием") or {}
    guid = priem.get("GUID")
    return str(guid).lower() if guid else None


@router.post("/push", response_model=PushResponse)
async def push(
    card: dict,
    org_access: tuple[str, str] = Depends(require_org_access),
) -> PushResponse:
    org_id, _ = org_access
    card_guid = _extract_card_guid(card)
    if not card_guid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Card is missing Прием.GUID",
        )

    async with DoneCardsStorage() as storage:
        await storage.upsert_pending(
            card_guid=card_guid,
            card_data=json.dumps(card, ensure_ascii=False),
            organization_id=org_id,
        )

    return PushResponse(card_guid=card_guid, status="pending")
```

Add `import json` to the top of `src/api/routes/cards.py` alongside the existing `from datetime
import date` import.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_cards_push_api.py -v`
Expected: 4 passed.

- [ ] **Step 6: Run the full existing cards API suite to check for regressions**

Run: `pytest tests/test_cards_api.py -v`
Expected: all passed.

- [ ] **Step 7: Commit**

```bash
git add src/api/routes/cards.py src/api/models.py tests/test_cards_push_api.py
git commit -m "feat(api): add POST /cards/push to accept updated cards from 1C orgs"
```

---

### Task 4: Nightly job picks up pending cards

**Files:**
- Create: `src/audit/pending_merge.py`
- Modify: `scripts/audit-one-c-period.py`
- Test: `tests/test_pending_merge.py`

**Interfaces:**
- Consumes: `parsers.json_parser.AppointmentParser.split` (existing), `storage.done_cards_storage
  .DoneCardsStorage.get_pending` (Task 2, returns `list[dict]` with `card_guid`/`card_data` keys).
- Produces: `audit.pending_merge.merge_pending_cards(payload: dict | list | str, pending_rows:
  list[dict]) -> list[dict]` — a flat list of visit dicts, importable from a real module (not a
  script), so both the production call site and the test import the exact same function.

`scripts/audit-one-c-period.py` runs `argparse` at import time, so it can't be imported directly
from a test process. The merge logic is therefore a plain function in a new
`src/audit/pending_merge.py` module (importable, side-effect-free), and the script just calls it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pending_merge.py
"""
Unit tests for audit.pending_merge.merge_pending_cards: folds pending
(pushed) done_cards rows into a 1C payload's visit list, regardless of
whether the payload is a bare list or an {"appointments": [...]} wrapper.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.pending_merge import merge_pending_cards


def test_merge_appends_pending_cards_to_bare_list_payload():
    payload = [{"Прием": {"GUID": "a"}}]
    pending = [{"card_guid": "b", "card_data": {"Прием": {"GUID": "b"}}}]
    merged = merge_pending_cards(payload, pending)
    guids = [v["Прием"]["GUID"] for v in merged]
    assert guids == ["a", "b"]


def test_merge_appends_pending_cards_to_wrapper_dict_payload():
    payload = {"appointments": [{"Прием": {"GUID": "a"}}]}
    pending = [{"card_guid": "b", "card_data": {"Прием": {"GUID": "b"}}}]
    merged = merge_pending_cards(payload, pending)
    guids = [v["Прием"]["GUID"] for v in merged]
    assert guids == ["a", "b"]


def test_merge_with_no_pending_cards_returns_payload_visits_unchanged():
    payload = [{"Прием": {"GUID": "a"}}]
    merged = merge_pending_cards(payload, [])
    assert merged == payload
```

Note: `card_data` as returned by `DoneCardsStorage.get_pending` comes back from psycopg3 as an
already-decoded Python object (the column is `jsonb`, and `BaseStorage` configures
`row_factory=psycopg.rows.dict_row`, which decodes `jsonb` columns to native dict/list) — the test
above reflects that by using a plain dict for `card_data`, not a JSON string.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pending_merge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'audit.pending_merge'`.

- [ ] **Step 3: Create `src/audit/pending_merge.py`**

```python
"""
audit/pending_merge.py — folds pending (pushed) done_cards rows into a 1C
payload's visit list before it's handed to AuditPipeline.run_batched.

Cards pushed via POST /cards/push are stored with status='pending' and never
re-appear from a normal 1C date-range pull if their visit date falls outside
that night's window — merging them in here is what lets a pushed update
actually get (re-)audited.
"""

from __future__ import annotations

from typing import Any

from parsers.json_parser import AppointmentParser


def merge_pending_cards(payload: dict | list | str, pending_rows: list[dict]) -> list[dict[str, Any]]:
    """Return the payload's visits plus every pending row's raw card_data, as one flat list."""
    visits = AppointmentParser.split(payload)
    pending_visits = [row["card_data"] for row in pending_rows]
    return visits + pending_visits
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pending_merge.py -v`
Expected: 3 passed.

- [ ] **Step 5: Wire the merge into `scripts/audit-one-c-period.py`**

Add the imports (alongside the existing imports near the top of the file, next to the other
`from audit... import ...` / `from storage... import ...` lines):

```python
from audit.pending_merge import merge_pending_cards
from storage.done_cards_storage import DoneCardsStorage
```

In `main()`, change stage 1/2:

```python
        # ── 1. Load raw JSON from cache or fetch it from 1C ───────────────────
        payload = _load_or_fetch_one_c_payload(org=_args.org, datebegin=DATEBEGIN, dateend=DATEEND)

        # ── 1b. Merge in any cards pushed to us since the last run ────────────
        async with DoneCardsStorage() as done_cards:
            pending_rows = await done_cards.get_pending(organization_id=org_id)
        if pending_rows:
            log.info("📥 Merging %d pending pushed card(s) into tonight's batch", len(pending_rows))
        merged_payload = merge_pending_cards(payload, pending_rows)

        # ── 2. Run pipeline — each card is persisted to DB on completion ──────
        async with AuditPipeline(org_id=org_id, card_filter=card_filter) as pipeline:
            pairs = await pipeline.run_batched(merged_payload, num_batches=_args.num_batches)
```

- [ ] **Step 6: Run the full test suite to check for regressions**

Run: `pytest -v`
Expected: all passed (existing tests still exercise `run_batched` directly with unchanged
behavior; this script-level change has no effect on tests that don't invoke
`scripts/audit-one-c-period.py`'s `main()`).

- [ ] **Step 7: Commit**

```bash
git add src/audit/pending_merge.py scripts/audit-one-c-period.py tests/test_pending_merge.py
git commit -m "feat(pipeline): merge pending pushed cards into nightly 1C audit batch"
```

---

### Task 5: End-to-end manual verification

**Files:** none (verification only, no code changes)

- [ ] **Step 1: Start the API container locally**

Run: `docker compose up --build api`
Expected: container starts, logs show FastAPI startup with no import errors.

- [ ] **Step 2: Push a test card**

```bash
curl -s -X POST "http://localhost:8000/cards/push?org=Alenka" \
  -H "Authorization: Bearer <a real or test-scoped API key>" \
  -H "Content-Type: application/json" \
  -d '{"Прием": {"GUID": "11111111-1111-1111-1111-111111111111", "DATE": "21.07.2026"}}'
```

Expected: `200 OK`, body `{"card_guid":"11111111-1111-1111-1111-111111111111","status":"pending"}`.

- [ ] **Step 3: Confirm the row landed as pending with no audit results**

```bash
psql "$(python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(f\"host={os.environ['POSTGRES_HOST']} port={os.environ.get('POSTGRES_PORT','5432')} dbname={os.environ['POSTGRES_DB']} user={os.environ['POSTGRES_USER']} password={os.environ['POSTGRES_PASSWORD']}\")")" \
  -c "SELECT card_guid, status, formal_result, diag_result FROM done_cards WHERE card_guid = '11111111-1111-1111-1111-111111111111';"
```

Expected: one row, `status = pending`, `formal_result` and `diag_result` both NULL.

- [ ] **Step 4: Run the nightly script against a period covering that card's date and confirm it gets audited**

Run:
```bash
python scripts/audit-one-c-period.py Alenka --date 21.07.2026 -y
```
Expected: log line `📥 Merging 1 pending pushed card(s) into tonight's batch`, followed by normal
pipeline audit logs for that card_guid, then re-check the DB query from Step 3 shows `status = done`
with populated `formal_result`.

- [ ] **Step 5: Clean up the test row**

```bash
psql "..." -c "DELETE FROM done_cards WHERE card_guid = '11111111-1111-1111-1111-111111111111';"
```

(Use the same connection string as Step 3.)

---

## Self-Review Notes

- **Spec coverage:** status column + backfill (Task 1) ✓, `upsert_pending`/`get_pending` with flag
  reset (Task 2) ✓, `POST /cards/push` reusing `require_org_access`, minimal ack, guid-only
  validation (Task 3) ✓, nightly merge into `run_batched` (Task 4) ✓, no new container (route added
  to existing `src/api` service, no `docker-compose.yml` change needed) ✓, manual end-to-end check
  (Task 5) ✓.
- **Critical fix found during planning, not in the original spec:** `get_done_guids()` had to be
  scoped to `status='done'`, otherwise a pushed card's guid would already be in `done_cards` (as
  `pending`) and `CardFilter.filter()`'s always-on dedup would skip it forever — it would never
  reach the audit step even after being merged into the nightly batch. This is called out explicitly
  in Task 2 and Task 4's docstring.
- **Type/signature consistency:** `get_pending` returns `list[dict]` with `card_guid`/`card_data`
  keys throughout (storage, route tests, merge helper) — no drift between tasks.
