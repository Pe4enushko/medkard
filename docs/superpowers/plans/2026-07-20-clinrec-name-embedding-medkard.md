# Clinrec Name Embedding (medkard) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `name_embedding VECTOR(1024)` column to the medkard `guidelines` registry and populate it (title + age category, passage mode) so the engine sync can copy it for semantic search over guideline names.

**Architecture:** New migration `025` adds the column + HNSW index. A pure `name_embed_input(name, age_category)` function builds the passage string. `GuidelinesStorage.upsert_many` computes the embedding for every upserted row via the existing async `embed()` and writes it — this needs a pgvector codec on the connection (same pattern as `DocsStorage`). All three fill paths (seed-guidelines, reingest full, reingest metadata-only) already go through `upsert_many`, so they inherit population for free.

**Tech Stack:** Python 3.x async, psycopg3 (`psycopg_pool.AsyncConnectionPool`), pgvector (`pgvector.psycopg.register_vector_async`), Qwen3-Embedding-0.6B via `src/RAG/retrieval/embeddings.py::embed`, migrations via `migrations/migrate.sh` (bash, lexicographic `[0-9]*.sql`, ledger table `schema_migrations`).

## Global Constraints

- Embedding dimension: **1024** (`VECTOR(1024)`), model Qwen3-Embedding-0.6B — copy verbatim.
- HNSW params: **`m = 16, ef_construction = 64`, `vector_cosine_ops`** (match `024_docs_reconcile.sql`).
- `name_embed_input` passage string is a **cross-project byte-for-byte contract** with engine (`integrations/clinrec/mapping.py`). Exact form (labeled fields, from spec):
  ```
  base = f"Название: {(name or '').strip()}"
  ages = [a.strip() for a in (age_category or []) if a and a.strip()]
  return f"{base}\nВозрастная группа: [{', '.join(ages)}]" if ages else base
  ```
  Examples: `Название: Бронхит\nВозрастная группа: [Взрослые, Дети]`; no age → `Название: Бронхит`; empty name → `Название: `.
- Embedding is **passage mode: bare `embed(text)`, no instruct prefix** (medkard has no query/passage modes; the engine query side adds the prefix).
- Migrations are **forward-only, idempotent** (`ADD COLUMN IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`).
- Next free migration number is **025** (last is `024_docs_reconcile.sql`).
- `guidelines.name` may be `NULL` (nullable column); `name_embed_input` must tolerate `None`.

---

### Task 1: Migration 025 — add `name_embedding` column + HNSW index

**Files:**
- Create: `migrations/025_guidelines_name_embedding.sql`
- Test: `tests/test_migration_025.py`

**Interfaces:**
- Consumes: nothing.
- Produces: column `guidelines.name_embedding VECTOR(1024)` (nullable), index `guidelines_name_embedding_idx`.

- [ ] **Step 1: Write the failing test** (static SQL assertions, no DB — mirrors `tests/test_migration_024.py`)

```python
# tests/test_migration_025.py
"""Static assertions on the guidelines name-embedding migration SQL (no DB required)."""
from pathlib import Path

SQL = (Path(__file__).resolve().parent.parent
       / "migrations" / "025_guidelines_name_embedding.sql").read_text()


def test_adds_name_embedding_column():
    assert "ADD COLUMN IF NOT EXISTS name_embedding VECTOR(1024)" in SQL


def test_creates_hnsw_index():
    assert "CREATE INDEX IF NOT EXISTS guidelines_name_embedding_idx" in SQL
    assert "hnsw (name_embedding vector_cosine_ops)" in SQL
    assert "m = 16, ef_construction = 64" in SQL


def test_index_is_partial_on_not_null():
    assert "WHERE name_embedding IS NOT NULL" in SQL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_migration_025.py -v`
Expected: FAIL — `FileNotFoundError` (migration file does not exist yet).

- [ ] **Step 3: Write the migration**

```sql
-- migrations/025_guidelines_name_embedding.sql
-- Add a vector column for guideline *names* to the registry, so semantic search
-- over guideline titles is possible. Embeds title + age category (passage mode,
-- bare embed, no instruct prefix). Populated by GuidelinesStorage.upsert_many.
-- Forward-only, idempotent. Dim 1024 (Qwen3-Embedding-0.6B).

ALTER TABLE guidelines
    ADD COLUMN IF NOT EXISTS name_embedding VECTOR(1024);

CREATE INDEX IF NOT EXISTS guidelines_name_embedding_idx
    ON guidelines USING hnsw (name_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE name_embedding IS NOT NULL;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_migration_025.py -v`
Expected: PASS (all three tests).

- [ ] **Step 5: Commit**

```bash
git add migrations/025_guidelines_name_embedding.sql tests/test_migration_025.py
git commit -m "feat(clinrec): миграция 025 — name_embedding в guidelines + HNSW"
```

---

### Task 2: Pure `name_embed_input` function

**Files:**
- Modify: `src/storage/models/guideline.py` (add function + `name_embedding` field)
- Test: `tests/test_guideline_name_embed_input.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `name_embed_input(name: str | None, age_category: list[str]) -> str` (module-level in `guideline.py`).
  - `Guideline.name_embedding: list[float] | None` field (default `None`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_guideline_name_embed_input.py
from storage.models.guideline import name_embed_input


def test_name_with_ages():
    assert name_embed_input("Бронхит", ["Взрослые", "Дети"]) == \
        "Название: Бронхит\nВозрастная группа: [Взрослые, Дети]"


def test_name_without_ages():
    assert name_embed_input("Бронхит", []) == "Название: Бронхит"


def test_none_ages_treated_as_empty():
    assert name_embed_input("Бронхит", None) == "Название: Бронхит"


def test_strips_name_and_ages():
    assert name_embed_input("  Бронхит  ", ["  Дети  "]) == \
        "Название: Бронхит\nВозрастная группа: [Дети]"


def test_drops_blank_age_entries():
    assert name_embed_input("Бронхит", ["Дети", "", "  "]) == \
        "Название: Бронхит\nВозрастная группа: [Дети]"


def test_none_name_becomes_empty_base():
    # name is nullable in the DB; must not crash.
    assert name_embed_input(None, []) == "Название: "
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_guideline_name_embed_input.py -v` (run from `medkard/`, `src` is on path via conftest/pytest config; if import fails, run with `PYTHONPATH=src`)
Expected: FAIL — `ImportError: cannot import name 'name_embed_input'`.

- [ ] **Step 3: Add the function and the field**

In `src/storage/models/guideline.py`, add this module-level function (after `_split_csv_cell`, before `@dataclass`):

```python
def name_embed_input(name: str | None, age_category: list[str] | None) -> str:
    """Passage string embedded for the guideline registry: labeled title + age category.

    CROSS-PROJECT CONTRACT: engine (integrations/clinrec/mapping.py) rebuilds this
    byte-for-byte for its fallback re-embed. Do not change form without updating both.
    Passage mode — bare embed, no instruct prefix.
    """
    base = f"Название: {(name or '').strip()}"
    ages = [a.strip() for a in (age_category or []) if a and a.strip()]
    return f"{base}\nВозрастная группа: [{', '.join(ages)}]" if ages else base
```

Then add the field to the `Guideline` dataclass (after `usage_status`):

```python
    usage_status: str | None = None
    name_embedding: list[float] | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_guideline_name_embed_input.py -v`
Expected: PASS (all six tests).

- [ ] **Step 5: Commit**

```bash
git add src/storage/models/guideline.py tests/test_guideline_name_embed_input.py
git commit -m "feat(clinrec): name_embed_input + Guideline.name_embedding"
```

---

### Task 3: Register pgvector codec + read `name_embedding` in `GuidelinesStorage`

**Files:**
- Modify: `src/storage/guidelines_storage.py`

**Interfaces:**
- Consumes: `Guideline.name_embedding` (Task 2).
- Produces:
  - `GuidelinesStorage.__aenter__` now opens its **own** pool with a pgvector codec (so `VECTOR` reads/writes work), matching `DocsStorage`.
  - `_COLS` includes `name_embedding`; `_row_to_guideline` populates `Guideline.name_embedding`.

This task is prep: it makes the storage vector-aware and round-trips the column on reads. Writing the embedding is Task 4.

- [ ] **Step 1: Write the failing test** (round-trip read; requires a DB with migration 025 applied — mark as DB test)

```python
# tests/test_guidelines_name_embedding_storage.py
import os
import pytest

from storage.guidelines_storage import GuidelinesStorage
from storage.models.guideline import Guideline

pytestmark = pytest.mark.skipif(
    not os.environ.get("POSTGRES_HOST"),
    reason="requires stand DB with migration 025 applied",
)


@pytest.mark.asyncio
async def test_name_embedding_round_trips():
    vec = [0.1] * 1024
    g = Guideline(file_id="TEST_NAME_EMB_1", name="Тестовая река",
                  age_category=["Взрослые"], name_embedding=vec)
    async with GuidelinesStorage() as s:
        await s.upsert_many([g])
        got = await s.get("TEST_NAME_EMB_1")
        await s.delete("TEST_NAME_EMB_1")
    assert got is not None
    assert got.name_embedding is not None
    assert len(got.name_embedding) == 1024
    assert abs(got.name_embedding[0] - 0.1) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_guidelines_name_embedding_storage.py -v`
Expected: FAIL — `_row_to_guideline` has no `name_embedding` / column not selected (or codec error writing the vector). (If `POSTGRES_HOST` unset, it SKIPS — set stand env to actually exercise.)

- [ ] **Step 3: Add codec, own pool, and column read**

In `src/storage/guidelines_storage.py`:

Replace the imports block at the top:

```python
"""GuidelinesStorage — async psycopg3 интерфейс к таблице guidelines."""
from __future__ import annotations

import psycopg.rows
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from .base import BaseStorage, _conninfo
from .models.guideline import Guideline

_COLS = ("file_id, name, mkb, age_category, developer, "
         "nps_status, published_at, usage_status, name_embedding")
```

Update `_row_to_guideline` to populate the vector (pgvector codec returns a numpy array → convert to list):

```python
def _row_to_guideline(row: dict) -> Guideline:
    emb = row.get("name_embedding")
    return Guideline(
        file_id=row["file_id"],
        name=row["name"],
        mkb=list(row["mkb"] or []),
        age_category=list(row["age_category"] or []),
        developer=row["developer"],
        nps_status=row["nps_status"],
        published_at=row["published_at"],
        usage_status=row["usage_status"],
        name_embedding=(list(emb) if emb is not None else None),
    )
```

Add `__aenter__`/`__aexit__`/`_configure_conn` to the class (own pool with codec, mirroring `DocsStorage:56-73`) — put these as the first methods inside `class GuidelinesStorage(BaseStorage):`:

```python
    async def __aenter__(self) -> "GuidelinesStorage":
        self._pool = AsyncConnectionPool(
            conninfo=_conninfo(),
            min_size=1,
            max_size=3,
            open=False,
            configure=self._configure_conn,
            kwargs={"row_factory": psycopg.rows.dict_row},
        )
        await self._pool.open()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._pool.close()

    async def _configure_conn(self, conn: psycopg.AsyncConnection) -> None:
        await register_vector_async(conn)
```

Note: `BaseStorage.__aenter__` used the shared pool; overriding it here gives `GuidelinesStorage` its own codec-configured pool (same trade-off `DocsStorage` already makes).

- [ ] **Step 4: Run test to verify it passes**

Run: `POSTGRES_HOST=<stand> ... python -m pytest tests/test_guidelines_name_embedding_storage.py -v`
Expected: PASS (or SKIP if no stand DB — then verify on stand during rollout).

- [ ] **Step 5: Commit**

```bash
git add src/storage/guidelines_storage.py tests/test_guidelines_name_embedding_storage.py
git commit -m "feat(clinrec): vector-codec + чтение name_embedding в GuidelinesStorage"
```

---

### Task 4: Compute + write `name_embedding` in `upsert_many`

**Files:**
- Modify: `src/storage/guidelines_storage.py` (`upsert_many` INSERT/params + embedding call)
- Test: `tests/test_guidelines_upsert_embeds_name.py`

**Interfaces:**
- Consumes: `name_embed_input` (Task 2), `Guideline.name_embedding` (Task 2), vector-codec pool (Task 3), `embed` from `RAG.retrieval.embeddings`.
- Produces: `upsert_many` computes `embed(name_embed_input(name, age_category))` for **every** row whose `name_embedding` is not already set, writes `name_embedding` in INSERT and `ON CONFLICT DO UPDATE`.

**Design note (why embed every upserted row, not "only if name changed"):** `INSERT ... ON CONFLICT` has no cheap pre-image of the old row, and a read-before-write would race. Seed/reingest run a few times a year over ~747 names — recomputing all is cheap and idempotent. If a caller already set `name_embedding` on the `Guideline` (e.g. engine-side copy), we respect it and skip embedding. `name.strip()`-empty rows still embed the bare string; that's fine (deterministic).

- [ ] **Step 1: Write the failing test** (fake `embed`, no DB — patch the storage's embed and pool)

```python
# tests/test_guidelines_upsert_embeds_name.py
import pytest
from storage.models.guideline import Guideline, name_embed_input


@pytest.mark.asyncio
async def test_upsert_computes_embedding_from_name_and_age(monkeypatch):
    seen_texts = []

    async def fake_embed(text: str) -> list[float]:
        seen_texts.append(text)
        return [0.5] * 1024

    import storage.guidelines_storage as gs
    monkeypatch.setattr(gs, "embed", fake_embed, raising=False)

    # Capture what gets written without a DB: fake the connection/pool.
    written = []

    class FakeConn:
        async def execute(self, sql, params):
            written.append(params)
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class FakePool:
        def connection(self): return FakeConn()

    s = gs.GuidelinesStorage.__new__(gs.GuidelinesStorage)
    s._pool = FakePool()

    g = Guideline(file_id="X1", name="Бронхит", age_category=["Дети"])
    n = await s.upsert_many([g])

    assert n == 1
    assert seen_texts == [name_embed_input("Бронхит", ["Дети"])]  # "Название: Бронхит\nВозрастная группа: [Дети]"
    assert written[0]["name_embedding"] == [0.5] * 1024


@pytest.mark.asyncio
async def test_upsert_respects_preset_embedding(monkeypatch):
    called = False

    async def fake_embed(text: str) -> list[float]:
        nonlocal called
        called = True
        return [0.0] * 1024

    import storage.guidelines_storage as gs
    monkeypatch.setattr(gs, "embed", fake_embed, raising=False)

    written = []

    class FakeConn:
        async def execute(self, sql, params): written.append(params)
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class FakePool:
        def connection(self): return FakeConn()

    s = gs.GuidelinesStorage.__new__(gs.GuidelinesStorage)
    s._pool = FakePool()

    preset = [0.9] * 1024
    g = Guideline(file_id="X2", name="Готовый", name_embedding=preset)
    await s.upsert_many([g])

    assert called is False  # preset embedding must not be recomputed
    assert written[0]["name_embedding"] == preset
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_guidelines_upsert_embeds_name.py -v`
Expected: FAIL — `upsert_many` neither calls `embed` nor writes `name_embedding` (KeyError on `written[0]["name_embedding"]`).

- [ ] **Step 3: Implement embedding in `upsert_many`**

In `src/storage/guidelines_storage.py`, add the import near the top (module level, alongside the others):

```python
from RAG.retrieval.embeddings import embed
```

Rewrite `upsert_many` to compute the embedding before the INSERT and include the column:

```python
    async def upsert_many(self, rows: list[Guideline]) -> int:
        if not rows:
            return 0
        written = 0
        async with self._pool.connection() as conn:
            for g in rows:
                if g.name_embedding is None:
                    g.name_embedding = await embed(name_embed_input(g.name, g.age_category))
                await conn.execute(
                    """
                    INSERT INTO guidelines
                        (file_id, name, mkb, age_category, developer,
                         nps_status, published_at, usage_status, name_embedding)
                    VALUES
                        (%(file_id)s, %(name)s, %(mkb)s, %(age_category)s, %(developer)s,
                         %(nps_status)s, %(published_at)s, %(usage_status)s, %(name_embedding)s)
                    ON CONFLICT (file_id) DO UPDATE SET
                        name           = EXCLUDED.name,
                        mkb            = EXCLUDED.mkb,
                        age_category   = EXCLUDED.age_category,
                        developer      = EXCLUDED.developer,
                        nps_status     = EXCLUDED.nps_status,
                        published_at   = EXCLUDED.published_at,
                        usage_status   = EXCLUDED.usage_status,
                        name_embedding = EXCLUDED.name_embedding
                    """,
                    {
                        "file_id": g.file_id,
                        "name": g.name,
                        "mkb": g.mkb,
                        "age_category": g.age_category,
                        "developer": g.developer,
                        "nps_status": g.nps_status,
                        "published_at": g.published_at,
                        "usage_status": g.usage_status,
                        "name_embedding": g.name_embedding,
                    },
                )
                written += 1
        return written
```

Add the `name_embed_input` import to the existing model import line:

```python
from .models.guideline import Guideline, name_embed_input
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_guidelines_upsert_embeds_name.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/storage/guidelines_storage.py tests/test_guidelines_upsert_embeds_name.py
git commit -m "feat(clinrec): upsert_many считает и пишет name_embedding"
```

---

### Task 5: Full-suite regression + stand rollout note

**Files:**
- None (verification only). Optionally: `docs/superpowers/plans/2026-07-20-clinrec-name-embedding-medkard.md` (check off items).

- [ ] **Step 1: Run the guideline-related unit tests together**

Run: `python -m pytest tests/test_migration_025.py tests/test_guideline_name_embed_input.py tests/test_guidelines_upsert_embeds_name.py -v`
Expected: PASS. (The `*_storage.py` DB test SKIPs without stand env — that's expected off-stand.)

- [ ] **Step 2: Sanity-check import wiring**

Run: `python -c "import sys; sys.path.insert(0,'src'); from storage.guidelines_storage import GuidelinesStorage; from storage.models.guideline import name_embed_input; print('ok')"`
Expected: prints `ok` (no circular-import or missing-symbol error from the new `RAG.retrieval.embeddings` import inside storage).

- [ ] **Step 3: Record stand rollout steps (do NOT run here — stand only)**

Rollout on the stand, in order:
1. `bash migrations/migrate.sh` — applies `025_guidelines_name_embedding.sql`.
2. Re-populate names: run `python scripts/seed-guidelines.py` (or a reingest pass) — `upsert_many` fills `name_embedding` for every row from `manifest.csv`. ~747 names, fast.
3. Verify: `SELECT count(*) FROM guidelines WHERE name_embedding IS NULL;` → expect `0`.

This must complete on the stand **before** the engine sync runs (engine copies these vectors).

- [ ] **Step 4: Commit plan check-offs (if any)**

```bash
git add docs/superpowers/plans/2026-07-20-clinrec-name-embedding-medkard.md
git commit -m "chore(clinrec): medkard name-embedding plan progress"
```
