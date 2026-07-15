# Migration Ledger + Baseline + Docs Reconcile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `migrate.sh` an applied-migrations ledger with a `--skip-until` baseline mode, and rewrite migration 024 into a guarded reconcile that converges the drifted live `docs` schema to a single `embedding` column.

**Architecture:** A `schema_migrations` ledger table (bootstrapped inline by `migrate.sh`) lets the runner apply only un-applied files instead of replaying from 001. `--skip-until FILE` records files sorting strictly before FILE as applied without running them (one-time baseline of the drifted stand DB). Migration `024_docs_reconcile.sql` drops the dead HyDE columns from *both* the file-declared scheme (`fact_q_*`) and the untracked live scheme (`chunk_embedding`/`hyde_reembedded`), keeps `embedding` + its index, and refuses to run if any row still holds a vector only in `chunk_embedding`.

**Tech Stack:** Bash + `psql` (migrate.sh), PostgreSQL + pgvector (migrations), pytest (dev-machine bash-logic tests via a stubbed `psql`).

## Global Constraints

- Migrations are **forward-only and idempotent** (`migrate.sh` never runs down-SQL).
- Do **not** rewrite historical migrations 001–023 (including 002). The ledger keeps them from re-running on the stand.
- Vector dimension is **`VECTOR(1024)`**; HNSW params **`m = 16, ef_construction = 64`**; the live embedding index is named **`docs_embedding_idx`**. Reuse these exact values — they already exist on the stand.
- `--skip-until FILE`: record every migration file whose basename sorts **strictly before** FILE as applied **without applying**; apply normally from FILE onward. Filenames are 3-digit zero-padded, so lexicographic `<` equals numeric order.
- Applying a migration and recording it in `schema_migrations` must be **atomic** (one `psql --single-transaction` invocation): a failed file must leave no ledger row.
- Migration 024 must carry a **data-loss guard**: if `chunk_embedding` and `embedding` both exist and any row has `embedding IS NULL AND chunk_embedding IS NOT NULL`, `RAISE EXCEPTION` (do not drop `chunk_embedding`).
- **No Postgres on the dev machine.** Dev-machine tests exercise `migrate.sh` control flow with a stubbed `psql` and static assertions on SQL text. SQL correctness against a real DB is a stand-only checklist (spec §5), not part of this plan's automated tests.

---

### Task 1: Reconcile migration `024_docs_reconcile.sql`

Rename the placeholder-schema 024 to the real reconcile migration and cover it with a static-text test (no DB needed).

**Files:**
- Rename: `migrations/024_docs_single_embedding.sql` → `migrations/024_docs_reconcile.sql` (content fully replaced)
- Modify: `tests/test_migration_024.py` (rewrite to assert the reconcile SQL text)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: file `migrations/024_docs_reconcile.sql` — the exact basename `024_docs_reconcile.sql` is referenced by the stand rollout (`--skip-until 024_docs_reconcile.sql`) and by Task 4's runbook.

- [ ] **Step 1: Rename the migration file (preserve history)**

```bash
cd /home/savoy/projects/medkard-remove-hyde
git mv migrations/024_docs_single_embedding.sql migrations/024_docs_reconcile.sql
```

- [ ] **Step 2: Replace the migration content**

Overwrite `migrations/024_docs_reconcile.sql` with exactly:

```sql
-- 024_docs_reconcile.sql
-- Reconcile the docs schema to a single contextual embedding.
-- Converges BOTH origins to one clean shape:
--   * fresh DB (fact_q_* columns/indexes created by 002)
--   * drifted stand DB (untracked chunk_embedding + hyde_reembedded + docs_chunk_embedding_hnsw)
-- Forward-only, idempotent. VECTOR dim stays 1024 (Qwen3-Embedding-0.6B).

-- Data-loss guard: only meaningful when both columns exist (live stand). On a fresh DB
-- chunk_embedding is absent -> skipped; docs is empty anyway.
DO $$
DECLARE
    unmigrated int;
BEGIN
    IF (SELECT count(*) FROM information_schema.columns
        WHERE table_name = 'docs'
          AND column_name IN ('chunk_embedding', 'embedding')) = 2 THEN
        SELECT count(*) INTO unmigrated
        FROM docs
        WHERE embedding IS NULL AND chunk_embedding IS NOT NULL;
        IF unmigrated > 0 THEN
            RAISE EXCEPTION
                '% строк docs держат вектор только в chunk_embedding — прогони reingest --force-all до этой миграции',
                unmigrated;
        END IF;
    END IF;
END$$;

DROP INDEX IF EXISTS docs_fact_q_embedding_idx;
DROP INDEX IF EXISTS docs_procedure_q_embedding_idx;
DROP INDEX IF EXISTS docs_constraint_q_embedding_idx;
DROP INDEX IF EXISTS docs_chunk_embedding_hnsw;

ALTER TABLE docs
    DROP COLUMN IF EXISTS fact_q,
    DROP COLUMN IF EXISTS procedure_q,
    DROP COLUMN IF EXISTS constraint_q,
    DROP COLUMN IF EXISTS fact_q_embedding,
    DROP COLUMN IF EXISTS procedure_q_embedding,
    DROP COLUMN IF EXISTS constraint_q_embedding,
    DROP COLUMN IF EXISTS chunk_embedding,
    DROP COLUMN IF EXISTS hyde_reembedded,
    ADD COLUMN IF NOT EXISTS embedding VECTOR(1024);

CREATE INDEX IF NOT EXISTS docs_embedding_idx
    ON docs USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

- [ ] **Step 3: Write the failing static test**

Replace the entire contents of `tests/test_migration_024.py` with:

```python
"""Static assertions on the reconcile migration SQL (no DB required)."""
from pathlib import Path

SQL = (Path(__file__).resolve().parent.parent
       / "migrations" / "024_docs_reconcile.sql").read_text()


def test_old_single_embedding_file_is_gone():
    old = (Path(__file__).resolve().parent.parent
           / "migrations" / "024_docs_single_embedding.sql")
    assert not old.exists()


def test_drops_live_stand_hyde_columns():
    assert "DROP COLUMN IF EXISTS chunk_embedding" in SQL
    assert "DROP COLUMN IF EXISTS hyde_reembedded" in SQL
    assert "DROP INDEX IF EXISTS docs_chunk_embedding_hnsw" in SQL


def test_drops_fresh_db_hyde_columns():
    for col in ("fact_q", "procedure_q", "constraint_q",
                "fact_q_embedding", "procedure_q_embedding", "constraint_q_embedding"):
        assert f"DROP COLUMN IF EXISTS {col}" in SQL


def test_keeps_single_embedding_and_index():
    assert "ADD COLUMN IF NOT EXISTS embedding VECTOR(1024)" in SQL
    assert "CREATE INDEX IF NOT EXISTS docs_embedding_idx" in SQL
    assert "hnsw (embedding vector_cosine_ops)" in SQL


def test_has_data_loss_guard():
    assert "RAISE EXCEPTION" in SQL
    assert "embedding IS NULL AND chunk_embedding IS NOT NULL" in SQL
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/savoy/projects/medkard-remove-hyde && python3 -m pytest tests/test_migration_024.py -v`
Expected: PASS (5 passed). If any FAIL, the SQL text in Step 2 does not match the assertions — fix the SQL, not the test.

- [ ] **Step 5: Commit**

```bash
git add migrations/024_docs_reconcile.sql migrations/024_docs_single_embedding.sql tests/test_migration_024.py
git commit -m "refactor(db): rewrite 024 as guarded docs reconcile (drop chunk_embedding/hyde_reembedded)"
```

---

### Task 2: `migrate.sh` ledger — bootstrap, skip-applied, atomic record

Route all `psql` calls through helpers, bootstrap the `schema_migrations` ledger, and apply only un-applied files (recording each atomically). A stubbed-`psql` pytest verifies the control flow on the dev machine.

**Files:**
- Modify: `migrations/migrate.sh` (replace the apply loop and its helpers; keep the `.env` loader unchanged)
- Test: `tests/test_migrate_sh.py` (new)

**Interfaces:**
- Consumes: nothing from Task 1 at runtime (Task 1 only renames a file the loop happens to pick up).
- Produces: shell functions `run_psql`, `is_applied`, `record_only`, `apply_and_record`; ledger table `schema_migrations(filename text PRIMARY KEY, applied_at timestamptz)`. Task 3 adds `--skip-until` parsing into this same script and reuses `record_only`.

- [ ] **Step 1: Write the failing test (fake psql harness)**

Create `tests/test_migrate_sh.py`:

```python
"""Exercise migrate.sh control flow with a stubbed psql (no real Postgres)."""
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REAL_MIGRATE = REPO / "migrations" / "migrate.sh"

FAKE_PSQL = r"""#!/usr/bin/env bash
# Logs every invocation (args joined by |) to $PSQL_LOG.
# For the is_applied SELECT, prints "1" iff the queried filename is in $APPLIED (comma list).
echo "$*" >> "$PSQL_LOG"
for a in "$@"; do :; done
joined="$*"
if [[ "$joined" == *"SELECT 1 FROM schema_migrations"* ]]; then
    name="$(printf '%s' "$joined" | sed -n "s/.*filename = '\([^']*\)'.*/\1/p")"
    IFS=',' read -ra applied <<< "${APPLIED:-}"
    for x in "${applied[@]}"; do
        [[ "$x" == "$name" ]] && { echo "1"; exit 0; }
    done
fi
exit 0
"""


def _harness(tmp_path, migration_names):
    """Build a temp migrations/ dir with copied migrate.sh, fake .env, empty .sql files,
    and a fake psql on PATH. Returns (env, migrations_dir)."""
    proj = tmp_path / "proj"
    migs = proj / "migrations"
    migs.mkdir(parents=True)
    shutil.copy(REAL_MIGRATE, migs / "migrate.sh")
    (proj / ".env").write_text(
        "POSTGRES_HOST=h\nPOSTGRES_PORT=5432\nPOSTGRES_DB=d\n"
        "POSTGRES_USER=u\nPOSTGRES_PASSWORD=p\n"
    )
    for n in migration_names:
        (migs / n).write_text("SELECT 1;\n")

    bindir = tmp_path / "bin"
    bindir.mkdir()
    psql = bindir / "psql"
    psql.write_text(FAKE_PSQL)
    psql.chmod(0o755)

    log = tmp_path / "psql.log"
    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["PSQL_LOG"] = str(log)
    return env, migs, log


def _run(env, migs, *args):
    r = subprocess.run(
        ["bash", str(migs / "migrate.sh"), *args],
        env=env, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    return r


def test_bootstraps_ledger_table(tmp_path):
    env, migs, log = _harness(tmp_path, ["001_a.sql", "002_b.sql"])
    _run(env, migs)
    text = log.read_text()
    assert "CREATE TABLE IF NOT EXISTS schema_migrations" in text


def test_applies_unrecorded_and_records_atomically(tmp_path):
    env, migs, log = _harness(tmp_path, ["001_a.sql", "002_b.sql"])
    env["APPLIED"] = ""  # nothing applied yet
    _run(env, migs)
    lines = log.read_text().splitlines()
    # each unrecorded file applied via -f AND recorded via INSERT, in one --single-transaction call
    applied = [l for l in lines if "--single-transaction" in l and "-f" in l]
    assert any("001_a.sql" in l and "INSERT INTO schema_migrations" in l for l in applied)
    assert any("002_b.sql" in l and "INSERT INTO schema_migrations" in l for l in applied)


def test_skips_already_recorded_file(tmp_path):
    env, migs, log = _harness(tmp_path, ["001_a.sql", "002_b.sql"])
    env["APPLIED"] = "001_a.sql"  # 001 already in ledger
    _run(env, migs)
    lines = log.read_text().splitlines()
    # 001 must NOT be applied (no -f for it); 002 must be applied
    assert not any("-f" in l and "001_a.sql" in l for l in lines)
    assert any("--single-transaction" in l and "002_b.sql" in l for l in lines)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/savoy/projects/medkard-remove-hyde && python3 -m pytest tests/test_migrate_sh.py -v`
Expected: FAIL — the current `migrate.sh` neither bootstraps the ledger nor records/skips (it applies every file with `--file`, no `INSERT`, no `--single-transaction`).

- [ ] **Step 3: Rewrite the apply loop in `migrate.sh`**

In `migrations/migrate.sh`, keep everything up to and including `export PGPASSWORD="$POSTGRES_PASSWORD"` unchanged. Replace from the line `echo "Running migrations against ..."` to the end of the file with:

```bash
echo "Running migrations against $POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB ..."

run_psql() {
    psql \
        --host="$POSTGRES_HOST" \
        --port="$POSTGRES_PORT" \
        --dbname="$POSTGRES_DB" \
        --username="$POSTGRES_USER" \
        --set=ON_ERROR_STOP=1 \
        "$@"
}

# Ledger of applied migrations. Bootstrapped here because it cannot record itself.
run_psql -c "CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);"

is_applied() {  # $1 = basename
    [[ "$(run_psql -tA -c "SELECT 1 FROM schema_migrations WHERE filename = '$1'")" == "1" ]]
}

record_only() {  # $1 = basename — mark applied without running the file
    run_psql -c "INSERT INTO schema_migrations(filename) VALUES ('$1') ON CONFLICT (filename) DO NOTHING;"
}

apply_and_record() {  # $1 = path, $2 = basename — file + ledger row in ONE transaction
    run_psql --single-transaction -f "$1" \
        -c "INSERT INTO schema_migrations(filename) VALUES ('$2');"
}

for sql_file in "$SCRIPT_DIR"/[0-9]*.sql; do
    name="$(basename "$sql_file")"
    if is_applied "$name"; then
        echo "  skip (already applied) $name"
        continue
    fi
    echo "  Applying $name ..."
    apply_and_record "$sql_file" "$name"
done

echo "All migrations applied."
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/savoy/projects/medkard-remove-hyde && python3 -m pytest tests/test_migrate_sh.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add migrations/migrate.sh tests/test_migrate_sh.py
git commit -m "feat(migrate): schema_migrations ledger — apply only un-applied files, atomic record"
```

---

### Task 3: `migrate.sh --skip-until FILE` baseline mode

Add a one-time baseline mode that records files sorting strictly before FILE as applied without running them, then applies from FILE onward.

**Files:**
- Modify: `migrations/migrate.sh` (add arg parsing + baseline branch in the loop)
- Test: `tests/test_migrate_sh.py` (add cases)

**Interfaces:**
- Consumes: `run_psql`, `is_applied`, `record_only`, `apply_and_record` from Task 2.
- Produces: CLI flag `--skip-until FILE` on `migrate.sh`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_migrate_sh.py`:

```python
def test_skip_until_records_earlier_files_without_applying(tmp_path):
    env, migs, log = _harness(
        tmp_path, ["001_a.sql", "002_b.sql", "023_x.sql", "024_docs_reconcile.sql"]
    )
    env["APPLIED"] = ""
    _run(env, migs, "--skip-until", "024_docs_reconcile.sql")
    lines = log.read_text().splitlines()

    # 001/002/023 recorded via ON CONFLICT INSERT, never applied with -f
    for earlier in ("001_a.sql", "002_b.sql", "023_x.sql"):
        assert any("ON CONFLICT" in l and earlier in l for l in lines), earlier
        assert not any("-f" in l and earlier in l for l in lines), earlier

    # 024 (== FILE) is applied for real
    assert any("--single-transaction" in l and "024_docs_reconcile.sql" in l for l in lines)


def test_skip_until_unknown_arg_rejected(tmp_path):
    env, migs, _ = _harness(tmp_path, ["001_a.sql"])
    r = subprocess.run(
        ["bash", str(migs / "migrate.sh"), "--bogus"],
        env=env, capture_output=True, text=True,
    )
    assert r.returncode != 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/savoy/projects/medkard-remove-hyde && python3 -m pytest tests/test_migrate_sh.py -v -k skip_until`
Expected: FAIL — `--skip-until` is not parsed; `--bogus` is currently ignored (returncode 0).

- [ ] **Step 3: Add arg parsing and the baseline branch**

In `migrations/migrate.sh`, insert argument parsing immediately **before** the `echo "Running migrations against ..."` line:

```bash
SKIP_UNTIL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-until) SKIP_UNTIL="${2:?--skip-until needs a filename}"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
```

Then, in the `for` loop, replace the `if is_applied "$name"; then ... fi` block with:

```bash
    if [[ -n "$SKIP_UNTIL" && "$name" < "$SKIP_UNTIL" ]]; then
        echo "  baseline (record only) $name"
        record_only "$name"
        continue
    fi
    if is_applied "$name"; then
        echo "  skip (already applied) $name"
        continue
    fi
```

(The `echo "  Applying $name ..."` + `apply_and_record` lines that follow stay unchanged.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/savoy/projects/medkard-remove-hyde && python3 -m pytest tests/test_migrate_sh.py -v`
Expected: PASS (all 5). The Task 2 cases still pass (no `--skip-until` ⇒ `SKIP_UNTIL` empty ⇒ baseline branch never taken).

- [ ] **Step 5: Commit**

```bash
git add migrations/migrate.sh tests/test_migrate_sh.py
git commit -m "feat(migrate): --skip-until FILE to baseline a drifted DB without re-running history"
```

---

### Task 4: Rollout runbook

Document the exact stand rollout order so the operator does not run 024 before the re-embed (which the guard would reject).

**Files:**
- Create: `migrations/RUNBOOK.md`

**Interfaces:**
- Consumes: the `--skip-until 024_docs_reconcile.sql` flag (Task 3) and the guarded 024 (Task 1).
- Produces: operator-facing runbook. No code depends on it.

- [ ] **Step 1: Write the runbook**

Create `migrations/RUNBOOK.md`:

```markdown
# Migrations runbook

`migrate.sh` keeps an applied-migrations ledger (`schema_migrations`) and applies only
files not yet recorded. It never re-runs history.

## Fresh database

```bash
bash migrations/migrate.sh
```

Applies every migration in order and records each. No baseline needed.

## Existing / drifted database (one-time baseline)

The stand DB predates the ledger and its `docs` schema drifted from the files (it has an
untracked `chunk_embedding` + `hyde_reembedded`, not the files' `fact_q_*`). Baseline it so
history is not replayed, then run the reconcile:

```bash
# 1. Deploy the remove-hyde code (reads/writes docs.embedding).
# 2. Mark 001–023 applied WITHOUT running them:
bash migrations/migrate.sh --skip-until 024_docs_reconcile.sql
# 3. Fill docs.embedding for every row:
python scripts/reingest-pdfs.py --force-all
# 4. Apply 024 (drops chunk_embedding/hyde_reembedded, keeps embedding):
bash migrations/migrate.sh
```

Order 3→4 is mandatory: `024_docs_reconcile.sql` aborts with a clear error if any row still
holds its vector only in `chunk_embedding`. Skipping the re-embed fails step 4 loudly rather
than dropping live vectors.

## Verify

```sql
\d docs
```

Expect: no `chunk_embedding` / `hyde_reembedded` / `fact_q_*`; `embedding VECTOR(1024)` present
with index `docs_embedding_idx`. Then spot-check retrieval.
```

- [ ] **Step 2: Commit**

```bash
git add migrations/RUNBOOK.md
git commit -m "docs(migrate): stand rollout runbook (baseline + reingest + reconcile order)"
```

---

## Notes for the executor

- **Baseline of Task 1's tests already covered by remove-hyde work.** The pure test suite in this worktree is green except for pre-existing infra-absent failures (no Postgres/fastapi/OPENAI env). Run new tests file-scoped (`pytest tests/test_migrate_sh.py tests/test_migration_024.py -v`); do not treat the infra-absent failures as regressions.
- **`bash` string `<` is lexicographic and locale-sensitive.** All migration basenames use a 3-digit zero-padded prefix, so lexicographic order equals numeric order. Do not "fix" this with numeric parsing — it would break on the multi-file `013_*` prefix pair.
- **Stand-only checklist (not automated here, spec §5):** ledger created; second `migrate.sh` run skips all; `--skip-until` records 23 rows and applies only 024; 024 idempotent on re-run; guard fires when an unmigrated row exists; fresh-DB full run converges `docs` to single-embedding.
