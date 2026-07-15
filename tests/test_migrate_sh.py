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


def test_help_prints_usage_and_exits_zero(tmp_path):
    env, migs, log = _harness(tmp_path, ["001_a.sql"])
    r = _run(env, migs, "--help")
    assert "Usage" in r.stdout
    assert "--skip-until" in r.stdout
    # help short-circuits before touching the DB
    assert (not log.exists()) or log.read_text() == ""


def test_help_works_without_env(tmp_path):
    env, migs, _ = _harness(tmp_path, ["001_a.sql"])
    (migs.parent / ".env").unlink()  # help must not require a configured DB
    r = subprocess.run(
        ["bash", str(migs / "migrate.sh"), "-h"],
        env=env, capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "Usage" in r.stdout


def test_skip_until_unknown_arg_rejected(tmp_path):
    env, migs, _ = _harness(tmp_path, ["001_a.sql"])
    r = subprocess.run(
        ["bash", str(migs / "migrate.sh"), "--bogus"],
        env=env, capture_output=True, text=True,
    )
    assert r.returncode != 0
