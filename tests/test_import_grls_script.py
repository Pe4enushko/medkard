import importlib.util
import sys
import zipfile
from datetime import date
from pathlib import Path

from grls import status as st
from grls.dump import read_dump
from tests.grls_fixtures import make_sheet, sample_row

_spec = importlib.util.spec_from_file_location(
    "import_grls", Path(__file__).resolve().parent.parent / "scripts" / "import-grls.py")
imp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = imp  # dataclasses (PEP 563 annotations) needs the module registered
_spec.loader.exec_module(imp)


def _archive(tmp_path: Path) -> Path:
    d = tmp_path / "x"
    d.mkdir()
    dup = sample_row()
    make_sheet(d / "1.xlsx", st.STATUS_ACTIVE, [dup, dup, sample_row(reg_number="ЛП-000002")])
    make_sheet(d / "2.xlsx", st.STATUS_CHANGED, [sample_row()])
    make_sheet(d / "3.xlsx", st.STATUS_EXPIRED, [sample_row(reg_number="ЛП-000003", expires_at="31.12.2025")])
    z = tmp_path / "grls2026-08-17-1.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in sorted(d.iterdir()):
            zf.write(f, arcname=f.name)
    return z


def test_plan_import_dedups_and_counts(tmp_path: Path):
    from grls.parser import read_archive
    plan = imp.plan_import(read_archive(_archive(tmp_path)))
    assert plan.registry_date == date(2026, 8, 17)
    assert plan.status_counts == {st.STATUS_ACTIVE: 2, st.STATUS_EXPIRED: 1}
    assert plan.duplicates_dropped == 1
    assert plan.skipped_files and "2.xlsx" in plan.skipped_files[0]
    assert len(plan.records) == 3


def test_dry_run_prints_summary_and_writes_nothing(tmp_path: Path, capsys):
    rc = imp.main([str(_archive(tmp_path)), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "registry_date: 2026-08-17" in out
    assert f"{st.STATUS_ACTIVE}: 2" in out
    assert "dry-run" in out


def test_make_dump_without_db(tmp_path: Path):
    dump = tmp_path / "grls.jsonl.gz"
    rc = imp.main([str(_archive(tmp_path)), "--dry-run", "--make-dump", str(dump)])
    assert rc == 0
    meta, records = read_dump(dump)
    assert meta["row_count"] == 3
    assert meta["archive_name"] == "grls2026-08-17-1.zip"
    assert len(records) == 3
