import csv
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "check_guidelines_integrity",
    Path(__file__).resolve().parent.parent / "scripts" / "checks" / "check-guidelines-integrity.py")
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


def _write_manifest(path: Path, ids: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ID", "Наименование"])
        w.writeheader()
        for i in ids:
            w.writerow({"ID": i, "Наименование": f"name-{i}"})


def test_read_manifest_ids_strips_blank_rows(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, ["1027_1", "", "  ", "340_2"])
    assert check._read_manifest_ids(manifest) == ["1027_1", "340_2"]


def test_read_manifest_ids_preserves_order_and_duplicates(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, ["3_2", "1_4", "3_2"])
    assert check._read_manifest_ids(manifest) == ["3_2", "1_4", "3_2"]
