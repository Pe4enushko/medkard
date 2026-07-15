import csv
from pathlib import Path

from RAG.ingestion.data_loader import load_documents


def _make_manifest(dir_: Path, ids: list[str]) -> Path:
    mpath = dir_ / "manifest.csv"
    with open(mpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ID", "Наименование"])
        w.writeheader()
        for i in ids:
            w.writerow({"ID": i, "Наименование": f"name-{i}"})
    return mpath


def test_only_yields_selected_ids(tmp_path: Path):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    for i in ["A", "B", "C"]:
        (pdfs / f"{i}.pdf").write_bytes(b"%PDF-1.4")
    manifest = _make_manifest(tmp_path, ["A", "B", "C"])

    got = [r.metadata["ID"] for r in load_documents(manifest_path=manifest, pdfs_dir=pdfs, only={"B"})]
    assert got == ["B"]


def test_only_and_exceptions_combine(tmp_path: Path):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    for i in ["A", "B"]:
        (pdfs / f"{i}.pdf").write_bytes(b"%PDF-1.4")
    manifest = _make_manifest(tmp_path, ["A", "B"])

    got = [r.metadata["ID"] for r in
           load_documents(manifest_path=manifest, pdfs_dir=pdfs, only={"A", "B"}, exceptions={"A"})]
    assert got == ["B"]
