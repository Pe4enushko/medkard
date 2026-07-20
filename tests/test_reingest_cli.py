import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "reingest_pdfs", Path(__file__).resolve().parent.parent / "scripts" / "reingest-pdfs.py")
reingest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reingest)


def test_resolve_paths_data_dir(tmp_path: Path):
    pdfs, manifest = reingest._resolve_paths(tmp_path, None)
    assert pdfs == tmp_path
    assert manifest == tmp_path / "manifest.csv"


def test_resolve_paths_default():
    from RAG.ingestion.data_loader import MANIFEST_PATH, PDFS_DIR
    pdfs, manifest = reingest._resolve_paths(None, None)
    assert pdfs == PDFS_DIR
    assert manifest == MANIFEST_PATH


def test_resolve_paths_manifest_overrides_data_dir(tmp_path: Path):
    explicit = tmp_path / "resources" / "manifest.csv"
    pdfs, manifest = reingest._resolve_paths(tmp_path, explicit)
    assert pdfs == tmp_path          # PDFs still from data-dir
    assert manifest == explicit      # manifest from --manifest-path


def test_resolve_paths_manifest_only():
    from RAG.ingestion.data_loader import PDFS_DIR
    explicit = Path("/some/where/m.csv")
    pdfs, manifest = reingest._resolve_paths(None, explicit)
    assert pdfs == PDFS_DIR          # default PDFs
    assert manifest == explicit


def test_summarize_counts():
    wl = [("A", "full"), ("B", "full"), ("C", "skip"), ("D", "metadata_only")]
    assert reingest._summarize(wl) == {"full": 2, "skip": 1, "metadata_only": 1}


def test_forced_full_worklist_marks_found_files_full(tmp_path: Path):
    (tmp_path / "A.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "B.pdf").write_bytes(b"%PDF-1.4")
    manifest_rows = {"A": {"ID": "A"}, "B": {"ID": "B"}}
    wl, absent = reingest._forced_full_worklist(manifest_rows, tmp_path)
    assert wl == [("A", "full"), ("B", "full")]
    assert absent == []


def test_forced_full_worklist_reports_missing_pdf_with_no_rag_data(tmp_path: Path):
    (tmp_path / "A.pdf").write_bytes(b"%PDF-1.4")
    manifest_rows = {"A": {"ID": "A"}, "B": {"ID": "B"}}
    wl, absent = reingest._forced_full_worklist(manifest_rows, tmp_path)
    assert wl == [("A", "full")]
    assert absent == ["B"]  # no PDF, and no docs_file_ids given -> genuinely absent


def test_forced_full_worklist_does_not_report_missing_pdf_already_in_rag(tmp_path: Path):
    (tmp_path / "A.pdf").write_bytes(b"%PDF-1.4")
    manifest_rows = {"A": {"ID": "A"}, "B": {"ID": "B"}}
    # B has no PDF here but already has chunks in docs -> not "absent from RAG"
    wl, absent = reingest._forced_full_worklist(manifest_rows, tmp_path, docs_file_ids={"B"})
    assert wl == [("A", "full")]
    assert absent == []


def test_force_all_flag_parses():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-all", action="store_true")
    assert parser.parse_args(["--force-all"]).force_all is True
