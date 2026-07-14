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
