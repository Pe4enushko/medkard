from pathlib import Path

from RAG.ingestion.data_loader import resolve_pdf_path


def test_new_convention_primary_match(tmp_path: Path):
    (tmp_path / "КР1027.pdf").write_bytes(b"%PDF-1.4")
    got = resolve_pdf_path("1027_1", tmp_path)
    assert got == tmp_path / "КР1027.pdf"


def test_new_convention_ignores_revision_suffix(tmp_path: Path):
    (tmp_path / "КР1027.pdf").write_bytes(b"%PDF-1.4")
    assert resolve_pdf_path("1027_1", tmp_path) == tmp_path / "КР1027.pdf"
    assert resolve_pdf_path("1027_2", tmp_path) == tmp_path / "КР1027.pdf"


def test_falls_back_to_old_convention_when_new_missing(tmp_path: Path):
    (tmp_path / "1027_1.pdf").write_bytes(b"%PDF-1.4")
    got = resolve_pdf_path("1027_1", tmp_path)
    assert got == tmp_path / "1027_1.pdf"


def test_prefers_new_convention_when_both_present(tmp_path: Path):
    (tmp_path / "КР1027.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "1027_1.pdf").write_bytes(b"%PDF-1.4")
    got = resolve_pdf_path("1027_1", tmp_path)
    assert got == tmp_path / "КР1027.pdf"


def test_returns_none_when_neither_exists(tmp_path: Path):
    assert resolve_pdf_path("1027_1", tmp_path) is None


def test_non_numeric_id_uses_fallback_only(tmp_path: Path):
    (tmp_path / "A.pdf").write_bytes(b"%PDF-1.4")
    got = resolve_pdf_path("A", tmp_path)
    assert got == tmp_path / "A.pdf"
