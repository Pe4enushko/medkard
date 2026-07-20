from pathlib import Path
from unittest.mock import patch

import fitz

from RAG.ingestion.data_loader import PDFContentReader


_CYRILLIC_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _make_scanned_pdf(tmp_path: Path, text: str) -> Path:
    """A single-page PDF with `text` rasterized into an image (no text layer) —
    mimics a scanned document: page.get_text() returns "" for it.

    Uses a DejaVu Sans fontfile explicitly: pymupdf's base14 fonts (the
    default for insert_text()) have no Cyrillic glyphs, which would
    silently render Russian text as placeholder boxes/dots — invisible to
    OCR and defeating the point of this fixture.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=24, fontname="DejaVuSans", fontfile=_CYRILLIC_FONT)
    pix = page.get_pixmap(dpi=150)
    img_doc = fitz.open()
    img_page = img_doc.new_page(width=page.rect.width, height=page.rect.height)
    img_page.insert_image(img_page.rect, pixmap=pix)
    out = tmp_path / "scanned.pdf"
    img_doc.save(out)
    return out


def test_iter_chunks_ocrs_page_with_no_text_layer(tmp_path: Path):
    pdf_path = _make_scanned_pdf(tmp_path, "Клинические рекомендации")
    reader = PDFContentReader(pdf_path, metadata={"ID": "TEST_1"})
    chunks = list(reader.iter_chunks())
    text_chunks = [c for c in chunks if c["type"] == "text"]
    assert len(text_chunks) >= 1
    combined = " ".join(c["content"] for c in text_chunks).lower()
    assert "клинические" in combined or "рекомендации" in combined


def test_iter_chunks_ocr_failure_yields_no_text_not_exception(tmp_path: Path):
    pdf_path = _make_scanned_pdf(tmp_path, "Текст страницы")
    reader = PDFContentReader(pdf_path, metadata={"ID": "TEST_2"})
    with patch("RAG.ingestion.data_loader.ocr.ocr_page", return_value=""):
        chunks = list(reader.iter_chunks())  # must not raise
    assert chunks == []  # no text extracted, no table on this page -> no chunks
