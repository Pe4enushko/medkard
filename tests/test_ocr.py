from unittest.mock import patch

import fitz
import pytest

from RAG.ingestion.ocr import ensure_tesseract_available, ocr_page


def test_ensure_tesseract_available_raises_when_missing():
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="tesseract"):
            ensure_tesseract_available()


def test_ensure_tesseract_available_passes_when_present():
    with patch("shutil.which", return_value="/usr/bin/tesseract"):
        ensure_tesseract_available()  # must not raise


def _page_with_text(text: str) -> fitz.Page:
    """A single-page in-memory PDF with `text` drawn on it, for OCR round-trip testing."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_font(fontname="china-s")
    page.insert_text((72, 72), text, fontsize=24, fontname="china-s")
    return page


def test_ocr_page_round_trips_known_text():
    page = _page_with_text("Привет мир")
    result = ocr_page(page)
    assert "привет" in result.lower() or "мир" in result.lower()


def test_ocr_page_returns_empty_string_on_blank_page():
    doc = fitz.open()
    page = doc.new_page()
    result = ocr_page(page)
    assert result == ""
