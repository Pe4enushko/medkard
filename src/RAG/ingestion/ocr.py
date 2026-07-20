"""OCR fallback for PDF pages with no extractable text layer (scans).

Rasterizes a pymupdf page and runs Tesseract (Russian) over it. Used by
data_loader.PDFContentReader.iter_chunks() when a page's native
page.get_text() output is below a character threshold.
"""
import io
import shutil

import fitz  # pymupdf
import pytesseract
from PIL import Image

_TESSERACT_LANG = "rus"
_OCR_DPI = 300
_TESSERACT_CONFIG = "--psm 3"


def ensure_tesseract_available() -> None:
    """Raise RuntimeError if the `tesseract` binary isn't on PATH.

    Called once at the start of ingestion (load_documents) rather than
    per-page: a missing binary means every scanned page would fail
    identically, so this fails the whole run loudly instead of producing
    N repeated per-page log lines.
    """
    if shutil.which("tesseract") is None:
        raise RuntimeError(
            "tesseract binary not found on PATH — required for OCR fallback on "
            "scanned PDF pages. Install it (Debian/Ubuntu: "
            "apt-get install -y tesseract-ocr tesseract-ocr-rus) and retry."
        )
