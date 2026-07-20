from unittest.mock import patch

import pytest

from RAG.ingestion.ocr import ensure_tesseract_available


def test_ensure_tesseract_available_raises_when_missing():
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="tesseract"):
            ensure_tesseract_available()


def test_ensure_tesseract_available_passes_when_present():
    with patch("shutil.which", return_value="/usr/bin/tesseract"):
        ensure_tesseract_available()  # must not raise
