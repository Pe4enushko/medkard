# Scanned PDF OCR Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guideline PDFs with no text layer (scans, ~22 of ~700) currently produce zero chunks during ingestion, silently. Add a per-page OCR fallback (Tesseract, Russian) so scanned pages contribute text like any other page.

**Architecture:** New isolated module `src/RAG/ingestion/ocr.py` owns page rasterization + Tesseract invocation. `PDFContentReader.iter_chunks()` in `src/RAG/ingestion/data_loader.py` calls into it per-page-clip when native `page.get_text()` output is below a character threshold, using the OCR result in place of the empty/sparse fragment. A missing `tesseract` binary fails the whole ingestion run loudly at start; a per-page OCR failure is caught locally and that page just contributes no text (mirrors the existing tabula error-handling pattern in the same file).

**Tech Stack:** Python 3, pymupdf (`fitz`, already a dependency) for page rasterization, `pytesseract` + `Pillow` (new dependencies) as the Tesseract binding, system `tesseract-ocr` + `tesseract-ocr-rus` packages.

## Global Constraints

- OCR config, from the spec's benchmark: `dpi=300`, `--psm 3`, `lang="rus"`, no image preprocessing (grayscale/contrast/binarization tested, no measurable benefit).
- Detection is per-page-clip (not whole-document): trigger OCR when a page's native extracted fragment is under `_OCR_MIN_CHARS = 50` characters.
- Missing `tesseract` binary → `RuntimeError` raised once, immediately, at the start of `load_documents()` iteration — never silently degrade per-page for this case.
- Per-page OCR failure (rasterization or tesseract subprocess error) → caught inside `ocr_page()`, returns `""`, logged via `print(f"[data_loader] ocr error — ...")` matching the existing tabula error format at `data_loader.py:258-263`. Never aborts the whole document.
- No table-structure reconstruction on scanned pages — OCR'd page text flows through the existing plain-text path only.
- Default `pytest` run must not require the system `tesseract` binary to be installed — slow/binary-dependent tests are opt-in via a registered marker.

---

### Task 1: `ocr.py` — `ensure_tesseract_available()`

**Files:**
- Create: `src/RAG/ingestion/ocr.py`
- Test: `tests/test_ocr.py`

**Interfaces:**
- Produces: `ensure_tesseract_available() -> None` — raises `RuntimeError` if the `tesseract` binary is not found on `PATH`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ocr.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ocr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'RAG.ingestion.ocr'`

- [ ] **Step 3: Write minimal implementation**

Create `src/RAG/ingestion/ocr.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ocr.py -v`
Expected: PASS (2 passed) — if `pytesseract`/`Pillow` aren't installed yet in `.venv`, this will fail on import instead; if so, run Task 2's Step 0 first (dependency install), then return here.

- [ ] **Step 5: Commit**

```bash
git add src/RAG/ingestion/ocr.py tests/test_ocr.py
git commit -m "feat(ocr): add ensure_tesseract_available guard"
```

---

### Task 2: `ocr.py` — `ocr_page()`

**Files:**
- Modify: `src/RAG/ingestion/ocr.py`
- Modify: `requirements.txt`
- Test: `tests/test_ocr.py`

**Interfaces:**
- Consumes: nothing from Task 1 directly (independent function in the same module).
- Produces: `ocr_page(page: fitz.Page) -> str` — returns OCR'd text, or `""` on any failure (rasterization or Tesseract).

- [ ] **Step 0: Install dependencies**

Add to `requirements.txt`, in the "PDF ingestion" section (after `tabula-py`):

```
pymupdf
tabula-py
pytesseract
Pillow
langchain-text-splitters
```

Install into the project venv:

```bash
uv pip install --python .venv/bin/python pytesseract Pillow
```

Verify the system binary is present (should already be true in this environment — installed and benchmarked earlier):

```bash
which tesseract && tesseract --list-langs
```

Expected: prints a path, and `rus` appears in the language list. If `rus` is missing, install it: `sudo apt-get install -y tesseract-ocr-rus`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ocr.py`:

```python
import fitz

from RAG.ingestion.ocr import ocr_page


def _page_with_text(text: str) -> fitz.Page:
    """A single-page in-memory PDF with `text` drawn on it, for OCR round-trip testing."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=24)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ocr.py -v -k ocr_page`
Expected: FAIL with `ImportError: cannot import name 'ocr_page'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/RAG/ingestion/ocr.py` (after `ensure_tesseract_available`):

```python
def ocr_page(page: fitz.Page) -> str:
    """Rasterize `page` and run Tesseract OCR over it.

    Returns the recognized text, or "" if rasterization or OCR fails for
    any reason (caller treats "" the same as "no extractable text on this
    page" — this must never raise).
    """
    try:
        pix = page.get_pixmap(dpi=_OCR_DPI)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang=_TESSERACT_LANG, config=_TESSERACT_CONFIG)
        return text.strip()
    except Exception:
        return ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ocr.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/RAG/ingestion/ocr.py tests/test_ocr.py requirements.txt
git commit -m "feat(ocr): add ocr_page rasterize+recognize helper"
```

---

### Task 3: Wire OCR fallback into `PDFContentReader.iter_chunks()`

**Files:**
- Modify: `src/RAG/ingestion/data_loader.py:67-75` (constants block), `:190-201` (per-page text collection loop), `:319-344` (`load_documents` docstring/body)
- Test: `tests/test_data_loader_ocr.py`

**Interfaces:**
- Consumes: `ocr.ensure_tesseract_available()`, `ocr.ocr_page(page: fitz.Page) -> str` from Tasks 1-2.
- Produces: no new public symbols — behavioral change inside `PDFContentReader.iter_chunks()` and `load_documents()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_data_loader_ocr.py`:

```python
from pathlib import Path
from unittest.mock import patch

import fitz

from RAG.ingestion.data_loader import PDFContentReader


def _make_scanned_pdf(tmp_path: Path, text: str) -> Path:
    """A single-page PDF with `text` rasterized into an image (no text layer) —
    mimics a scanned document: page.get_text() returns "" for it."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=24)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_data_loader_ocr.py -v`
Expected: FAIL — `test_iter_chunks_ocrs_page_with_no_text_layer` fails because no text chunk is produced yet (native `get_text()` on the image-only page returns `""`, and there's no OCR fallback wired in yet, so `full_text` stays empty and no chunks are yielded).

- [ ] **Step 3: Add the OCR fallback**

In `src/RAG/ingestion/data_loader.py`, add the import near the top (after the `tabula` import):

```python
import fitz  # pymupdf
import tabula
from langchain_text_splitters import RecursiveCharacterTextSplitter

from RAG.ingestion import ocr
```

In the constants block (after `_BASE_ID_PATTERN`, before the closing `# ───` divider at line 75):

```python
_OCR_MIN_CHARS: int = 50  # page fragments shorter than this trigger OCR fallback
```

In the per-page text collection loop (replace lines 190-201):

```python
        # ── Collect full document text (all content pages, non-table clips) ──
        full_parts: list[str] = []
        for page_idx in range(len(doc)):
            if page_idx < first_content_page:
                continue
            page = doc[page_idx]
            bboxes = table_pages.get(page_idx, [])
            clips = _non_table_clips(page.rect, bboxes)
            for clip_rect in clips:
                fragment = page.get_text("text", clip=clip_rect).strip()
                if len(fragment) < _OCR_MIN_CHARS:
                    try:
                        ocr_text = ocr.ocr_page(page)
                    except Exception as exc:
                        print(
                            f"[data_loader] ocr error — {self.filepath.name} "
                            f"page {page_idx + 1}: {exc}"
                        )
                        ocr_text = ""
                    if ocr_text:
                        fragment = ocr_text
                if fragment:
                    full_parts.append(fragment)
```

(Note: `ocr.ocr_page()` already catches its own internal exceptions and returns `""` per Task 2 — the `try/except` here is defense in depth matching the tabula call's pattern in the same file, in case of an unexpected error at the call boundary itself.)

In `load_documents()`, call the startup guard once before the generator's main loop (right after the docstring, before `with open(manifest_path, ...)`):

```python
    ocr.ensure_tesseract_available()
    with open(manifest_path, newline="", encoding="utf-8") as fh:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_data_loader_ocr.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full existing data_loader test suite to check for regressions**

Run: `.venv/bin/python -m pytest tests/test_data_loader_only.py tests/test_data_loader_sections.py tests/test_data_loader_ocr.py tests/test_ocr.py tests/test_reingest_cli.py tests/test_reingest_planner.py -v`
Expected: all PASS. `test_data_loader_only.py`/`test_reingest_cli.py` fixtures write real tiny PDFs with actual text (not scans) via `write_bytes(b"%PDF-1.4")` stub content or real pymupdf docs — check their fragments are well above `_OCR_MIN_CHARS` so OCR never triggers for them, keeping those tests fast and independent of Tesseract. If any such test's synthetic PDF has near-empty text and starts invoking real OCR (slow, or failing in an environment without `tesseract-ocr-rus`), that test's fixture needs more filler text — fix inline before proceeding.

- [ ] **Step 6: Commit**

```bash
git add src/RAG/ingestion/data_loader.py tests/test_data_loader_ocr.py
git commit -m "feat(ocr): wire per-page OCR fallback into PDFContentReader"
```

---

### Task 4: Register the `slow` pytest marker + document the system dependency

**Files:**
- Modify: `pytest.ini`
- Modify: `CLAUDE.md`
- Test: none (config/docs only) — verified via a manual pytest invocation in Step 2

- [ ] **Step 1: Register the marker**

In `pytest.ini`, add a `markers` section:

```ini
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = session
asyncio_default_test_loop_scope = session
pythonpath = src
markers =
    slow: slow tests requiring external binaries/real files (deselect with '-m "not slow"')
```

- [ ] **Step 2: Verify the marker is recognized**

Run: `.venv/bin/python -m pytest --markers | grep slow`
Expected: `@pytest.mark.slow: slow tests requiring external binaries/real files (deselect with '-m "not slow"')`

- [ ] **Step 3: Document the system dependencies**

In `CLAUDE.md`, under the `## Environment` section, add a new subsection after the existing env-var list (find the line starting with `- \`ALENKA_ONE_C_*\`` and insert after it):

```markdown

### System dependencies (not pip-installable)

- **Java** (`java` on `PATH`) — required by `tabula-py` for table extraction from PDFs during ingestion (`scripts/ingest-pdfs.py`, `scripts/reingest-pdfs.py`). Debian/Ubuntu: `apt-get install -y default-jre-headless`. Missing Java doesn't crash ingestion — table chunks for affected pages are silently skipped and logged (`[data_loader] tabula error — ...`).
- **Tesseract OCR** (`tesseract` on `PATH`, with Russian language data) — required for the OCR fallback on scanned guideline PDFs (no text layer). Debian/Ubuntu: `apt-get install -y tesseract-ocr tesseract-ocr-rus`. Missing Tesseract fails ingestion immediately and loudly at startup (unlike Java/tabula) — see `RAG.ingestion.ocr.ensure_tesseract_available()`.
```

- [ ] **Step 4: Commit**

```bash
git add pytest.ini CLAUDE.md
git commit -m "docs: register slow test marker, document Java/Tesseract system deps"
```

---

### Task 5: Opt-in integration test against real scanned PDFs

**Files:**
- Create: `tests/test_ocr_integration.py`
- Uses (read-only, not modified): `206_2.pdf`, `1003_1.pdf` at the repo root (already present, dropped in manually for benchmarking — confirm they still exist before starting this task; if absent, skip this task and tell the user, since these are manually-provided fixtures, not something to regenerate)

**Interfaces:**
- Consumes: `PDFContentReader` from `data_loader.py` (Task 3), unmodified public interface.
- Produces: nothing new — this is a verification-only test.

- [ ] **Step 1: Confirm the fixture files are present**

```bash
ls -la 206_2.pdf 1003_1.pdf
```

Expected: both files listed, ~10MB and ~similar range respectively. If either is missing, stop this task and report to the user rather than fabricating a replacement fixture.

- [ ] **Step 2: Write the integration test**

Create `tests/test_ocr_integration.py`:

```python
"""Slow integration test against real scanned guideline PDFs (206_2.pdf,
1003_1.pdf, repo root — manually provided fixtures, not regenerated by
any script). Requires the system `tesseract` binary with Russian language
data. Run explicitly:

    pytest tests/test_ocr_integration.py -v -m slow
"""
from pathlib import Path

import pytest

from RAG.ingestion.data_loader import PDFContentReader

_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
@pytest.mark.parametrize("filename", ["206_2.pdf", "1003_1.pdf"])
def test_scanned_pdf_produces_text_chunks(filename):
    pdf_path = _ROOT / filename
    if not pdf_path.exists():
        pytest.skip(f"{filename} not present at repo root — manual fixture required")

    reader = PDFContentReader(pdf_path, metadata={"ID": "INTEGRATION_TEST"})
    chunks = list(reader.iter_chunks())
    text_chunks = [c for c in chunks if c["type"] == "text"]

    assert len(text_chunks) > 0, f"{filename}: OCR fallback produced zero text chunks"
    total_chars = sum(len(c["content"]) for c in text_chunks)
    assert total_chars > 500, f"{filename}: only {total_chars} chars extracted — OCR likely not firing"
```

- [ ] **Step 3: Run it explicitly and inspect the result**

Run: `.venv/bin/python -m pytest tests/test_ocr_integration.py -v -m slow`
Expected: PASS (2 passed), each taking roughly 30-60s (26-33 pages × ~1.8-2s/page from the spec's benchmark). Read the test output to confirm both files actually produced chunks — this is the real-world confirmation that Tasks 1-3 work end-to-end, not just against synthetic test PDFs.

- [ ] **Step 4: Confirm default test runs still skip this file by default**

Run: `.venv/bin/python -m pytest tests/ -q -m "not slow" 2>&1 | tail -15`
Expected: the full suite runs and completes without picking up `test_ocr_integration.py`'s tests (they're deselected by the marker filter). Confirms CI/default `pytest` won't be slowed down or require Tesseract.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ocr_integration.py
git commit -m "test(ocr): add opt-in integration test against real scanned guideline PDFs"
```

---

## Post-plan verification

After Task 5, run the full relevant test surface once more to confirm nothing regressed across the whole change:

```bash
.venv/bin/python -m pytest tests/test_ocr.py tests/test_data_loader_ocr.py tests/test_data_loader_only.py tests/test_data_loader_sections.py tests/test_reingest_cli.py tests/test_reingest_planner.py tests/test_resolve_pdf_path.py -v
```

Expected: all PASS. This does not include `test_ocr_integration.py` (opt-in, run separately per Task 5 Step 3) since it needs the real fixture PDFs and takes much longer.
