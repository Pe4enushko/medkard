"""
Data loader for ambulatory card PDFs.

Reads files listed in manifest.csv from the pdfs/ directory and yields
PDFContentReader instances. Each reader lazily iterates over text chunks
(with table regions excluded via `clip`) and table chunks (parsed with tabula,
split into row batches).

Chunk shape:
    {
        "type": "text" | "table",
        "content": str | list[dict],   # str for text, list of row dicts for table
        "metadata": {
            # chunk-intrinsic only — manifest columns (ID, Наименование, МКБ-10, ...)
            # live in the `guidelines` table, joined by docs.file_id at read time.
            "content_type": "text" | "table",
            # text-only:
            "section": str | None,     # numbered section title (e.g. "1.1 Title"),
                                       # extracted by regex from the document text;
                                       # None when no numbered sections are found
            "chunk_index": int,        # ordinal of this chunk across the whole document
            # table-only:
            "table_index": int,        # ordinal of the table on the page
            "chunk_index": int,        # ordinal of this row-batch within the table
        },
    }
"""

import csv
import json
import re
from pathlib import Path
from typing import Generator, Iterator

import fitz  # pymupdf
import tabula
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Configurable ─────────────────────────────────────────────────────────────
TABLE_ROW_CHUNK_SIZE: int = 8   # max rows per table chunk yielded to the pipeline
TEXT_CHUNK_SIZE: int = 2000     # characters per text chunk
TEXT_CHUNK_OVERLAP: int = 200   # character overlap between consecutive text chunks

# Numbered section titles such as "1.1 Title" or "1.1.1 Title".  Dot leaders
# are rejected so printed-ToC entries such as "3.1 Диагностика....12" do not
# become document boundaries.
_NUMBERED_SECTION_PREFIX = r'\d+\.\d+(?:\.\d+)?'
_SECTION_TITLE_TEXT = (
    r'(?=[^\n]*[A-Za-zА-Яа-я])'
    r'(?![^\n]*\.{2,})'
    r'[^\n]+'
)
_NUMBERED_SECTION_TITLE = (
    _NUMBERED_SECTION_PREFIX
    + r'(?:'
    + r'[^\S\r\n]+'
    + _SECTION_TITLE_TEXT
    + r'|[^\S\r\n]*(?:\r?\n[^\S\r\n]*)+(?!'
    + _NUMBERED_SECTION_PREFIX
    + r'(?:[ \t]|\r?(?:\n|\Z)))'
    + _SECTION_TITLE_TEXT
    + r')'
)

# Clinical guidelines commonly end their numbered body with an unnumbered
# bibliography.  Without treating it as a boundary, the last numbered section
# absorbs the bibliography and every appendix that follows it.  Keep this an
# explicit allow-list: arbitrary unnumbered lines are too ambiguous to promote
# to section titles reliably.
# TODO(clinical-pdf-tail): a bibliography boundary labels the tail but does not
# classify or discard references, OCR debris, and unrecognized appendices after
# it.  Keep the remaining corpus debt measured in
# evals/clinical_pdf_sections/README.md.
_BIBLIOGRAPHY_SECTION_TITLE = (
    r'(?:(?:[IVXLCDM]+|\d+)\.[ \t]*)?'
    r'(?:'
    r'Список[ \t]+литературы'
    r'|Список[ \t]+использованной[ \t]+литературы'
    r'|Список[ \t]+использованных[ \t]+источников'
    r'|Библиографический[ \t]+список'
    r'|Литература'
    r')[.:]?'
)

# Quality-criteria sections are semantically stable but structurally varied:
# depending on the guideline template they can be unnumbered, use a one-level
# Arabic/Roman number, or be introduced as a numbered table.  Matching the
# stable phrase is safer than enabling arbitrary one-level headings, which
# would turn numbered recommendations and bibliography entries into sections.
_CRITERIA_SECTION_TITLE = (
    r'(?:(?:'
    r'(?:[IVXLCDM]+|\d+)[.-]'
    r'|Таблица[ \t]+\d+(?:\.\d+)?[ \t]*(?:[.\-–—:][ \t]*)?'
    r')[ \t]+)?'
    r'Критерии[ \t]+оценки[ \t]+качества'
    r'(?![^\n]*\.{2,})'
    r'(?![^\n]*(?:\n[^\n]*){0,3}\.{2,})'
    r'(?![^\n]*(?:приведен\w*|представлен\w*|указан\w*|приказ\w*))'
    r'[^\n]*'
)

# Appendices follow the quality-criteria table in many guideline templates.
# Only descriptive appendix headings are boundaries.  A bare cross-reference
# such as "Приложение А3/5." is deliberately left as body text.
_APPENDIX_CODE = r'[A-Za-zА-Яа-я0-9]+(?:[/][A-Za-zА-Яа-я0-9]+)*'
_APPENDIX_SECTION_TITLE = (
    r'Приложени[ея][ \t]+'
    + _APPENDIX_CODE
    + r'(?:[ \t]*-[ \t]*'
    + _APPENDIX_CODE
    + r')?'
    + r'[.]?[ \t]+'
    + r'(?=[^\n]*[A-Za-zА-Яа-я])'
    + r'(?![^\n]*\.{2,})'
    + r'[^\n]+'
)

_SECTION_TITLE_PATTERN: re.Pattern = re.compile(
    r'^('
    + _NUMBERED_SECTION_TITLE
    + r'|[ \t]*(?:'
    + _BIBLIOGRAPHY_SECTION_TITLE
    + r'|'
    + _CRITERIA_SECTION_TITLE
    + r'|'
    + _APPENDIX_SECTION_TITLE
    + r')'
    + r')[ \t]*$',
    re.MULTILINE | re.IGNORECASE,
)
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
PDFS_DIR: Path = _PROJECT_ROOT / "pdfs"
MANIFEST_PATH: Path = _PROJECT_ROOT / "resources" / "manifest.csv"
PDF_EXTENSION: str = ".pdf"
PDF_PREFIX: str = "КР"  # new-convention filename prefix, e.g. "КР1027.pdf"
# Manifest IDs are "{base}_{revision}" (e.g. "1027_1"); the revision changes
# across updates but the base is stable, so it's what filenames key on.
_BASE_ID_PATTERN: re.Pattern = re.compile(r'^\d+')
# ─────────────────────────────────────────────────────────────────────────────

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TEXT_CHUNK_SIZE,
    chunk_overlap=TEXT_CHUNK_OVERLAP,
)


def _section_for_page(sorted_toc: list[tuple[str, int]], page_idx: int) -> str | None:
    """Return the TOC section title whose start page is <= page_idx.

    sorted_toc is a list of (title, 0-based page index) sorted ascending by page.
    """
    section = None
    for title, start in sorted_toc:
        if page_idx >= start:
            section = title
        else:
            break
    return section


def _non_table_clips(page_rect: fitz.Rect, table_bboxes: list[tuple]) -> list[fitz.Rect]:
    """Return horizontal-slice Rect list covering page_rect minus table areas.

    Table bboxes (x0, y0, x1, y1) are sorted by top edge and subtracted from the
    full page height to produce clip regions passed to page.get_text(clip=...).
    """
    if not table_bboxes:
        return [page_rect]

    sorted_bboxes = sorted(table_bboxes, key=lambda b: b[1])  # by y0 (top)

    clips: list[fitz.Rect] = []
    cursor_y = page_rect.y0

    for bbox in sorted_bboxes:
        top, bottom = bbox[1], bbox[3]
        if top > cursor_y:
            clips.append(fitz.Rect(page_rect.x0, cursor_y, page_rect.x1, top))
        cursor_y = max(cursor_y, bottom)

    if cursor_y < page_rect.y1:
        clips.append(fitz.Rect(page_rect.x0, cursor_y, page_rect.x1, page_rect.y1))

    return clips


def _split_rows(rows: list[dict], chunk_size: int) -> list[list[dict]]:
    """Partition rows into consecutive batches of at most chunk_size."""
    return [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]


def _split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """Split *text* into numbered sections and explicit terminal sections.

    Returns a list of (section_title, section_text) pairs.  If no numbered
    sections are found the whole text is returned as a single entry with
    title ``None``.
    """
    matches = list(_SECTION_TITLE_PATTERN.finditer(text))
    first_numbered = next(
        (
            index
            for index, match in enumerate(matches)
            if re.match(r'\d+\.\d+(?:\.\d+)?\s+', match.group(1))
        ),
        None,
    )
    if first_numbered is None:
        return [(None, text)]

    # Preserve the historical behaviour of dropping front matter before the
    # first numbered content section.  An isolated semantic heading in an
    # otherwise unnumbered document must not discard everything before it.
    matches = matches[first_numbered:]
    result: list[tuple[str | None, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[match.start() : end].strip()
        if not section_text:
            continue
        title = match.group(1).strip()
        result.append((title, section_text))
    return result


class PDFContentReader:
    """Lazy reader for a single PDF document.

    Usage::
        for chunk in reader.iter_chunks():
            process(chunk)
    """

    def __init__(self, filepath: Path, metadata: dict) -> None:
        self.filepath = filepath
        self.metadata = metadata  # row from manifest, includes "ID" and all columns

    # ------------------------------------------------------------------
    def iter_chunks(self) -> Iterator[dict]:
        doc = fitz.open(self.filepath)

        # ── TOC → section map ─────────────────────────────────────────
        # get_toc() returns [[level, title, page_1based], ...]
        raw_toc = doc.get_toc()
        sorted_toc: list[tuple[str, int]] = []
        if raw_toc:
            sorted_toc = sorted(
                [(title, page - 1) for _, title, page in raw_toc],
                key=lambda x: x[1],
            )

        # Pages strictly before the first content section (title page, printed TOC,
        # etc.) are skipped entirely — they are not useful for retrieval.
        first_content_page: int = sorted_toc[0][1] if sorted_toc else 0

        # ── Detect table bounding boxes per page ──────────────────────
        # Stored as {page_idx: [(x0, y0, x1, y1), ...]}
        table_pages: dict[int, list[tuple]] = {}
        for page_idx in range(len(doc)):
            if page_idx < first_content_page:
                continue
            found = doc[page_idx].find_tables()
            if found.tables:
                table_pages[page_idx] = [t.bbox for t in found.tables]

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
                if fragment:
                    full_parts.append(fragment)

        full_text = "\n".join(full_parts).strip()

        # ── Strip TOC / title pages ───────────────────────────────────
        # PDFs typically contain "Список сокращений" twice: once in the
        # printed table of contents and once as the actual section heading.
        # Everything up to (and including) the second match is front matter
        # and should be discarded.
        _ABBR = "Список сокращений"
        first = full_text.find(_ABBR)
        if first != -1:
            second = full_text.find(_ABBR, first + len(_ABBR))
            if second != -1:
                full_text = full_text[second:].strip()

        # ── Text chunks — split by numbered sections, then chunk within each ──
        if full_text:
            sections = _split_into_sections(full_text)
            chunk_counter = 0
            for section_title, section_text in sections:
                sub_chunks = _text_splitter.split_text(section_text)
                for sub_chunk in sub_chunks:
                    yield {
                        "type": "text",
                        "content": sub_chunk,
                        "metadata": {
                            "section": section_title,
                            "content_type": "text",
                            "chunk_index": chunk_counter,
                        },
                    }
                    chunk_counter += 1

        # ── Table chunks ──────────────────────────────────────────────
        for page_idx, bboxes in table_pages.items():
            section = _section_for_page(sorted_toc, page_idx)
            base_meta = {
                "page": page_idx,
                "section": section,
                "content_type": "table",
            }

            for table_idx, bbox in enumerate(bboxes):
                # tabula area: [top, left, bottom, right] from top-left in points —
                # same coordinate origin as pymupdf bbox (y=0 at page top).
                area = [bbox[1], bbox[0], bbox[3], bbox[2]]

                try:
                    dfs = tabula.read_pdf(
                        str(self.filepath),
                        pages=page_idx + 1,          # tabula uses 1-based pages
                        area=area,
                        multiple_tables=True,
                        pandas_options={"dtype": str},
                        silent=True,
                    )
                except Exception as exc:
                    print(
                        f"[data_loader] tabula error — {self.filepath.name} "
                        f"page {page_idx + 1} table {table_idx}: {exc}"
                    )
                    continue

                if not dfs or dfs[0].empty:
                    continue

                df = dfs[0].fillna("")
                rows: list[dict] = df.to_dict(orient="records")

                for chunk_idx, row_batch in enumerate(_split_rows(rows, TABLE_ROW_CHUNK_SIZE)):
                    yield {
                        "type": "table",
                        "content": row_batch,  # list of row dicts; headers are dict keys
                        "metadata": {
                            **base_meta,
                            "table_index": table_idx,
                            "chunk_index": chunk_idx,
                        },
                    }

        doc.close()


# ── Public API ────────────────────────────────────────────────────────────────

def base_id(file_id: str) -> str | None:
    """Return the stable numeric base of a manifest file_id (e.g. "1027" for
    "1027_1"), or None if file_id doesn't start with digits. Two file_ids
    sharing a base are different revisions of the same guideline."""
    match = _BASE_ID_PATTERN.match(file_id)
    return match.group() if match else None


def resolve_pdf_path(file_id: str, pdfs_dir: Path = PDFS_DIR) -> Path | None:
    """Resolve a manifest file_id (e.g. "1027_1") to its PDF on disk.

    Tries the new naming convention first — "{PREFIX}{base}.pdf" where base is
    the numeric part of file_id before the "_{revision}" suffix (revisions
    change across manifest updates, the base doesn't) — then falls back to the
    old convention of the raw file_id as the filename stem, for files not yet
    migrated to the new convention.

    Returns None if neither candidate exists.
    """
    base = base_id(file_id)
    if base is not None:
        primary = pdfs_dir / f"{PDF_PREFIX}{base}{PDF_EXTENSION}"
        if primary.exists():
            return primary

    fallback = pdfs_dir / (file_id + PDF_EXTENSION)
    if fallback.exists():
        return fallback

    return None


def load_documents(
    manifest_path: Path = MANIFEST_PATH,
    pdfs_dir: Path = PDFS_DIR,
    exceptions: set[str] | None = None,
    only: set[str] | None = None,
) -> Generator[PDFContentReader, None, None]:
    """Generator over all documents listed in manifest.csv.

    Yields a PDFContentReader for each row whose file exists in pdfs_dir.
    The file is located via resolve_pdf_path() (new "{PREFIX}{base}.pdf"
    convention first, old "{ID}.pdf" convention as fallback).

    Args:
        manifest_path: Path to the CSV manifest.
        pdfs_dir:      Directory containing PDFs.
        exceptions:    Optional set of ID strings to skip (e.g. already ingested).
        only:          Optional set of ID strings to yield exclusively (everything
                       else is skipped). Applied together with `exceptions`
                       (exceptions win).

    Example::
        for reader in load_documents():
            for chunk in reader.iter_chunks():
                ingest(chunk)
    """
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            file_id = row["ID"]
            if only is not None and file_id not in only:
                continue
            if exceptions is not None and file_id in exceptions:
                continue
            filepath = resolve_pdf_path(file_id, pdfs_dir)
            if filepath is None:
                print(f"[data_loader] missing file, skipping: {file_id}")
                continue
            yield PDFContentReader(filepath, dict(row))
