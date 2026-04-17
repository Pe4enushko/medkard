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
            "ID": str,                 # original manifest ID / filename stem
            # ... all other manifest columns
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

# Regex that matches numbered sections like "1.1 Title ..." spanning multiple lines.
_SECTION_PATTERN: re.Pattern = re.compile(r'(?ms)^(\d+\.\d+\s+.*?)(?=^\d+\.\d+\s+|\Z)')
_SECTION_TITLE_PATTERN: re.Pattern = re.compile(r'^(\d+\.\d+\s+[^\n]+)')
PDFS_DIR: Path = Path("../pdfs")
MANIFEST_PATH: Path = Path("manifest.csv")
PDF_EXTENSION: str = ".pdf"
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
    """Split *text* into numbered sections (e.g. 1.1, 2.3) using regex.

    Returns a list of (section_title, section_text) pairs.  If no numbered
    sections are found the whole text is returned as a single entry with
    title ``None``.
    """
    matches = _SECTION_PATTERN.findall(text)
    if not matches:
        return [(None, text)]

    result: list[tuple[str | None, str]] = []
    for section_text in matches:
        section_text = section_text.strip()
        if not section_text:
            continue
        m = _SECTION_TITLE_PATTERN.match(section_text)
        title: str | None = m.group(1).strip() if m else None
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
                            **self.metadata,
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
                **self.metadata,
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

def load_documents(
    manifest_path: Path = MANIFEST_PATH,
    pdfs_dir: Path = PDFS_DIR,
    exceptions: set[str] | None = None,
) -> Generator[PDFContentReader, None, None]:
    """Generator over all documents listed in manifest.csv.

    Yields a PDFContentReader for each row whose file exists in pdfs_dir.
    The 'ID' column is used as the filename stem; PDF_EXTENSION is appended.

    Args:
        manifest_path: Path to the CSV manifest.
        pdfs_dir:      Directory containing PDFs.
        exceptions:    Optional set of ID strings to skip (e.g. already ingested).

    Example::
        for reader in load_documents():
            for chunk in reader.iter_chunks():
                ingest(chunk)
    """
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            file_id = row["ID"]
            if exceptions is not None and file_id in exceptions:
                continue
            filename = file_id + PDF_EXTENSION
            filepath = pdfs_dir / filename
            if not filepath.exists():
                print(f"[data_loader] missing file, skipping: {filepath}")
                continue
            yield PDFContentReader(filepath, dict(row))
