"""Read GRLS xlsx exports (one status sheet per file) into GrlsRecord objects."""
from __future__ import annotations

import logging
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import openpyxl

from grls.normalize import (clean_cell, derive_dispensing, derive_dosage_forms,
                            is_substance, parse_date, parse_narcotic, parse_yes_no,
                            row_hash, split_forms)
from grls.status import ALL_STATUSES, STATUS_CHANGED
from storage.models.grls_record import GrlsRecord

logger = logging.getLogger(__name__)

TITLE_ROW, HEADER_ROW, MARKER_ROW = 3, 5, 6
FIRST_COL = 2          # zero-based index of column C
N_COLS = 15            # C..Q
_TITLE_COL = 3         # column D
_REGISTRY_DATE_RE = re.compile(r"по состоянию на\s+(\d{2}\.\d{2}\.\d{4})")
# Header prefixes (whitespace-collapsed); None = column must have no header (H = country).
EXPECTED_HEADER_PREFIXES: tuple[str | None, ...] = (
    "Номер регистрационного удостоверения", "Дата регистрации", "Дата окончания действия",
    "Дата аннулирования", "Юридическое лицо", None, "Торговое наименование",
    "Международное непатентованное", "Формы выпуска", "Сведения о стадиях производства",
    "Нормативная документация", "Фармако-терапевтическая группа",
    "Наличие лекарственного препарата в перечне ЖНВЛП",
    "Наличие в лекарственном препарате наркотических", "Орфанный",
)


class GrlsFormatError(ValueError):
    """The xlsx does not look like a GRLS export (layout changed?)."""


@dataclass
class SheetResult:
    path: Path
    source_name: str
    status: str
    registry_date: date
    records: list[GrlsRecord]
    skipped: bool = False


def _slice(row: Sequence[object] | None) -> tuple:
    row = tuple(row or ()) + (None,) * (FIRST_COL + N_COLS)
    return row[FIRST_COL:FIRST_COL + N_COLS]


def _norm_header(value: object) -> str | None:
    text = " ".join(str(value).split()) if value is not None else ""
    return text or None


def _check_headers(cells: Sequence[object], name: str) -> None:
    for i, (expected, got) in enumerate(zip(EXPECTED_HEADER_PREFIXES, cells)):
        actual = _norm_header(got)
        if expected is None:
            if actual is not None:
                raise GrlsFormatError(f"{name}: column {i} expected empty header, got {actual!r}")
        elif actual is None or not actual.startswith(expected):
            raise GrlsFormatError(f"{name}: column {i} header {actual!r} does not start with {expected!r}")


def build_record(status: str, cells: Sequence[object]) -> GrlsRecord | None:
    """15 cells (C..Q) → GrlsRecord; None for blank/trailer/nameless rows."""
    (reg_number, registered_at, expires_at, annulled_at, holder, holder_country,
     trade_name, inn_name, forms_raw, production_stages, normative_docs, pharm_group,
     vital, narcotic, orphan) = (clean_cell(c) for c in cells)
    if reg_number is None:
        return None
    others = (registered_at, expires_at, annulled_at, holder, holder_country, trade_name,
              inn_name, forms_raw, production_stages, normative_docs, pharm_group)
    if all(v is None for v in others):
        return None  # trailer row (export date) or junk
    if trade_name is None:
        logger.warning("GRLS: row %s without trade name skipped", reg_number)
        return None
    reg_d, exp_d, ann_d = parse_date(registered_at), parse_date(expires_at), parse_date(annulled_at)
    is_vital, is_orphan, narcotic_list = parse_yes_no(vital), parse_yes_no(orphan), parse_narcotic(narcotic)
    forms = split_forms(forms_raw)
    dosage_forms = derive_dosage_forms(forms)
    return GrlsRecord(
        status=status, reg_number=reg_number, trade_name=trade_name,
        registered_at=reg_d, expires_at=exp_d, annulled_at=ann_d,
        holder=holder, holder_country=holder_country, inn_name=inn_name,
        forms=forms, forms_raw=forms_raw, dosage_forms=dosage_forms,
        dispensing=derive_dispensing(forms),
        is_substance=is_substance(reg_number, dosage_forms),
        production_stages=production_stages, normative_docs=normative_docs,
        pharm_group=pharm_group, is_vital=is_vital, narcotic_list=narcotic_list,
        is_orphan=is_orphan,
        row_hash=row_hash(
            status=status, reg_number=reg_number, registered_at=reg_d, expires_at=exp_d,
            annulled_at=ann_d, holder=holder, holder_country=holder_country,
            trade_name=trade_name, inn_name=inn_name, forms_raw=forms_raw,
            production_stages=production_stages, normative_docs=normative_docs,
            pharm_group=pharm_group, is_vital=is_vital, narcotic_list=narcotic_list,
            is_orphan=is_orphan),
    )


def read_sheet(path: Path, source_name: str | None = None) -> SheetResult:
    name = source_name or path.name
    wb = openpyxl.load_workbook(path, read_only=True)
    try:
        ws = wb.worksheets[0]
        rows = ws.iter_rows(min_row=1, values_only=True)
        head = [tuple(next(rows, ())) for _ in range(MARKER_ROW)]
        title_cells = head[TITLE_ROW - 1] + (None,) * (_TITLE_COL + 1)
        m = _REGISTRY_DATE_RE.search(str(title_cells[_TITLE_COL] or ""))
        if not m:
            raise GrlsFormatError(f"{name}: registry date not found in row {TITLE_ROW}")
        registry_date = parse_date(m.group(1))
        assert registry_date is not None
        _check_headers(_slice(head[HEADER_ROW - 1]), name)
        status = clean_cell(_slice(head[MARKER_ROW - 1])[0])
        if status == STATUS_CHANGED:
            logger.info("GRLS: %s is the revision journal (%s) — skipped", name, status)
            return SheetResult(path, name, status, registry_date, [], skipped=True)
        if status not in ALL_STATUSES:
            raise GrlsFormatError(f"{name}: unknown status marker {status!r}")
        records = [r for r in (build_record(status, _slice(row)) for row in rows) if r is not None]
        return SheetResult(path, name, status, registry_date, records)
    finally:
        wb.close()


def _decode_zip_name(raw: str) -> str:
    try:
        return raw.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return raw


def read_archive(path: Path) -> list[SheetResult]:
    """zip or directory with *.xlsx → one SheetResult per sheet (sorted by source name)."""
    path = Path(path)
    if path.is_dir():
        return [read_sheet(p) for p in sorted(path.glob("*.xlsx"))]
    results: list[SheetResult] = []
    with zipfile.ZipFile(path) as z, tempfile.TemporaryDirectory() as tmp:
        members = [i for i in z.infolist() if i.filename.lower().endswith(".xlsx")]
        for idx, info in enumerate(sorted(members, key=lambda i: i.filename)):
            # Names in the export are cp437-garbled and too long for the FS — use our own.
            target = Path(tmp) / f"sheet_{idx}.xlsx"
            target.write_bytes(z.read(info))
            results.append(read_sheet(target, source_name=_decode_zip_name(info.filename)))
    return results
