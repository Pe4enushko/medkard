"""
excel.py — append audit results to an xlsx workbook.

Column layout (left to right):
  A  Специализация   — Врач.SPECIALIZATION
  B  Дата приема     — Прием.DATE
  C  Данные карты    — Пациент + Врач + Прием dicts
  D  Данные осмотра  — ДанныеОсмотра list
  E  Услуги          — Услуги list
  F  Диагнозы        — Диагнозы list
  G  formal_structure
  H  diagnosis

Legacy column layout (left to right):
  A  Данные карты    — full visit dump
  B  formal_structure
  C  diagnosis

Usage::
    from parsers.excel import AuditExcelWriter
    from audit.models import DiagnosisAuditResult, FormalStructureResult

    writer = AuditExcelWriter("results.xlsx")
    writer.append(
        visit=raw_visit_dict,
        formal=FormalStructureResult(...),
        diagnosis=DiagnosisAuditResult(...),
    )
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import logging
from pathlib import Path
from typing import Any

import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Alignment
from openpyxl.worksheet.worksheet import Worksheet

from audit.models import FormalStructureResult

_HEADERS = [
    "Специализация",
    "Дата приема",
    "Данные карты",
    "Данные осмотра",
    "Услуги",
    "Диагнозы",
    "formal_structure",
    "diagnosis",
]
_COLUMN_WIDTHS = {
    "A": 25,
    "B": 15,
    "C": 60,
    "D": 80,
    "E": 60,
    "F": 60,
    "G": 80,
    "H": 100,
}
_WRAPPED_COLUMNS = ("C", "D", "E", "F", "G", "H")
_AUTOFILTER_RANGE = "A1:B1"

_LEGACY_HEADERS = ["visits", "formal", "diagnosis"]
_LEGACY_COLUMN_WIDTHS = {"A": 100, "B": 80, "C": 100}
_LEGACY_WRAPPED_COLUMNS = ("A", "B", "C")
_LEGACY_AUTOFILTER_RANGE = "A1:A1"

logger = logging.getLogger(__name__)


def _pretty(obj: Any) -> str:
    if hasattr(obj, "pretty_format") and callable(obj.pretty_format):
        return obj.pretty_format()

    if isinstance(obj, list) and all(
        hasattr(item, "pretty_format") and callable(item.pretty_format)
        for item in obj
    ):
        if not obj:
            return "—"
        return "\n\n".join(
            f"{idx}.\n{item.pretty_format()}"
            for idx, item in enumerate(obj, start=1)
        )

    if is_dataclass(obj):
        obj = asdict(obj)

    return _format_value(obj)


def _format_value(value: Any, indent: int = 0) -> str:
    prefix = " " * indent

    if is_dataclass(value):
        value = asdict(value)

    if isinstance(value, dict):
        if not value:
            return f"{prefix}—"

        lines: list[str] = []
        for key, item in value.items():
            label = str(key)
            if _is_scalar(item):
                lines.append(f"{prefix}{label}: {_format_scalar(item)}")
            else:
                lines.append(f"{prefix}{label}:")
                lines.append(_format_value(item, indent + 2))
        return "\n".join(lines)

    if isinstance(value, list):
        if not value:
            return f"{prefix}—"

        lines: list[str] = []
        for idx, item in enumerate(value, start=1):
            if _is_scalar(item):
                lines.append(f"{prefix}{idx}. {_format_scalar(item)}")
            else:
                lines.append(f"{prefix}{idx}.")
                lines.append(_format_value(item, indent + 2))
        return "\n".join(lines)

    return f"{prefix}{_format_scalar(value)}"


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _format_scalar(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    return str(value)


def _card_data_text(visit: dict[str, Any]) -> str:
    """Render Пациент + Врач + Прием as a single text block."""
    parts: list[str] = []
    for key in ("Пациент", "Врач", "Прием"):
        val = visit.get(key)
        if val is not None:
            parts.append(f"{key}:\n{_format_value(val, indent=2)}")
    return "\n\n".join(parts) if parts else "—"


class AuditExcelWriter:
    """Append audit results to an xlsx file, creating it with a header row if absent.

    Args:
        path:   Path to the xlsx output file (created automatically if missing).
        legacy: Use the legacy 3-column layout (visits / formal / diagnosis).
    """

    def __init__(self, path: str | Path, *, legacy: bool = False) -> None:
        self._path = Path(path)
        self._legacy = legacy

    def _open_or_create(self) -> tuple[Workbook, Worksheet]:
        if self._path.exists():
            wb = openpyxl.load_workbook(self._path)
            ws = wb.active
        else:
            wb = Workbook()
            ws = wb.active
        if self._legacy:
            headers = _LEGACY_HEADERS
            widths = _LEGACY_COLUMN_WIDTHS
            wrapped = _LEGACY_WRAPPED_COLUMNS
            autofilter = _LEGACY_AUTOFILTER_RANGE
        else:
            headers = _HEADERS
            widths = _COLUMN_WIDTHS
            wrapped = _WRAPPED_COLUMNS
            autofilter = _AUTOFILTER_RANGE
        for idx, header in enumerate(headers, start=1):
            ws.cell(row=1, column=idx, value=header)
        for column, width in widths.items():
            ws.column_dimensions[column].width = width
        for column in wrapped:
            for cell in ws[column]:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.auto_filter.ref = autofilter
        return wb, ws  # type: ignore[return-value]

    def append(
        self,
        visit: dict[str, Any],
        formal: FormalStructureResult,
        diagnosis: Any,
    ) -> None:
        """Append one result row and save the workbook.

        Args:
            visit:     Raw visit dict (source JSON from 1C).
            formal:    Formal-structure audit result.
            diagnosis: Diagnosis audit result(s) for this visit.
        """
        try:
            if self._legacy:
                row = [
                    _pretty(visit),
                    _pretty(formal),
                    _pretty(diagnosis),
                ]
                wrapped = _LEGACY_WRAPPED_COLUMNS
            else:
                specialization = (visit.get("Врач") or {}).get("SPECIALIZATION") or "—"
                visit_date = (visit.get("Прием") or {}).get("DATE") or "—"
                row = [
                    specialization,
                    visit_date,
                    _card_data_text(visit),
                    _pretty(visit.get("ДанныеОсмотра") or []),
                    _pretty(visit.get("Услуги") or []),
                    _pretty(visit.get("Диагнозы") or []),
                    _pretty(formal),
                    _pretty(diagnosis),
                ]
                wrapped = _WRAPPED_COLUMNS
            wb, ws = self._open_or_create()
            ws.append(row)
            new_row = ws.max_row
            for col_letter in wrapped:
                col_idx = openpyxl.utils.column_index_from_string(col_letter)
                ws.cell(row=new_row, column=col_idx).alignment = Alignment(wrap_text=True, vertical="top")
            ws.row_dimensions[new_row].height = None
            wb.save(self._path)
            logger.info("📊 EXCEL APPEND OK path=%s", self._path)
        except Exception:
            logger.exception("📊 EXCEL APPEND FAILED path=%s", self._path)
            raise

    def rows_count(self) -> int:
        """Return current number of rows in the active worksheet (0 if file absent)."""
        if not self._path.exists():
            return 0
        wb = openpyxl.load_workbook(self._path)
        try:
            return wb.active.max_row
        finally:
            wb.close()
