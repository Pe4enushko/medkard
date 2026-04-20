"""
excel.py — append audit results to an xlsx workbook.

Each row contains three columns with pretty-printed JSON:
  - ``input``            — raw visit payload (source JSON from 1C)
  - ``formal_structure`` — FormalStructureResult as JSON
  - ``diagnosis``        — DiagnosisAuditResult as JSON

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

import json
from pathlib import Path
from typing import Any

import openpyxl
from openpyxl import Workbook
from openpyxl.worksheet.worksheet import Worksheet

from audit.models import DiagnosisAuditResult, FormalStructureResult

_HEADERS = ["input", "formal_structure", "diagnosis"]


def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


class AuditExcelWriter:
    """Append audit results to an xlsx file, creating it with a header row if absent.

    Args:
        path: Path to the xlsx output file (created automatically if missing).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def _open_or_create(self) -> tuple[Workbook, Worksheet]:
        if self._path.exists():
            wb = openpyxl.load_workbook(self._path)
            ws = wb.active
        else:
            wb = Workbook()
            ws = wb.active
            ws.append(_HEADERS)
        return wb, ws  # type: ignore[return-value]

    def append(
        self,
        visit: dict[str, Any],
        formal: FormalStructureResult,
        diagnosis: DiagnosisAuditResult,
    ) -> None:
        """Append one result row and save the workbook.

        Args:
            visit:     Raw visit dict (source JSON from 1C).
            formal:    Formal-structure audit result.
            diagnosis: Diagnosis audit result (all three checker agents).
        """
        row = [
            _pretty(visit),
            _pretty(formal.to_dict()),
            _pretty(diagnosis.to_dict()),
        ]
        wb, ws = self._open_or_create()
        ws.append(row)
        wb.save(self._path)
