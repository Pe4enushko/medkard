"""
reporting/api_formatter.py — ApiFormatter: done_cards rows -> pull API responses.

Analogue to audit/excel_formatter.py's ExcelFormatter, but for the pull API
instead of a report on disk: owns the org-scoped, date-scoped done_cards
queries and builds an in-memory xlsx workbook (via parsers/excel.py) for
`pull`, or a row count for `check` — the integrating service compares that
count against how many rows it ingested from the last report and re-pulls
on a mismatch. Kept out of src/api/ so route handlers stay limited to
request parsing and auth.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from parsers.excel import build_workbook_bytes
from reporting.result_parser import load_manifest_meta, parse_diagnosis, parse_formal, parse_icd_check
from storage.base import BaseStorage

_VISIT_DATE_CTE = (
    "WITH cards AS ("
    "  SELECT id, card_guid, card_data, formal_result, diag_result, icd_check_result, "
    "         to_date(card_data -> 'Прием' ->> 'DATE', 'DD.MM.YYYY') AS visit_date "
    "  FROM done_cards "
    "  WHERE ignored = FALSE "
    "    AND broken = FALSE "
    "    AND organization_id = %(org_id)s::uuid"
    ") "
)


class _ApiCardsReader(BaseStorage):
    @staticmethod
    def _decode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for row in rows:
            if isinstance(row["card_data"], str):
                row["card_data"] = json.loads(row["card_data"])
        return rows

    async def count_by_date(self, visit_date: date, organization_id: str) -> int:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                _VISIT_DATE_CTE + "SELECT count(*) AS n FROM cards WHERE visit_date = %(date)s::date",
                {"org_id": organization_id, "date": visit_date},
            )
            row = await cur.fetchone()
        return row["n"]

    async def fetch_by_date(self, visit_date: date, organization_id: str) -> list[dict[str, Any]]:
        query = (
            _VISIT_DATE_CTE
            + "SELECT card_guid, card_data::text, formal_result, diag_result, icd_check_result "
            "FROM cards WHERE visit_date = %(date)s::date "
            "ORDER BY card_guid"
        )
        params: dict[str, Any] = {"org_id": organization_id, "date": visit_date}

        async with self._pool.connection() as conn:
            cur = await conn.execute(query, params)
            return self._decode_rows(await cur.fetchall())


class ApiFormatter:
    """Async context-manager producing pull-API responses for one organization."""

    def __init__(self) -> None:
        self._reader = _ApiCardsReader()

    async def __aenter__(self) -> "ApiFormatter":
        await self._reader.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._reader.__aexit__(*args)

    async def check(self, visit_date: date, organization_id: str) -> int:
        """Return the number of audited cards for *organization_id* on *visit_date*."""
        return await self._reader.count_by_date(visit_date, organization_id)

    async def make_xlsx(self, visit_date: date, organization_id: str) -> bytes:
        """Return an in-memory xlsx workbook (bytes), one row per card.

        Same row layout as the Excel reports produced by
        audit/excel_formatter.py — built in memory, no disk I/O.
        """
        rows = await self._reader.fetch_by_date(visit_date, organization_id)
        manifest_meta = load_manifest_meta()

        workbook_rows = []
        for row in rows:
            formal = parse_formal(row["formal_result"])
            diagnosis = parse_diagnosis(row["diag_result"], manifest_meta)
            icd_check = parse_icd_check(row.get("icd_check_result") or [])
            workbook_rows.append((row["card_data"], formal, diagnosis, icd_check))

        return build_workbook_bytes(workbook_rows)
