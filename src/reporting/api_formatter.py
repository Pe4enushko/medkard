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
from reporting.result_parser import build_manifest_meta, parse_diagnosis, parse_formal, parse_icd_check
from storage.base import BaseStorage
from storage.guidelines_storage import GuidelinesStorage

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

    async def fetch_export(
        self, organization_id: str, since: str | None, limit: int, cursor: int
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT card_guid, card_data, formal_result, diag_result, "
            "       icd_check_result, updated_at "
            "FROM done_cards "
            "WHERE organization_id = %(org_id)s::uuid "
            "  AND ignored = FALSE AND broken = FALSE "        # audited cards only
            "  AND (%(since)s::timestamptz IS NULL OR updated_at > %(since)s::timestamptz) "
            "ORDER BY updated_at, card_guid "
        )
        params: dict[str, Any] = {"org_id": organization_id, "since": since}
        if limit and limit > 0:
            query += "LIMIT %(limit)s OFFSET %(cursor)s"
            params["limit"] = limit
            params["cursor"] = cursor

        async with self._pool.connection() as conn:
            cur = await conn.execute(query, params)
            return await cur.fetchall()

    async def fetch_changed(self, organization_id: str, since: str | None) -> list[dict[str, Any]]:
        """Rows changed since a client-supplied timestamp, in every status.

        Deliberately kept separate from fetch_export rather than adding a flag to
        it: this one omits the ignored/broken filter (the raw card_data of an
        unaudited card is the whole point) and the boundary is inclusive, since
        the caller derives `since` from a clock rather than from returned rows.

        status/ignored/broken are collapsed into a single status on the way out.
        They already describe one thing between them — migration 014 forbids
        ignored and broken overlapping, and 025's status treats 'done' as
        "audited, ignored or broken" — so the wire format states the outcome
        once instead of making every consumer re-derive it from three fields.
        Storage keeps all three; this is a response shape, not a schema change.
        """
        query = (
            "SELECT card_guid, card_data, "
            "       CASE WHEN broken THEN 'broken' "
            "            WHEN ignored THEN 'ignored' "
            "            ELSE status END AS status, "
            "       formal_result, diag_result, icd_check_result, updated_at "
            "FROM done_cards "
            "WHERE organization_id = %(org_id)s::uuid "
            "  AND updated_at >= COALESCE(%(since)s::timestamptz, now() - interval '7 days') "
            "ORDER BY updated_at, card_guid"
        )
        async with self._pool.connection() as conn:
            cur = await conn.execute(query, {"org_id": organization_id, "since": since})
            return await cur.fetchall()


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
        async with GuidelinesStorage() as _store:
            manifest_meta = build_manifest_meta(await _store.all())

        workbook_rows = []
        for row in rows:
            formal = parse_formal(row["formal_result"])
            diagnosis = parse_diagnosis(row["diag_result"], manifest_meta)
            icd_check = parse_icd_check(row.get("icd_check_result") or [])
            workbook_rows.append((row["card_data"], formal, diagnosis, icd_check))

        return build_workbook_bytes(workbook_rows)

    async def export(
        self, organization_id: str, since: str | None, limit: int, cursor: int
    ) -> list[dict[str, Any]]:
        """Return done_cards rows for one org as native dicts.

        since=None → all history; limit=0 → no LIMIT/OFFSET (one-shot daily).
        limit>0 uses cursor as an OFFSET for the backfill loop.
        """
        return await self._reader.fetch_export(organization_id, since, limit, cursor)

    async def check_updates(self, organization_id: str, since: str | None) -> list[dict[str, Any]]:
        """Return cards changed at or after `since`, in every status.

        since=None → the last week, not all history: a bare call shouldn't drain
        the table. Unlike export, includes pending/ignored/broken cards — the
        consumer needs a card's raw data whether or not it has been audited.
        Each row's outcome arrives as a single `status`:
        pending | done | ignored | broken.
        """
        return await self._reader.fetch_changed(organization_id, since)
