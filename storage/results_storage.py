"""
ResultsStorage — async psycopg3 interface for the *results* table.
"""

import json

from .base import BaseStorage
from .models import ClinicalSource, Result, Source


def _row_to_result(row: dict) -> Result:
    clinical_sources = [
        ClinicalSource(
            flag=cs["flag"],
            sources=[
                Source(
                    file=s["file"],
                    file_metadata=s["file_metadata"],
                    page=s["page"],
                    section=s.get("section"),
                )
                for s in cs.get("sources", [])
            ],
        )
        for cs in (row.get("clinical_sources") or [])
    ]
    return Result(
        id=row["id"],
        input=row["input"],
        flags=row["flags"],
        clinical_sources=clinical_sources,
    )


def _serialize_clinical_sources(clinical_sources: list[ClinicalSource]) -> str:
    return json.dumps([
        {
            "flag": cs.flag,
            "sources": [
                {
                    "file": s.file,
                    "file_metadata": s.file_metadata,
                    "page": s.page,
                    **( {"section": s.section} if s.section is not None else {}),
                }
                for s in cs.sources
            ],
        }
        for cs in clinical_sources
    ])


# ── Storage class ─────────────────────────────────────────────────────────────

class ResultsStorage(BaseStorage):
    """Async context-manager for the results table.

    Usage::
        async with ResultsStorage() as storage:
            result_id = await storage.insert(result)
            result    = await storage.get(result_id)
    """

    # ── Writes ────────────────────────────────────────────────────────────────

    async def insert(self, result: Result) -> str:
        """Insert a Result and return its UUID. Also sets result.id."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                INSERT INTO results (input, flags, clinical_sources)
                VALUES (%(input)s, %(flags)s, %(clinical_sources)s)
                RETURNING id::text
                """,
                {
                    "input": json.dumps(result.input),
                    "flags": result.flags,
                    "clinical_sources": _serialize_clinical_sources(result.clinical_sources),
                },
            )
            row = await cur.fetchone()
        result.id = row["id"]
        return row["id"]

    # ── Reads ─────────────────────────────────────────────────────────────────

    async def get(self, result_id: str) -> Result | None:
        """Fetch a single Result by UUID; returns None if not found."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id::text, input, flags, clinical_sources
                FROM results
                WHERE id = %(id)s::uuid
                """,
                {"id": result_id},
            )
            row = await cur.fetchone()
        return _row_to_result(row) if row else None

    async def get_by_flag(self, flag: str) -> list[Result]:
        """Fetch all Results that contain *flag* in their flags array."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id::text, input, flags, clinical_sources
                FROM results
                WHERE %(flag)s = ANY(flags)
                ORDER BY id
                """,
                {"flag": flag},
            )
            rows = await cur.fetchall()
        return [_row_to_result(r) for r in rows]
