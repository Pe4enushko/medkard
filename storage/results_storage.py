"""
ResultsStorage — async psycopg3 interface for the *results* table.
"""

import json

from .base import BaseStorage
from .models import Issue, IssueSource, Result


def _row_to_result(row: dict) -> Result:
    issues = [
        Issue(
            issue=item["issue"],
            sources=[
                IssueSource(
                    doc_title=s["doc_title"],
                    section=s.get("section"),
                )
                for s in item.get("sources", [])
            ],
        )
        for item in (row.get("issues") or [])
    ]
    return Result(
        id=row["id"],
        input=row["input"],
        flags=row["flags"],
        issues=issues,
    )


def _serialize_issues(issues: list[Issue]) -> str:
    return json.dumps([
        {
            "issue": iss.issue,
            "sources": [
                {
                    "doc_title": s.doc_title,
                    **({"section": s.section} if s.section is not None else {}),
                }
                for s in iss.sources
            ],
        }
        for iss in issues
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
                INSERT INTO results (input, flags, issues)
                VALUES (%(input)s, %(flags)s, %(issues)s)
                RETURNING id::text
                """,
                {
                    "input": json.dumps(result.input),
                    "flags": result.flags,
                    "issues": _serialize_issues(result.issues),
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
                SELECT id::text, input, flags, issues
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
                SELECT id::text, input, flags, issues
                FROM results
                WHERE %(flag)s = ANY(flags)
                ORDER BY id
                """,
                {"flag": flag},
            )
            rows = await cur.fetchall()
        return [_row_to_result(r) for r in rows]
