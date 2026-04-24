"""
ResultsStorage — async psycopg3 interface for the *results* table.
"""

import json
import logging

from .base import BaseStorage
from .models import DiagnosisResult, FormalFinding, FormalStructureResult, DiagnisisIssue, IssueSource, Result

logger = logging.getLogger(__name__)


def _deserialize_diagnosis(raw: list[dict]) -> list[DiagnosisResult]:
    results = []
    for entry in (raw or []):
        issues = [
            DiagnisisIssue(
                issue=item["issue"],
                sources=[
                    IssueSource(
                        doc_title=s["doc_title"],
                        section=s.get("section"),
                        cite=s.get("cite"),
                    )
                    for s in item.get("sources", [])
                ],
            )
            for item in entry.get("issues", [])
        ]
        results.append(DiagnosisResult(icd_code=entry.get("icd_code", ""), issues=issues))
    return results


def _row_to_result(row: dict) -> Result:
    return Result(
        id=row["id"],
        input=row["input"],
        formal=FormalStructureResult(
            findings=[FormalFinding(flag=f, issue="") for f in (row.get("flags") or [])]
        ),
        diagnosis=_deserialize_diagnosis(row.get("issues") or []),
    )


def _serialize_diagnosis(diagnosis: list[DiagnosisResult]) -> str:
    return json.dumps([
        {
            "icd_code": dr.icd_code,
            "issues": [
                {
                    "issue": iss.issue,
                    "sources": [
                        {
                            "doc_title": s.doc_title,
                            **({"section": s.section} if s.section is not None else {}),
                            **({"cite": s.cite} if s.cite is not None else {}),
                        }
                        for s in iss.sources
                    ],
                }
                for iss in dr.issues
            ],
        }
        for dr in diagnosis
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
        try:
            async with self._pool.connection() as conn:
                cur = await conn.execute(
                    """
                    INSERT INTO results (input, flags, issues)
                    VALUES (%(input)s, %(flags)s, %(issues)s)
                    RETURNING id::text
                    """,
                    {
                        "input": json.dumps(result.input),
                        "flags": result.formal.flags,
                        "issues": _serialize_diagnosis(result.diagnosis),
                    },
                )
                row = await cur.fetchone()
            result.id = row["id"]
            logger.info("💾 DB INSERT OK id=%s", row["id"])
            return row["id"]
        except Exception:
            logger.exception("💾 DB INSERT FAILED")
            raise

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
