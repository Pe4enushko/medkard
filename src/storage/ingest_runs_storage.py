"""ingest_runs_storage.py — resume-state for reingest (table: 023_ingest_runs.sql).

Invariant: content_hash is written ONLY by mark_done; upsert_pending and
mark_failed preserve the existing hash, so it always reflects the last
successful ('done') reingest of the file.
"""
from __future__ import annotations

from .base import BaseStorage


class IngestRunsStorage(BaseStorage):
    async def get_all(self) -> dict[str, tuple[str, str | None]]:
        """file_id -> (status, content_hash) for every recorded file."""
        async with self._pool.connection() as conn:
            cur = await conn.execute("SELECT file_id, status, content_hash FROM ingest_runs")
            rows = await cur.fetchall()
        return {r["file_id"]: (r["status"], r["content_hash"]) for r in rows}

    async def upsert_pending(self, file_id: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO ingest_runs (file_id, status)
                VALUES (%(file_id)s, 'pending')
                ON CONFLICT (file_id) DO UPDATE SET
                    status = 'pending', updated_at = now()
                """,
                {"file_id": file_id},
            )

    async def mark_done(self, file_id: str, content_hash: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO ingest_runs (file_id, status, content_hash, error)
                VALUES (%(file_id)s, 'done', %(h)s, NULL)
                ON CONFLICT (file_id) DO UPDATE SET
                    status = 'done', content_hash = %(h)s, error = NULL, updated_at = now()
                """,
                {"file_id": file_id, "h": content_hash},
            )

    async def mark_failed(self, file_id: str, error: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO ingest_runs (file_id, status, error)
                VALUES (%(file_id)s, 'failed', %(e)s)
                ON CONFLICT (file_id) DO UPDATE SET
                    status = 'failed', error = %(e)s, updated_at = now()
                """,
                {"file_id": file_id, "e": error},
            )
