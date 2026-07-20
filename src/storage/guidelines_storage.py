"""GuidelinesStorage — async psycopg3 интерфейс к таблице guidelines."""
from __future__ import annotations

from .base import BaseStorage
from .models.guideline import Guideline

_COLS = "file_id, name, mkb, age_category, developer, nps_status, published_at, usage_status"


def _row_to_guideline(row: dict) -> Guideline:
    return Guideline(
        file_id=row["file_id"],
        name=row["name"],
        mkb=list(row["mkb"] or []),
        age_category=list(row["age_category"] or []),
        developer=row["developer"],
        nps_status=row["nps_status"],
        published_at=row["published_at"],
        usage_status=row["usage_status"],
    )


class GuidelinesStorage(BaseStorage):
    """Async context-manager для таблицы guidelines (общий пул BaseStorage)."""

    async def upsert_many(self, rows: list[Guideline]) -> int:
        if not rows:
            return 0
        written = 0
        async with self._pool.connection() as conn:
            for g in rows:
                await conn.execute(
                    """
                    INSERT INTO guidelines
                        (file_id, name, mkb, age_category, developer,
                         nps_status, published_at, usage_status)
                    VALUES
                        (%(file_id)s, %(name)s, %(mkb)s, %(age_category)s, %(developer)s,
                         %(nps_status)s, %(published_at)s, %(usage_status)s)
                    ON CONFLICT (file_id) DO UPDATE SET
                        name         = EXCLUDED.name,
                        mkb          = EXCLUDED.mkb,
                        age_category = EXCLUDED.age_category,
                        developer    = EXCLUDED.developer,
                        nps_status   = EXCLUDED.nps_status,
                        published_at = EXCLUDED.published_at,
                        usage_status = EXCLUDED.usage_status
                    """,
                    {
                        "file_id": g.file_id,
                        "name": g.name,
                        "mkb": g.mkb,
                        "age_category": g.age_category,
                        "developer": g.developer,
                        "nps_status": g.nps_status,
                        "published_at": g.published_at,
                        "usage_status": g.usage_status,
                    },
                )
                written += 1
        return written

    async def get(self, file_id: str) -> Guideline | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_COLS} FROM guidelines WHERE file_id = %(file_id)s",
                {"file_id": file_id},
            )
            row = await cur.fetchone()
        return _row_to_guideline(row) if row else None

    async def all(self) -> list[Guideline]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(f"SELECT {_COLS} FROM guidelines ORDER BY file_id")
            rows = await cur.fetchall()
        return [_row_to_guideline(r) for r in rows]

    async def find_by_code(self, code: str) -> list[Guideline]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_COLS} FROM guidelines WHERE %(code)s = ANY(mkb)",
                {"code": code.strip().upper()},
            )
            rows = await cur.fetchall()
        return [_row_to_guideline(r) for r in rows]

    async def delete(self, file_id: str) -> int:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM guidelines WHERE file_id = %(file_id)s", {"file_id": file_id}
            )
        return cur.rowcount

    async def find_by_prefix(self, prefix: str) -> list[Guideline]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_COLS} FROM guidelines "
                "WHERE EXISTS (SELECT 1 FROM unnest(mkb) c WHERE split_part(c, '.', 1) = %(prefix)s)",
                {"prefix": prefix.strip().upper()},
            )
            rows = await cur.fetchall()
        return [_row_to_guideline(r) for r in rows]
