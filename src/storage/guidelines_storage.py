"""GuidelinesStorage — async psycopg3 интерфейс к таблице guidelines."""
from __future__ import annotations

import asyncio
import os

import psycopg.rows
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from RAG.retrieval.embeddings import embed

from .base import BaseStorage, _conninfo
from .models.guideline import Guideline, name_embed_input

_COLS = "file_id, name, mkb, age_category, developer, nps_status, published_at, usage_status"
_shared_guidelines_pool: AsyncConnectionPool | None = None
_shared_guidelines_pool_lock = asyncio.Lock()


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
    """Async context-manager для таблицы guidelines (собственный пул с pgvector кодеком)."""

    async def __aenter__(self) -> "GuidelinesStorage":
        global _shared_guidelines_pool
        async with _shared_guidelines_pool_lock:
            if _shared_guidelines_pool is None or _shared_guidelines_pool.closed:
                _shared_guidelines_pool = AsyncConnectionPool(
                    conninfo=_conninfo(),
                    min_size=int(os.environ.get("GUIDELINES_POOL_MIN_SIZE", "1")),
                    max_size=int(os.environ.get("GUIDELINES_POOL_MAX_SIZE", "3")),
                    open=False,
                    configure=self._configure_conn,
                    kwargs={"row_factory": psycopg.rows.dict_row},
                )
                await _shared_guidelines_pool.open()
        self._pool = _shared_guidelines_pool
        return self  # type: ignore[return-value]

    async def __aexit__(self, *args: object) -> None:
        pass  # shared pool; keep it open for concurrent audit cards

    async def _configure_conn(self, conn: psycopg.AsyncConnection) -> None:
        await register_vector_async(conn)

    async def upsert_many(self, rows: list[Guideline]) -> int:
        if not rows:
            return 0
        written = 0
        async with self._pool.connection() as conn:
            for g in rows:
                if g.name_embedding is None:
                    g.name_embedding = await embed(name_embed_input(g.name, g.age_category))
                await conn.execute(
                    """
                    INSERT INTO guidelines
                        (file_id, name, mkb, age_category, developer,
                         nps_status, published_at, usage_status, name_embedding)
                    VALUES
                        (%(file_id)s, %(name)s, %(mkb)s, %(age_category)s, %(developer)s,
                         %(nps_status)s, %(published_at)s, %(usage_status)s, %(name_embedding)s)
                    ON CONFLICT (file_id) DO UPDATE SET
                        name           = EXCLUDED.name,
                        mkb            = EXCLUDED.mkb,
                        age_category   = EXCLUDED.age_category,
                        developer      = EXCLUDED.developer,
                        nps_status     = EXCLUDED.nps_status,
                        published_at   = EXCLUDED.published_at,
                        usage_status   = EXCLUDED.usage_status,
                        name_embedding = EXCLUDED.name_embedding
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
                        "name_embedding": g.name_embedding,
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
