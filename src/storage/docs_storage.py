"""
DocsStorage — async psycopg3 interface for the *docs* table.
"""

import json

import psycopg
import psycopg.rows
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from .base import BaseStorage
from .models import Doc


def _row_to_doc(row: dict) -> Doc:
    return Doc(
        id=row["id"],
        file_id=row["file_id"],
        chunk=row["chunk"],
        metadata=row["metadata"],
        embedding=row.get("embedding"),
        name=row.get("g_name"),
        mkb=list(row.get("g_mkb") or []),
        age_category=list(row.get("g_age_category") or []),
    )


_INSERT_DOC_SQL = """
    INSERT INTO docs (file_id, chunk, metadata, embedding)
    VALUES (%(file_id)s, %(chunk)s, %(metadata)s, %(embedding)s)
    RETURNING id::text
"""


def _doc_params(doc: Doc) -> dict:
    return {
        "file_id": doc.file_id,
        "chunk": doc.chunk,
        "metadata": json.dumps(doc.metadata),
        "embedding": doc.embedding,
    }


# ── Storage class ─────────────────────────────────────────────────────────────

class DocsStorage(BaseStorage):
    """Async context-manager for the docs table.

    Usage::
        async with DocsStorage() as storage:
            doc_id = await storage.insert(doc)
            doc    = await storage.get(doc_id)
    """

    async def __aenter__(self) -> "DocsStorage":
        from .base import _conninfo
        self._pool = AsyncConnectionPool(
            conninfo=_conninfo(),
            min_size=1,
            max_size=3,
            open=False,
            configure=self._configure_conn,
            kwargs={"row_factory": psycopg.rows.dict_row},
        )
        await self._pool.open()
        return self  # type: ignore[return-value]

    async def __aexit__(self, *args: object) -> None:
        await self._pool.close()

    async def _configure_conn(self, conn: psycopg.AsyncConnection) -> None:
        await register_vector_async(conn)

    # ── Writes ────────────────────────────────────────────────────────────────

    async def insert(self, doc: Doc) -> str:
        """Insert a single Doc and return its UUID. Also sets doc.id."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(_INSERT_DOC_SQL, _doc_params(doc))
            row = await cur.fetchone()
        doc.id = row["id"]
        return row["id"]

    async def insert_many(self, docs: list[Doc]) -> list[str]:
        """Bulk-insert multiple Docs in one transaction; returns UUIDs, sets each doc.id."""
        ids: list[str] = []
        async with self._pool.connection() as conn:
            for doc in docs:
                cur = await conn.execute(_INSERT_DOC_SQL, _doc_params(doc))
                result = await cur.fetchone()
                doc.id = result["id"]
                ids.append(result["id"])
        return ids

    async def replace_by_file_id(self, file_id: str, docs: list[Doc]) -> list[str]:
        """Atomically delete all rows for file_id and bulk-insert `docs` (one transaction).

        Returns new UUIDs and sets each doc.id. `docs` may be empty (pure delete).
        """
        ids: list[str] = []
        async with self._pool.connection() as conn:
            await conn.execute("DELETE FROM docs WHERE file_id = %(file_id)s", {"file_id": file_id})
            for doc in docs:
                cur = await conn.execute(_INSERT_DOC_SQL, _doc_params(doc))
                result = await cur.fetchone()
                doc.id = result["id"]
                ids.append(result["id"])
        return ids

    # ── Reads ─────────────────────────────────────────────────────────────────

    async def get(self, doc_id: str) -> Doc | None:
        """Fetch a single Doc by UUID; returns None if not found."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT
                    docs.id::text AS id, docs.file_id, docs.chunk, docs.metadata,
                    g.name AS g_name, g.mkb AS g_mkb, g.age_category AS g_age_category
                FROM docs
                LEFT JOIN guidelines g ON g.file_id = docs.file_id
                WHERE docs.id = %(id)s::uuid
                """,
                {"id": doc_id},
            )
            row = await cur.fetchone()
        return _row_to_doc(row) if row else None

    async def get_many(self, doc_ids: list[str]) -> list[Doc]:
        """Fetch multiple Docs by UUID list; preserves order."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT
                    docs.id::text AS id, docs.file_id, docs.chunk, docs.metadata,
                    g.name AS g_name, g.mkb AS g_mkb, g.age_category AS g_age_category
                FROM docs
                LEFT JOIN guidelines g ON g.file_id = docs.file_id
                WHERE docs.id = ANY(%(ids)s::uuid[])
                """,
                {"ids": doc_ids},
            )
            rows = await cur.fetchall()
        by_id = {r["id"]: _row_to_doc(r) for r in rows}
        return [by_id[i] for i in doc_ids if i in by_id]

    async def get_ingested_file_ids(self) -> set[str]:
        """Return the set of distinct file_ids already present in the docs table."""
        async with self._pool.connection() as conn:
            cur = await conn.execute("SELECT DISTINCT file_id FROM docs")
            rows = await cur.fetchall()
        return {r["file_id"] for r in rows}

    async def get_chunk_counts(self) -> dict[str, int]:
        """Return {file_id: chunk count} for every file_id present in the docs table."""
        async with self._pool.connection() as conn:
            cur = await conn.execute("SELECT file_id, COUNT(*) AS n FROM docs GROUP BY file_id")
            rows = await cur.fetchall()
        return {r["file_id"]: r["n"] for r in rows}

    async def get_duplicate_chunk_counts(self) -> dict[str, int]:
        """Return {file_id: number of duplicate chunk rows} where duplicate means
        the same (file_id, chunk) text appears more than once. For a file_id with
        one chunk repeated 3x, this counts 2 (the extra copies beyond the first)."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT file_id, SUM(n - 1) AS extra
                FROM (
                    SELECT file_id, chunk, COUNT(*) AS n
                    FROM docs
                    GROUP BY file_id, chunk
                    HAVING COUNT(*) > 1
                ) dup
                GROUP BY file_id
                """
            )
            rows = await cur.fetchall()
        return {r["file_id"]: r["extra"] for r in rows}

    async def delete_by_file_id(self, file_id: str) -> int:
        """Delete all rows for the given file_id; returns number of deleted rows."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM docs WHERE file_id = %(file_id)s",
                {"file_id": file_id},
            )
        return cur.rowcount
