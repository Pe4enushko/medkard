"""
DocsStorage — async psycopg3 interface for the *docs* table.
"""

import json

import psycopg
from pgvector.psycopg import register_vector_async

from .base import BaseStorage
from .models import Doc


def _row_to_doc(row: dict) -> Doc:
    return Doc(
        id=row["id"],
        file_id=row["file_id"],
        chunk=row["chunk"],
        metadata=row["metadata"],
        fact_q=row.get("fact_q"),
        procedure_q=row.get("procedure_q"),
        constraint_q=row.get("constraint_q"),
    )


# ── Storage class ─────────────────────────────────────────────────────────────

class DocsStorage(BaseStorage):
    """Async context-manager for the docs table.

    Usage::
        async with DocsStorage() as storage:
            doc_id = await storage.insert(doc)
            doc    = await storage.get(doc_id)
    """

    async def _configure(self, conn: psycopg.AsyncConnection) -> None:
        await register_vector_async(conn)

    # ── Writes ────────────────────────────────────────────────────────────────

    async def insert(self, doc: Doc) -> str:
        """Insert a single Doc and return its UUID. Also sets doc.id."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                INSERT INTO docs (
                    file_id, chunk, metadata,
                    fact_q, procedure_q, constraint_q,
                    fact_q_embedding, procedure_q_embedding, constraint_q_embedding
                ) VALUES (
                    %(file_id)s, %(chunk)s, %(metadata)s,
                    %(fact_q)s, %(procedure_q)s, %(constraint_q)s,
                    %(fact_q_embedding)s, %(procedure_q_embedding)s, %(constraint_q_embedding)s
                )
                RETURNING id::text
                """,
                {
                    "file_id": doc.file_id,
                    "chunk": doc.chunk,
                    "metadata": json.dumps(doc.metadata),
                    "fact_q": doc.fact_q,
                    "procedure_q": doc.procedure_q,
                    "constraint_q": doc.constraint_q,
                    "fact_q_embedding": doc.fact_q_embedding,
                    "procedure_q_embedding": doc.procedure_q_embedding,
                    "constraint_q_embedding": doc.constraint_q_embedding,
                },
            )
            row = await cur.fetchone()
        doc.id = row["id"]
        return row["id"]

    async def insert_many(self, docs: list[Doc]) -> list[str]:
        """Bulk-insert multiple Docs; returns list of UUIDs in insertion order.
        Also sets each Doc's id field.
        """
        ids: list[str] = []
        async with self._pool.connection() as conn:
            for doc in docs:
                cur = await conn.execute(
                    """
                    INSERT INTO docs (
                        file_id, chunk, metadata,
                        fact_q, procedure_q, constraint_q,
                        fact_q_embedding, procedure_q_embedding, constraint_q_embedding
                    ) VALUES (
                        %(file_id)s, %(chunk)s, %(metadata)s,
                        %(fact_q)s, %(procedure_q)s, %(constraint_q)s,
                        %(fact_q_embedding)s, %(procedure_q_embedding)s, %(constraint_q_embedding)s
                    )
                    RETURNING id::text
                    """,
                    {
                        "file_id": doc.file_id,
                        "chunk": doc.chunk,
                        "metadata": json.dumps(doc.metadata),
                        "fact_q": doc.fact_q,
                        "procedure_q": doc.procedure_q,
                        "constraint_q": doc.constraint_q,
                        "fact_q_embedding": doc.fact_q_embedding,
                        "procedure_q_embedding": doc.procedure_q_embedding,
                        "constraint_q_embedding": doc.constraint_q_embedding,
                    },
                )
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
                    id::text, file_id, chunk, metadata,
                    fact_q, procedure_q, constraint_q
                FROM docs
                WHERE id = %(id)s::uuid
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
                    id::text, file_id, chunk, metadata,
                    fact_q, procedure_q, constraint_q
                FROM docs
                WHERE id = ANY(%(ids)s::uuid[])
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
