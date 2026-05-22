"""
DrugsStorage — async psycopg3 interface for the drugs lookup table.
"""

from .base import BaseStorage
from .models.drug import Drug


def _row_to_drug(row: dict) -> Drug:
    return Drug(
        id=row["id"],
        trade_name=row["trade_name"],
        inn_name=row.get("inn_name"),
        dosage_form=row.get("dosage_form"),
        dosage=row.get("dosage"),
        patient_exclusions=row.get("patient_exclusions"),
    )


class DrugsStorage(BaseStorage):
    """Async context-manager for the drugs table.

    Usage::
        async with DrugsStorage() as storage:
            results = await storage.search("аспирин")
    """

    async def insert(self, drug: Drug) -> int:
        """Insert a single Drug row and return its id. Also sets drug.id."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                INSERT INTO drugs (trade_name, inn_name, dosage_form, dosage, patient_exclusions)
                VALUES (%(trade_name)s, %(inn_name)s, %(dosage_form)s, %(dosage)s, %(patient_exclusions)s)
                RETURNING id
                """,
                {
                    "trade_name": drug.trade_name,
                    "inn_name": drug.inn_name,
                    "dosage_form": drug.dosage_form,
                    "dosage": drug.dosage,
                    "patient_exclusions": drug.patient_exclusions,
                },
            )
            row = await cur.fetchone()
        drug.id = row["id"]
        return row["id"]

    async def insert_many(self, drugs: list[Drug]) -> list[int]:
        """Bulk-insert multiple Drug rows; returns ids in insertion order."""
        ids: list[int] = []
        async with self._pool.connection() as conn:
            for drug in drugs:
                cur = await conn.execute(
                    """
                    INSERT INTO drugs (trade_name, inn_name, dosage_form, dosage, patient_exclusions)
                    VALUES (%(trade_name)s, %(inn_name)s, %(dosage_form)s, %(dosage)s, %(patient_exclusions)s)
                    RETURNING id
                    """,
                    {
                        "trade_name": drug.trade_name,
                        "inn_name": drug.inn_name,
                        "dosage_form": drug.dosage_form,
                        "dosage": drug.dosage,
                        "patient_exclusions": drug.patient_exclusions,
                    },
                )
                row = await cur.fetchone()
                drug.id = row["id"]
                ids.append(row["id"])
        return ids

    async def search(self, query: str, limit: int = 6, threshold: float = 0.3) -> list[Drug]:
        """Trigram similarity search on trade_name only.

        Only rows whose similarity score meets *threshold* are returned.
        """
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, trade_name, inn_name, dosage_form, dosage, patient_exclusions,
                       similarity(trade_name, %(q)s) AS sim
                FROM drugs
                WHERE trade_name %% %(q)s
                  AND similarity(trade_name, %(q)s) >= %(threshold)s
                ORDER BY sim DESC
                LIMIT %(limit)s
                """,
                {"q": query, "limit": limit, "threshold": threshold},
            )
            rows = await cur.fetchall()
        return [_row_to_drug(r) for r in rows]

    async def search_by_inn(self, query: str, limit: int = 4) -> list[Drug]:
        """Exact case-insensitive match on inn_name."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, trade_name, inn_name, dosage_form, dosage, patient_exclusions
                FROM drugs
                WHERE LOWER(inn_name) = LOWER(%(query)s)
                LIMIT %(limit)s
                """,
                {"query": query, "limit": limit},
            )
            rows = await cur.fetchall()
        return [_row_to_drug(r) for r in rows]

    async def get(self, drug_id: int) -> Drug | None:
        """Fetch a single Drug by id; returns None if not found."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id, trade_name, inn_name, dosage_form, dosage, patient_exclusions "
                "FROM drugs WHERE id = %(id)s",
                {"id": drug_id},
            )
            row = await cur.fetchone()
        return _row_to_drug(row) if row else None
