"""
DietarySupplementsStorage — async psycopg3 interface for the dietary_supplements lookup table.
"""

from .base import BaseStorage
from .models.dietary_supplement import DietarySupplement


def _row_to_supplement(row: dict) -> DietarySupplement:
    registered_at = row.get("registered_at")
    return DietarySupplement(
        id=row["id"],
        registration_number=row.get("registration_number"),
        status=row.get("status"),
        product_name=row["product_name"],
        manufacturer_name=row.get("manufacturer_name"),
        country_of_manufacture=row.get("country_of_manufacture"),
        scope_of_application=row.get("scope_of_application"),
        label_info=row.get("label_info"),
        registered_at=registered_at.isoformat() if registered_at else None,
    )


class DietarySupplementsStorage(BaseStorage):
    """Async context-manager for the dietary_supplements table.

    Usage::
        async with DietarySupplementsStorage() as storage:
            results = await storage.search("омега")
    """

    async def insert(self, supplement: DietarySupplement) -> int:
        """Insert a single DietarySupplement row and return its id. Also sets supplement.id."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                INSERT INTO dietary_supplements (
                    registration_number, status, product_name,
                    manufacturer_name, country_of_manufacture,
                    scope_of_application, label_info, registered_at
                ) VALUES (
                    %(registration_number)s, %(status)s, %(product_name)s,
                    %(manufacturer_name)s, %(country_of_manufacture)s,
                    %(scope_of_application)s, %(label_info)s, %(registered_at)s
                )
                RETURNING id
                """,
                {
                    "registration_number": supplement.registration_number,
                    "status": supplement.status,
                    "product_name": supplement.product_name,
                    "manufacturer_name": supplement.manufacturer_name,
                    "country_of_manufacture": supplement.country_of_manufacture,
                    "scope_of_application": supplement.scope_of_application,
                    "label_info": supplement.label_info,
                    "registered_at": supplement.registered_at,
                },
            )
            row = await cur.fetchone()
        supplement.id = row["id"]
        return row["id"]

    async def insert_many(self, supplements: list[DietarySupplement]) -> list[int]:
        """Bulk-insert multiple DietarySupplement rows; returns ids in insertion order."""
        ids: list[int] = []
        async with self._pool.connection() as conn:
            for supplement in supplements:
                cur = await conn.execute(
                    """
                    INSERT INTO dietary_supplements (
                        registration_number, status, product_name,
                        manufacturer_name, country_of_manufacture,
                        scope_of_application, label_info, registered_at
                    ) VALUES (
                        %(registration_number)s, %(status)s, %(product_name)s,
                        %(manufacturer_name)s, %(country_of_manufacture)s,
                        %(scope_of_application)s, %(label_info)s, %(registered_at)s
                    )
                    RETURNING id
                    """,
                    {
                        "registration_number": supplement.registration_number,
                        "status": supplement.status,
                        "product_name": supplement.product_name,
                        "manufacturer_name": supplement.manufacturer_name,
                        "country_of_manufacture": supplement.country_of_manufacture,
                        "scope_of_application": supplement.scope_of_application,
                        "label_info": supplement.label_info,
                        "registered_at": supplement.registered_at,
                    },
                )
                row = await cur.fetchone()
                supplement.id = row["id"]
                ids.append(row["id"])
        return ids

    async def search(self, query: str, limit: int = 3) -> list[DietarySupplement]:
        """Full-text search on product_name using TSV (russian dictionary)."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, registration_number, status, product_name,
                       manufacturer_name, country_of_manufacture,
                       scope_of_application, label_info, registered_at
                FROM dietary_supplements
                WHERE product_name_tsv @@ plainto_tsquery('russian', %(q)s)
                ORDER BY ts_rank(product_name_tsv, plainto_tsquery('russian', %(q)s)) DESC
                LIMIT %(limit)s
                """,
                {"q": query, "limit": limit},
            )
            rows = await cur.fetchall()
        return [_row_to_supplement(r) for r in rows]

    async def get(self, supplement_id: int) -> DietarySupplement | None:
        """Fetch a single DietarySupplement by id; returns None if not found."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, registration_number, status, product_name,
                       manufacturer_name, country_of_manufacture,
                       scope_of_application, label_info, registered_at
                FROM dietary_supplements WHERE id = %(id)s
                """,
                {"id": supplement_id},
            )
            row = await cur.fetchone()
        return _row_to_supplement(row) if row else None
