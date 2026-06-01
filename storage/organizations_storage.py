"""
OrganizationsStorage — async psycopg3 interface for the *organizations* table.
"""

from __future__ import annotations

from .base import BaseStorage


class OrganizationsStorage(BaseStorage):
    async def get_id_by_name(self, name: str) -> str:
        """Return an organization's UUID for its exact name."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id::text FROM organizations WHERE name = %(name)s",
                {"name": name},
            )
            row = await cur.fetchone()

        if row is None:
            raise ValueError(f"Organization not found: {name}")
        return row["id"]
