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

    async def get_id_by_name_ci(self, name: str) -> str:
        """Return an organization's UUID for its name, matched case-insensitively.

        For external callers (the pull API) that may not send the name with
        the exact casing stored in the DB.
        """
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id::text FROM organizations WHERE lower(name) = lower(%(name)s)",
                {"name": name},
            )
            row = await cur.fetchone()

        if row is None:
            raise ValueError(f"Organization not found: {name}")
        return row["id"]

    async def get_name_by_id(self, organization_id: str) -> str:
        """Return an organization's name for its UUID."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT name FROM organizations WHERE id = %(id)s::uuid",
                {"id": organization_id},
            )
            row = await cur.fetchone()

        if row is None:
            raise ValueError(f"Organization not found: {organization_id}")
        return row["name"]
