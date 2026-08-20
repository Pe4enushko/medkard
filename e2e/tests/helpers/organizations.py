"""
organizations.py — create/delete throwaway organizations for e2e tests.

Direct INSERT/DELETE against the organizations table: there is no public API
route for creating an organization (organizations are provisioned manually
today), so a throwaway test org has no contract to go through — it talks to
the table the same way any operator/migration would.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from storage.base import BaseStorage  # noqa: E402


class OrganizationFixtures(BaseStorage):
    """Async context-manager for creating/removing e2e-test organizations.

    Usage::
        async with OrganizationFixtures() as orgs:
            org_id = await orgs.create_org("smoke-push-log-a1b2c3d4")
            ...
            await orgs.delete_org(org_id)
    """

    async def create_org(self, name: str) -> str:
        """Insert a new organization and return its UUID (as text)."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO organizations (name) VALUES (%(name)s) RETURNING id::text",
                {"name": name},
            )
            row = await cur.fetchone()
        return row["id"]

    async def delete_org(self, org_id: str) -> None:
        """Delete an organization by id. No-op if it no longer exists."""
        async with self._pool.connection() as conn:
            await conn.execute(
                "DELETE FROM organizations WHERE id = %(id)s::uuid",
                {"id": org_id},
            )
