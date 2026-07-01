"""
ApiKeysStorage — async psycopg3 interface for the *api_keys* table and its
*api_key_organizations* scoping join table.

Unified per-app keys, not per-organization: there's a single integrating
service today, so a key authenticates "this is our trusted app", but it is
scoped to a specific set of organizations (every key must have at least
one) — the caller still names which org's cards it wants per request
(?org=...), and access is only granted if that org is in the key's scope.
Only the SHA-256 hash of the raw key is ever stored.
"""

from __future__ import annotations

import hashlib

from .base import BaseStorage


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


class ApiKeysStorage(BaseStorage):
    async def create_key(self, label: str, raw_key: str, organization_ids: list[str]) -> str:
        """Insert a new active key under *label*, scoped to *organization_ids*.

        *organization_ids* must be non-empty — a key must authorize at
        least one organization.
        """
        if not organization_ids:
            raise ValueError("organization_ids must be non-empty: a key must be scoped to at least one org")

        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                INSERT INTO api_keys (label, key_hash, key_prefix)
                VALUES (%(label)s, %(hash)s, %(prefix)s)
                RETURNING id::text
                """,
                {"label": label, "hash": hash_key(raw_key), "prefix": raw_key[:8]},
            )
            row = await cur.fetchone()
            key_id = row["id"]

            await conn.execute(
                """
                INSERT INTO api_key_organizations (api_key_id, organization_id)
                SELECT %(key_id)s::uuid, org_id FROM unnest(%(org_ids)s::uuid[]) AS org_id
                """,
                {"key_id": key_id, "org_ids": organization_ids},
            )
        return key_id

    async def revoke_key(self, key_id: str) -> None:
        """Disable a key by id, without deleting its row."""
        async with self._pool.connection() as conn:
            await conn.execute(
                "UPDATE api_keys SET revoked_at = now() WHERE id = %(id)s::uuid",
                {"id": key_id},
            )

    async def revoke_by_raw_key(self, raw_key: str) -> bool:
        """Disable a key by its raw value (no id lookup needed). Returns True if a row was updated."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "UPDATE api_keys SET revoked_at = now() WHERE key_hash = %(hash)s AND revoked_at IS NULL",
                {"hash": hash_key(raw_key)},
            )
        return cur.rowcount > 0

    async def is_key_authorized_for_org(self, raw_key: str, organization_id: str) -> bool:
        """Return True if *raw_key* is active and scoped to *organization_id*."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT 1 FROM api_keys k
                JOIN api_key_organizations ko ON ko.api_key_id = k.id
                WHERE k.key_hash = %(hash)s
                  AND k.revoked_at IS NULL
                  AND ko.organization_id = %(org_id)s::uuid
                """,
                {"hash": hash_key(raw_key), "org_id": organization_id},
            )
            row = await cur.fetchone()
        return row is not None

    async def is_valid_key(self, raw_key: str) -> bool:
        """Return True if *raw_key* matches any active (non-revoked) key, regardless of org scope."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT 1 FROM api_keys WHERE key_hash = %(hash)s AND revoked_at IS NULL",
                {"hash": hash_key(raw_key)},
            )
            row = await cur.fetchone()
        return row is not None
