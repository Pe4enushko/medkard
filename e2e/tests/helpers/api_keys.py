"""
api_keys.py — mint/delete throwaway API keys for e2e tests.

issue_key goes through ApiKeysStorage.create_key (the same path
scripts/operator/create-api-key.py uses) rather than inserting rows directly, so a
test key is authorized exactly the way a real one would be.

delete_key removes the row outright (not ApiKeysStorage.revoke_key, which
only sets revoked_at): a real key's revocation history is worth keeping, but
a key minted here lives for seconds and leaves no audit trail worth
preserving — a dead revoked row on every run would just accumulate.

Deletion is by label, not by id: create_key inserts the key and its org
scope as two separate statements with no explicit transaction, so a failure
between them can leave an inserted key whose id the caller never received.
The label is always known up front (the caller generates it), so it is the
reliable handle for cleanup.
"""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from storage.api_keys_storage import ApiKeysStorage  # noqa: E402
from storage.base import BaseStorage  # noqa: E402


async def issue_key(label: str, org_id: str) -> tuple[str, str]:
    """Mint a key scoped to one organization. Returns (key_id, raw_key)."""
    raw_key = f"medkard_e2e_{secrets.token_urlsafe(24)}"
    async with ApiKeysStorage() as api_keys:
        key_id = await api_keys.create_key(label, raw_key, [org_id])
    return str(key_id), raw_key


class ApiKeyFixtures(BaseStorage):
    """Async context-manager for removing e2e-test API keys.

    Usage::
        async with ApiKeyFixtures() as keys:
            deleted = await keys.delete_key("smoke-push-log-a1b2c3d4")
    """

    async def delete_key(self, label: str) -> int:
        """Delete every key row with this label. Returns the row count deleted."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM api_keys WHERE label = %(label)s",
                {"label": label},
            )
            return cur.rowcount

    async def count_key_scopes(self, key_id: str) -> int:
        """Count remaining api_key_organizations rows for a key id.

        Used only to assert teardown actually cascaded — organizations does
        NOT cascade-delete api_key_organizations (nothing links a key to an
        org in that direction), so this is a sanity check, not a cleanup step.
        """
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT count(*) AS n FROM api_key_organizations WHERE api_key_id = %(id)s::uuid",
                {"id": key_id},
            )
            row = await cur.fetchone()
        return row["n"]
