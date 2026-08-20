"""
cards.py — push cards over HTTP and inspect/clean up done_cards + push_log
rows for e2e tests.

push_card is a thin wrapper over POST /visits/push — no retry, no auth
handling beyond passing the bearer token through, so a test's assertions
see the real HTTP response untouched.

stage_audited fabricates a completed formal-structure audit result directly
in the database (status='done' with a non-null formal_result), without
running any real LLM checker. This exists purely to put a done_cards row
into the state migration 027's push_log trigger needs to see in order to
log overrode_audit=true on the next push — it is not a substitute for
actually exercising the audit pipeline (scripts/smoke-cards-push.sh's
--with-audit flag does that, at the cost of real LLM calls).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from storage.base import BaseStorage  # noqa: E402


async def push_card(
    client: httpx.AsyncClient, base_url: str, org: str, raw_key: str, card: dict
) -> httpx.Response:
    """POST card to /visits/push?org=<org>, bearer-authenticated. Returns the raw response."""
    return await client.post(
        f"{base_url.rstrip('/')}/visits/push",
        params={"org": org},
        json=card,
        headers={"Authorization": f"Bearer {raw_key}"},
    )


class CardFixtures(BaseStorage):
    """Async context-manager for staging/inspecting/cleaning up e2e-test cards.

    Usage::
        async with CardFixtures() as cards:
            await cards.stage_audited(card_guid)
            row = await cards.card_row(card_guid)
            log = await cards.push_log_rows(card_guid)
            await cards.delete_push_log(card_guid)
            await cards.delete_cards(card_guid)
    """

    async def stage_audited(self, card_guid: str) -> None:
        """Mark an existing done_cards row as a completed formal-structure audit.

        Sets status='done' and a non-null formal_result (one fabricated
        finding), ignored=FALSE, broken=FALSE. The row must already exist
        (created by a prior push) — this only flips its state.
        """
        fake_formal_result = json.dumps(
            [{"flag": "e2e_fixture", "issue": "e2e fixture finding", "source": "", "comment": ""}],
            ensure_ascii=False,
        )
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                UPDATE done_cards
                SET status = 'done',
                    formal_result = %(formal)s::jsonb,
                    ignored = FALSE,
                    broken = FALSE
                WHERE card_guid = %(guid)s
                """,
                {"guid": card_guid, "formal": fake_formal_result},
            )

    async def card_row(self, card_guid: str) -> dict | None:
        """Return the full done_cards row for a guid, or None if it doesn't exist."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM done_cards WHERE card_guid = %(guid)s",
                {"guid": card_guid},
            )
            return await cur.fetchone()

    async def push_log_rows(self, card_guid: str) -> list[dict]:
        """Return every push_log row for a guid, oldest first."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM push_log WHERE card_guid = %(guid)s ORDER BY pushed_at",
                {"guid": card_guid},
            )
            return await cur.fetchall()

    async def delete_cards(self, card_guid: str) -> int:
        """Delete the done_cards row for a guid. Returns the row count deleted."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM done_cards WHERE card_guid = %(guid)s",
                {"guid": card_guid},
            )
            return cur.rowcount

    async def delete_push_log(self, card_guid: str) -> int:
        """Delete every push_log row for a guid. Returns the row count deleted."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM push_log WHERE card_guid = %(guid)s",
                {"guid": card_guid},
            )
            return cur.rowcount

    async def push_metrics_for_org_today(self, organization_name: str) -> dict | None:
        """Return today's push_metrics_by_date row for an organization, or None if absent.

        Keys: pushes_total, pushes_overrode_audit, pushes_no_override (matching
        the view's columns exactly — see migrations/027_audit_overwrite_journal.sql).
        """
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT pushes_total, pushes_overrode_audit, pushes_no_override "
                "FROM push_metrics_by_date "
                "WHERE organization_name = %(org)s AND push_date = current_date",
                {"org": organization_name},
            )
            return await cur.fetchone()
