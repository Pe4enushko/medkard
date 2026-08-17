"""
StatsStorage — async psycopg3 interface for per-organization storage figures.

Spans done_cards and audit_overwrite_journal, so it lives here rather than in
DoneCardsStorage (which is scoped to a single table).

Sizes are measured with pg_column_size() summed over the organization's rows,
not pg_total_relation_size(): the question is how much space one org's data
occupies, not how large the table is. pg_column_size() reports the stored size
of each row including TOAST compression, which matters because card_data and
diag_result are large compressed JSONB.

Consequence worth knowing when reading the numbers: this counts payload only.
Index space and per-page overhead are excluded, so summing every org will come
out below the table's on-disk size — it answers "how much data does this org
have", not "how much disk would deleting it free".
"""

from __future__ import annotations

import logging

from .base import BaseStorage

logger = logging.getLogger(__name__)


class StatsStorage(BaseStorage):
    """Async context-manager for cross-table statistics queries.

    Usage::
        async with StatsStorage() as stats:
            sizes = await stats.storage_kb(organization_id=org_id)
    """

    async def storage_kb(self, *, organization_id: str) -> dict[str, float]:
        """Return kilobytes stored for an organization, per table plus total.

        Keys: done_cards_kb, audit_overwrite_journal_kb, total_kb.
        An organization with no rows yields zeros, never None.
        """
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT
                    COALESCE((
                        SELECT sum(pg_column_size(done_cards.*))
                        FROM done_cards
                        WHERE organization_id = %(org_id)s
                    ), 0) / 1024.0 AS done_cards_kb,
                    COALESCE((
                        SELECT sum(pg_column_size(audit_overwrite_journal.*))
                        FROM audit_overwrite_journal
                        WHERE organization_id = %(org_id)s
                    ), 0) / 1024.0 AS audit_overwrite_journal_kb
                """,
                {"org_id": organization_id},
            )
            row = await cur.fetchone()

        done_cards_kb = round(float(row["done_cards_kb"]), 2)
        journal_kb = round(float(row["audit_overwrite_journal_kb"]), 2)
        result = {
            "done_cards_kb": done_cards_kb,
            "audit_overwrite_journal_kb": journal_kb,
            # Rounded from the unrounded parts so the total never drifts from
            # their sum by more than a cent of a kilobyte.
            "total_kb": round(
                float(row["done_cards_kb"]) + float(row["audit_overwrite_journal_kb"]), 2
            ),
        }
        logger.info("💾 storage stats org_id=%s %s", organization_id, result)
        return result
