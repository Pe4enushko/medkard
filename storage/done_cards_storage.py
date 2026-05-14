"""
DoneCardsStorage — async psycopg3 interface for the *done_cards* table.

Upserts one row per card identified by card_guid. If a row with that guid
already exists it is updated in place; otherwise a new row is inserted.
Cards with no GUID are always inserted as new rows.
"""

from __future__ import annotations

import json
import logging

from .base import BaseStorage
from .models.result import DiagnosisResult, FormalStructureResult

logger = logging.getLogger(__name__)


def _formal_text(formal: FormalStructureResult) -> str:
    return formal.pretty_format()


def _diag_text(diagnosis: list[DiagnosisResult]) -> str:
    if not diagnosis:
        return "Diagnoses: none"
    return "\n".join(dr.pretty_format() for dr in diagnosis)


class DoneCardsStorage(BaseStorage):
    """Async context-manager for the done_cards table.

    Usage::
        async with DoneCardsStorage() as storage:
            await storage.upsert(
                card_guid=guid,
                card_data=visit_json_str,
                formal=formal_result,
                diagnosis=diagnosis_results,
                token_count=0,
                time_ms=elapsed_ms,
            )
    """

    async def upsert(
        self,
        *,
        card_data: str,
        formal: FormalStructureResult,
        diagnosis: list[DiagnosisResult],
        token_count: int,
        time_ms: int,
        card_guid: str | None = None,
    ) -> str:
        """Insert or update a done_cards row and return its UUID.

        If *card_guid* is provided and a row with that guid already exists,
        all fields are updated in place. Cards without a guid are always
        inserted as new rows.
        """
        formal_text = _formal_text(formal)
        diag_text = _diag_text(diagnosis)

        try:
            async with self._pool.connection() as conn:
                if card_guid:
                    cur = await conn.execute(
                        """
                        INSERT INTO done_cards
                            (card_guid, card_data, formal_result, diag_result, token_count, time_ms)
                        VALUES
                            (%(guid)s, %(data)s, %(formal)s, %(diag)s, %(tokens)s, %(ms)s)
                        ON CONFLICT (card_guid) DO UPDATE SET
                            card_data    = EXCLUDED.card_data,
                            formal_result = EXCLUDED.formal_result,
                            diag_result  = EXCLUDED.diag_result,
                            token_count  = EXCLUDED.token_count,
                            time_ms      = EXCLUDED.time_ms
                        RETURNING id::text
                        """,
                        {
                            "guid":   card_guid,
                            "data":   card_data,
                            "formal": formal_text,
                            "diag":   diag_text,
                            "tokens": token_count,
                            "ms":     time_ms,
                        },
                    )
                else:
                    cur = await conn.execute(
                        """
                        INSERT INTO done_cards
                            (card_guid, card_data, formal_result, diag_result, token_count, time_ms)
                        VALUES
                            (NULL, %(data)s, %(formal)s, %(diag)s, %(tokens)s, %(ms)s)
                        RETURNING id::text
                        """,
                        {
                            "data":   card_data,
                            "formal": formal_text,
                            "diag":   diag_text,
                            "tokens": token_count,
                            "ms":     time_ms,
                        },
                    )

                row = await cur.fetchone()
            row_id: str = row["id"]
            logger.info("💾 done_cards UPSERT OK id=%s guid=%s", row_id, card_guid)
            return row_id
        except Exception:
            logger.exception("💾 done_cards UPSERT FAILED guid=%s", card_guid)
            raise

    async def get_done_guids(self) -> set[str]:
        """Return the set of all non-null card_guid values in done_cards."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT card_guid FROM done_cards WHERE card_guid IS NOT NULL"
            )
            rows = await cur.fetchall()
        guids = {row["card_guid"] for row in rows}
        logger.info("💾 done_cards loaded %d done guid(s)", len(guids))
        return guids
