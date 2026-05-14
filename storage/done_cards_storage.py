"""
DoneCardsStorage — async psycopg3 interface for the *done_cards* table.

Upserts one row per card identified by card_guid. If a row with that guid
already exists it is updated in place; otherwise a new row is inserted.
Cards with no GUID are always inserted as new rows.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from .base import BaseStorage
from .models.result import DiagnosisResult, FormalStructureResult

logger = logging.getLogger(__name__)


def _formal_json(formal: FormalStructureResult) -> str:
    return json.dumps(
        [{"flag": f.flag, "issue": f.issue} for f in formal.findings],
        ensure_ascii=False,
    )


def _diag_json(diagnosis: list[DiagnosisResult]) -> str:
    return json.dumps(
        [
            {
                "icd_code": dr.icd_code,
                "issues": [
                    {
                        "issue": iss.issue,
                        "sources": [
                            {"doc_title": s.doc_title, "section": s.section, "cite": s.cite}
                            for s in iss.sources
                        ],
                    }
                    for iss in dr.issues
                ],
            }
            for dr in diagnosis
        ],
        ensure_ascii=False,
    )


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
        started_at: datetime,
        finished_at: datetime,
        card_guid: str | None = None,
    ) -> str:
        """Insert or update a done_cards row and return its UUID.

        If *card_guid* is provided and a row with that guid already exists,
        all fields are updated in place. Cards without a guid are always
        inserted as new rows.
        """
        formal_json = _formal_json(formal)
        diag_json = _diag_json(diagnosis)

        try:
            async with self._pool.connection() as conn:
                if card_guid:
                    cur = await conn.execute(
                        """
                        INSERT INTO done_cards
                            (card_guid, card_data, formal_result, diag_result, token_count, time_ms, started_at, finished_at, ignored)
                        VALUES
                            (%(guid)s, %(data)s::jsonb, %(formal)s::jsonb, %(diag)s::jsonb, %(tokens)s, %(ms)s, %(started_at)s, %(finished_at)s, FALSE)
                        ON CONFLICT (card_guid) DO UPDATE SET
                            card_data     = EXCLUDED.card_data,
                            formal_result = EXCLUDED.formal_result,
                            diag_result   = EXCLUDED.diag_result,
                            token_count   = EXCLUDED.token_count,
                            time_ms       = EXCLUDED.time_ms,
                            started_at    = EXCLUDED.started_at,
                            finished_at   = EXCLUDED.finished_at,
                            ignored       = FALSE
                        RETURNING id::text
                        """,
                        {
                            "guid":        card_guid,
                            "data":        card_data,
                            "formal":      formal_json,
                            "diag":        diag_json,
                            "tokens":      token_count,
                            "ms":          time_ms,
                            "started_at":  started_at,
                            "finished_at": finished_at,
                        },
                    )
                else:
                    cur = await conn.execute(
                        """
                        INSERT INTO done_cards
                            (card_guid, card_data, formal_result, diag_result, token_count, time_ms, started_at, finished_at, ignored)
                        VALUES
                            (NULL, %(data)s::jsonb, %(formal)s::jsonb, %(diag)s::jsonb, %(tokens)s, %(ms)s, %(started_at)s, %(finished_at)s, FALSE)
                        RETURNING id::text
                        """,
                        {
                            "data":        card_data,
                            "formal":      formal_json,
                            "diag":        diag_json,
                            "tokens":      token_count,
                            "ms":          time_ms,
                            "started_at":  started_at,
                            "finished_at": finished_at,
                        },
                    )

                row = await cur.fetchone()
            row_id: str = row["id"]
            logger.info("💾 done_cards UPSERT OK id=%s guid=%s", row_id, card_guid)
            return row_id
        except Exception:
            logger.exception("💾 done_cards UPSERT FAILED guid=%s", card_guid)
            raise

    async def upsert_ignored(self, *, card_guid: str, card_data: str) -> str:
        """Insert or update a done_cards row for an ICD-ignored card.

        Stores card_guid and raw input; all audit columns are left NULL.
        """
        try:
            async with self._pool.connection() as conn:
                cur = await conn.execute(
                    """
                    INSERT INTO done_cards (card_guid, card_data, ignored)
                    VALUES (%(guid)s, %(data)s::jsonb, TRUE)
                    ON CONFLICT (card_guid) DO UPDATE SET
                        card_data = EXCLUDED.card_data,
                        ignored   = TRUE
                    RETURNING id::text
                    """,
                    {"guid": card_guid, "data": card_data},
                )
                row = await cur.fetchone()
            row_id: str = row["id"]
            logger.info("💾 done_cards UPSERT_IGNORED OK id=%s guid=%s", row_id, card_guid)
            return row_id
        except Exception:
            logger.exception("💾 done_cards UPSERT_IGNORED FAILED guid=%s", card_guid)
            raise

    async def get_done_guids(self) -> set[str]:
        """Return the set of all non-null card_guid values in done_cards (including ignored)."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT card_guid FROM done_cards WHERE card_guid IS NOT NULL"
            )
            rows = await cur.fetchall()
        guids = {row["card_guid"] for row in rows}
        logger.info("💾 done_cards loaded %d done guid(s)", len(guids))
        return guids
