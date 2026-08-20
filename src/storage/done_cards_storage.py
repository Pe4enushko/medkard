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
from .models.result import DiagnosisResult, FormalStructureResult, IcdCodingIssue

logger = logging.getLogger(__name__)


def _formal_json(formal: FormalStructureResult) -> str:
    return json.dumps(
        [{"flag": f.flag, "issue": f.issue, "source": f.source, "comment": f.comment} for f in formal.findings],
        ensure_ascii=False,
    )


def _icd_check_json(issues: list[IcdCodingIssue]) -> str:
    return json.dumps(
        [issue.to_dict() for issue in issues],
        ensure_ascii=False,
    )


def _diag_json(diagnosis: list[DiagnosisResult]) -> str:
    return json.dumps(
        [
            {
                "icd_code": dr.icd_code,
                "guideline_file_id": dr.guideline_file_id,
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
        icd_check: list[IcdCodingIssue],
        token_count: int,
        time_ms: int,
        started_at: datetime,
        finished_at: datetime,
        card_guid: str | None = None,
        organization_id: str | None = None,
    ) -> str:
        """Insert or update a done_cards row and return its UUID.

        If *card_guid* is provided and a row with that guid already exists,
        all fields are updated in place. Cards without a guid are always
        inserted as new rows.
        """
        formal_json = _formal_json(formal)
        diag_json = _diag_json(diagnosis)
        icd_check_json = _icd_check_json(icd_check)

        try:
            async with self._pool.connection() as conn:
                if card_guid:
                    cur = await conn.execute(
                        """
                        INSERT INTO done_cards
                            (card_guid, card_data, formal_result, diag_result, icd_check_result, token_count, time_ms, started_at, finished_at, ignored, organization_id, status)
                        VALUES
                            (%(guid)s, %(data)s::jsonb, %(formal)s::jsonb, %(diag)s::jsonb, %(icd_check)s::jsonb, %(tokens)s, %(ms)s, %(started_at)s, %(finished_at)s, FALSE, %(org_id)s, 'done')
                        ON CONFLICT (card_guid) DO UPDATE SET
                            card_data         = EXCLUDED.card_data,
                            status            = 'done',
                            formal_result     = EXCLUDED.formal_result,
                            diag_result       = EXCLUDED.diag_result,
                            icd_check_result  = EXCLUDED.icd_check_result,
                            token_count       = EXCLUDED.token_count,
                            time_ms           = EXCLUDED.time_ms,
                            started_at        = EXCLUDED.started_at,
                            finished_at       = EXCLUDED.finished_at,
                            ignored           = FALSE,
                            broken            = FALSE,
                            organization_id   = EXCLUDED.organization_id
                        RETURNING id::text
                        """,
                        {
                            "guid":        card_guid,
                            "data":        card_data,
                            "formal":      formal_json,
                            "diag":        diag_json,
                            "icd_check":   icd_check_json,
                            "tokens":      token_count,
                            "ms":          time_ms,
                            "started_at":  started_at,
                            "finished_at": finished_at,
                            "org_id":      organization_id,
                        },
                    )
                else:
                    cur = await conn.execute(
                        """
                        INSERT INTO done_cards
                            (card_guid, card_data, formal_result, diag_result, icd_check_result, token_count, time_ms, started_at, finished_at, ignored, organization_id, status)
                        VALUES
                            (NULL, %(data)s::jsonb, %(formal)s::jsonb, %(diag)s::jsonb, %(icd_check)s::jsonb, %(tokens)s, %(ms)s, %(started_at)s, %(finished_at)s, FALSE, %(org_id)s, 'done')
                        RETURNING id::text
                        """,
                        {
                            "data":        card_data,
                            "formal":      formal_json,
                            "diag":        diag_json,
                            "icd_check":   icd_check_json,
                            "tokens":      token_count,
                            "ms":          time_ms,
                            "started_at":  started_at,
                            "finished_at": finished_at,
                            "org_id":      organization_id,
                        },
                    )

                row = await cur.fetchone()
            row_id: str = row["id"]
            logger.info("💾 done_cards UPSERT OK id=%s guid=%s", row_id, card_guid)
            return row_id
        except Exception:
            logger.exception("💾 done_cards UPSERT FAILED guid=%s", card_guid)
            raise

    async def upsert_ignored(
        self,
        *,
        card_guid: str,
        card_data: str,
        organization_id: str | None = None,
    ) -> str:
        """Insert or update a done_cards row for an ICD-ignored card.

        Stores card_guid and raw input; all audit columns are left NULL.
        """
        try:
            async with self._pool.connection() as conn:
                cur = await conn.execute(
                    """
                    INSERT INTO done_cards (card_guid, card_data, ignored, organization_id, status)
                    VALUES (%(guid)s, %(data)s::jsonb, TRUE, %(org_id)s, 'done')
                    ON CONFLICT (card_guid) DO UPDATE SET
                        card_data       = EXCLUDED.card_data,
                        status          = 'done',
                        ignored         = TRUE,
                        broken          = FALSE,
                        organization_id = EXCLUDED.organization_id
                    RETURNING id::text
                    """,
                    {"guid": card_guid, "data": card_data, "org_id": organization_id},
                )
                row = await cur.fetchone()
            row_id: str = row["id"]
            logger.info("💾 done_cards UPSERT_IGNORED OK id=%s guid=%s", row_id, card_guid)
            return row_id
        except Exception:
            logger.exception("💾 done_cards UPSERT_IGNORED FAILED guid=%s", card_guid)
            raise

    async def upsert_broken(
        self,
        *,
        card_data: str,
        stacktrace: str,
        started_at: datetime,
        card_guid: str | None = None,
        organization_id: str | None = None,
    ) -> str:
        """Insert or update a done_cards row for a card that failed with an exception.

        Sets broken=TRUE and stores the stacktrace; all audit columns are left NULL.
        """
        try:
            async with self._pool.connection() as conn:
                if card_guid:
                    cur = await conn.execute(
                        """
                        INSERT INTO done_cards
                            (card_guid, card_data, ignored, broken, stacktrace, started_at, organization_id, status)
                        VALUES
                            (%(guid)s, %(data)s::jsonb, FALSE, TRUE, %(stacktrace)s, %(started_at)s, %(org_id)s, 'done')
                        ON CONFLICT (card_guid) DO UPDATE SET
                            card_data       = EXCLUDED.card_data,
                            status          = 'done',
                            ignored         = FALSE,
                            broken          = TRUE,
                            stacktrace      = EXCLUDED.stacktrace,
                            started_at      = EXCLUDED.started_at,
                            organization_id = EXCLUDED.organization_id
                        RETURNING id::text
                        """,
                        {
                            "guid":       card_guid,
                            "data":       card_data,
                            "stacktrace": stacktrace,
                            "started_at": started_at,
                            "org_id":     organization_id,
                        },
                    )
                else:
                    cur = await conn.execute(
                        """
                        INSERT INTO done_cards
                            (card_guid, card_data, ignored, broken, stacktrace, started_at, organization_id, status)
                        VALUES
                            (NULL, %(data)s::jsonb, FALSE, TRUE, %(stacktrace)s, %(started_at)s, %(org_id)s, 'done')
                        RETURNING id::text
                        """,
                        {
                            "data":       card_data,
                            "stacktrace": stacktrace,
                            "started_at": started_at,
                            "org_id":     organization_id,
                        },
                    )

                row = await cur.fetchone()
            row_id: str = row["id"]
            logger.info("💾 done_cards UPSERT_BROKEN OK id=%s guid=%s", row_id, card_guid)
            return row_id
        except Exception:
            logger.exception("💾 done_cards UPSERT_BROKEN FAILED guid=%s", card_guid)
            raise

    async def upsert_pending(
        self,
        *,
        card_guid: str,
        card_data: str,
        organization_id: str | None = None,
    ) -> str:
        """Insert or update a done_cards row with fresh raw data awaiting audit.

        Sets status='pending' and clears every audit-derived column (results,
        ignored, broken, stacktrace) — a pushed update means the previous
        audit outcome, if any, is stale and must be recomputed from scratch.

        Also stamps pushed_at = now() on every call. This is the signal the
        done_cards_log_push trigger (migration 027) uses to tell a genuine
        push apart from an unrelated UPDATE that happens to touch an
        already-'pending' row (e.g. replace_priem) — see the trigger's WHEN
        clause and the comment beside it in the migration for why
        NEW.status = 'pending' alone is not sufficient. No other write path
        may set this column.
        """
        try:
            async with self._pool.connection() as conn:
                cur = await conn.execute(
                    """
                    INSERT INTO done_cards
                        (card_guid, card_data, status, organization_id, pushed_at)
                    VALUES
                        (%(guid)s, %(data)s::jsonb, 'pending', %(org_id)s, now())
                    ON CONFLICT (card_guid) DO UPDATE SET
                        card_data         = EXCLUDED.card_data,
                        status            = 'pending',
                        formal_result     = NULL,
                        diag_result       = NULL,
                        icd_check_result  = NULL,
                        ignored           = FALSE,
                        broken            = FALSE,
                        stacktrace        = NULL,
                        organization_id   = EXCLUDED.organization_id,
                        pushed_at         = now()
                    RETURNING id::text
                    """,
                    {"guid": card_guid, "data": card_data, "org_id": organization_id},
                )
                row = await cur.fetchone()
            row_id: str = row["id"]
            logger.info("💾 done_cards UPSERT_PENDING OK id=%s guid=%s", row_id, card_guid)
            return row_id
        except Exception:
            logger.exception("💾 done_cards UPSERT_PENDING FAILED guid=%s", card_guid)
            raise

    async def get_priem(self, card_guid: str) -> dict | None:
        """Return the stored "Прием" block for a card, or None if no row matches.

        Matching is case-insensitive: pipeline rows store the guid as sent
        by 1C while pushed rows store it lowercased.
        """
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT card_data -> 'Прием' AS priem FROM done_cards "
                "WHERE lower(card_guid) = lower(%(guid)s) AND card_data IS NOT NULL",
                {"guid": card_guid},
            )
            row = await cur.fetchone()
        return row["priem"] if row else None

    async def replace_priem(self, *, card_guid: str, priem: str) -> bool:
        """Replace the "Прием" block of card_data with the fresh 1C one.

        The block is overwritten as a whole — stale keys inside it do not
        survive; the rest of card_data is untouched. Matching is
        case-insensitive (see :meth:`get_priem`).

        Returns True if a row was updated.
        """
        try:
            async with self._pool.connection() as conn:
                cur = await conn.execute(
                    """
                    UPDATE done_cards
                    SET card_data = jsonb_set(card_data, '{Прием}', %(priem)s::jsonb)
                    WHERE lower(card_guid) = lower(%(guid)s)
                      AND card_data IS NOT NULL
                    RETURNING id::text
                    """,
                    {"guid": card_guid, "priem": priem},
                )
                row = await cur.fetchone()
            if row:
                logger.info("💾 done_cards REPLACE_PRIEM OK id=%s guid=%s", row["id"], card_guid)
            return row is not None
        except Exception:
            logger.exception("💾 done_cards REPLACE_PRIEM FAILED guid=%s", card_guid)
            raise

    async def get_pending(self, organization_id: str | None = None) -> list[dict]:
        """Return card_guid + card_data for pending rows in an organization."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT card_guid, card_data FROM done_cards "
                "WHERE status = 'pending' "
                "AND organization_id IS NOT DISTINCT FROM %(org_id)s",
                {"org_id": organization_id},
            )
            rows = await cur.fetchall()
        logger.info("💾 done_cards loaded %d pending card(s) for org_id=%s", len(rows), organization_id)
        return rows

    async def get_done_guids(self, organization_id: str | None = None) -> set[str]:
        """Return non-null card GUIDs with a terminal (done) status for an organization.

        Pending rows (freshly pushed, not yet audited) are excluded: their
        guid must not count as "already handled", or the nightly pipeline's
        always-on dedup would skip them forever.
        """
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT card_guid FROM done_cards "
                "WHERE card_guid IS NOT NULL "
                "AND status = 'done' "
                "AND organization_id IS NOT DISTINCT FROM %(org_id)s",
                {"org_id": organization_id},
            )
            rows = await cur.fetchall()
        guids = {row["card_guid"] for row in rows}
        logger.info("💾 done_cards loaded %d done guid(s) for org_id=%s", len(guids), organization_id)
        return guids
