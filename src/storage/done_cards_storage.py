"""
DoneCardsStorage — async psycopg3 interface for the *done_cards* table.

Upserts one row per card identified by card_guid. If a row with that guid
already exists it is updated in place; otherwise a new row is inserted.
Cards with no GUID are always inserted as new rows.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any

import psycopg

from .base import BaseStorage
from .base import reopen_shared_pool
from .models.result import DiagnosisResult, FormalStructureResult, IcdCodingIssue

logger = logging.getLogger(__name__)


def _formal_json(formal: FormalStructureResult) -> str:
    return json.dumps(
        [{"flag": f.flag, "issue": f.issue, "source": f.source, "comment": f.comment} for f in formal.findings],
        ensure_ascii=False,
    )


def _icd_check_json(issues: list[IcdCodingIssue] | None) -> str | None:
    """None → SQL NULL: чекер не отработал, а не «замечаний нет»."""
    if issues is None:
        return None
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
                        **({"aspect": iss.aspect} if iss.aspect is not None else {}),
                        "sources": [
                            {
                                "doc_title": s.doc_title,
                                "section": s.section,
                                "cite": s.cite,
                                "chunk_id": s.chunk_id,
                                "chunk_index": s.chunk_index,
                            }
                            for s in iss.sources
                        ],
                    }
                    for iss in dr.issues
                ],
                "guideline_sources": [
                    {
                        "file_id": source.file_id,
                        "doc_title": source.doc_title,
                        "sections": [
                            {
                                "section": section.section,
                                "chunk_indices": section.chunk_indices,
                                "cited": section.cited,
                            }
                            for section in source.sections
                        ],
                    }
                    for source in dr.guideline_sources
                ],
                "errors": dr.errors,
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
        icd_check: list[IcdCodingIssue] | None,
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
            return await self._upsert_once(
                card_data=card_data,
                formal_json=formal_json,
                diag_json=diag_json,
                icd_check_json=icd_check_json,
                token_count=token_count,
                time_ms=time_ms,
                started_at=started_at,
                finished_at=finished_at,
                card_guid=card_guid,
                organization_id=organization_id,
            )
        except psycopg.OperationalError:
            logger.warning(
                "💾 done_cards UPSERT retrying with fresh pool guid=%s",
                card_guid,
            )
            self._pool = await reopen_shared_pool()
            return await self._upsert_once(
                card_data=card_data,
                formal_json=formal_json,
                diag_json=diag_json,
                icd_check_json=icd_check_json,
                token_count=token_count,
                time_ms=time_ms,
                started_at=started_at,
                finished_at=finished_at,
                card_guid=card_guid,
                organization_id=organization_id,
            )

    async def _upsert_once(
        self,
        *,
        card_data: str,
        formal_json: str,
        diag_json: str,
        icd_check_json: str | None,
        token_count: int,
        time_ms: int,
        started_at: datetime,
        finished_at: datetime,
        card_guid: str | None = None,
        organization_id: str | None = None,
    ) -> str:
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
            return await self._upsert_broken_once(
                card_data=card_data,
                stacktrace=stacktrace,
                started_at=started_at,
                card_guid=card_guid,
                organization_id=organization_id,
            )
        except psycopg.OperationalError:
            logger.warning(
                "💾 done_cards UPSERT_BROKEN retrying with fresh pool guid=%s",
                card_guid,
            )
            self._pool = await reopen_shared_pool()
            return await self._upsert_broken_once(
                card_data=card_data,
                stacktrace=stacktrace,
                started_at=started_at,
                card_guid=card_guid,
                organization_id=organization_id,
            )

    async def _upsert_broken_once(
        self,
        *,
        card_data: str,
        stacktrace: str,
        started_at: datetime,
        card_guid: str | None = None,
        organization_id: str | None = None,
    ) -> str:
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

    async def get_broken(self, organization_id: str | None = None) -> list[dict]:
        """Return replayable broken rows, optionally scoped to one organization.

        Unlike ``get_pending`` and ``get_done_guids``, ``None`` means every
        organization. Rows without card data cannot be replayed, and rows
        without a GUID cannot be matched back after replay, so both are skipped.
        """
        query = (
            "SELECT card_guid, card_data, organization_id::text AS organization_id "
            "FROM done_cards "
            "WHERE broken = TRUE "
            "AND card_data IS NOT NULL "
            "AND card_guid IS NOT NULL"
        )
        params: dict[str, Any] = {}
        if organization_id is not None:
            query += " AND organization_id = %(org_id)s::uuid"
            params["org_id"] = organization_id

        async with self._pool.connection() as conn:
            cur = await conn.execute(query, params)
            rows = await cur.fetchall()

        logger.info(
            "💾 done_cards loaded %d broken card(s) for org_id=%s",
            len(rows),
            organization_id if organization_id is not None else "<all>",
        )
        return [dict(row) for row in rows]

    async def get_states_for_guids(self, guids: set[str]) -> dict[str, dict]:
        """Return broken/ignored flags and stacktraces for existing GUIDs."""
        if not guids:
            return {}

        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT card_guid, broken, ignored, stacktrace FROM done_cards "
                "WHERE card_guid = ANY(%(guids)s)",
                {"guids": list(guids)},
            )
            rows = await cur.fetchall()

        return {
            row["card_guid"]: {
                "broken": bool(row["broken"]),
                "ignored": bool(row["ignored"]),
                "stacktrace": row["stacktrace"],
            }
            for row in rows
        }

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

    async def list_cards_without_doctor(
        self, *, organization_id: str, limit: int = 0, since: date | None = None
    ) -> list[str]:
        """Guids of one org's cards whose Прием carries no doctor code.

        Serves the temporary demo-doctor backfill (scripts/hacks/backfill-demo-doctors.py,
        api/demo_doctors.py) and goes away with it. Audit status is deliberately
        not a filter: the crutch stamps every visit, pending ones included, the
        same way the push route does.

        An empty string counts as no doctor — 1C sends both that and no key at
        all for "no doctor". Cards with no card_data and cards with no Прием
        block have nothing to stamp and drop out. limit=0 means no cap.

        *since* bounds the visit date (medkard_visit_date reads both 1C's
        DD.MM.YYYY and ISO, migration 026). A card whose date is unparseable is
        on no date at all, so a boundary drops it — without one it stays in.
        """
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT card_guid
                FROM done_cards
                WHERE organization_id = %(org_id)s::uuid
                  AND card_data IS NOT NULL
                  AND jsonb_typeof(card_data -> 'Прием') = 'object'
                  AND COALESCE(card_data -> 'Прием' ->> 'Врач_код', '') = ''
                  AND (%(since)s::date IS NULL
                       OR medkard_visit_date(card_data -> 'Прием' ->> 'DATE')
                          >= %(since)s::date)
                ORDER BY card_guid
                LIMIT NULLIF(%(limit)s, 0)
                """,
                {"org_id": organization_id, "limit": limit, "since": since},
            )
            return [row["card_guid"] for row in await cur.fetchall()]

    async def list_cards_with_doctor_codes(
        self, *, organization_id: str, codes: list[str], limit: int = 0,
        since: date | None = None
    ) -> list[str]:
        """Guids of one org's cards stamped with one of *codes*.

        The revert half of the demo-doctor crutch: it works from the list of
        made-up codes, so a card carrying a real doctor from 1C is never
        touched by it. *since* bounds the visit date the same way it does for
        list_cards_without_doctor, so a bounded stamp can be taken back with
        the same boundary.
        """
        if not codes:
            return []
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT card_guid
                FROM done_cards
                WHERE organization_id = %(org_id)s::uuid
                  AND card_data IS NOT NULL
                  AND card_data -> 'Прием' ->> 'Врач_код' = ANY(%(codes)s)
                  AND (%(since)s::date IS NULL
                       OR medkard_visit_date(card_data -> 'Прием' ->> 'DATE')
                          >= %(since)s::date)
                ORDER BY card_guid
                LIMIT NULLIF(%(limit)s, 0)
                """,
                {"org_id": organization_id, "codes": codes, "limit": limit,
                 "since": since},
            )
            return [row["card_guid"] for row in await cur.fetchall()]

    async def set_doctor_on_cards(
        self, *, card_guids: list[str], code: str, name: str
    ) -> int:
        """Write one doctor into the Прием block of many cards. Returns rows written.

        Merges into the block rather than replacing it (replace_priem), so the
        1C data in it survives; one statement per doctor keeps a backfill over
        the whole clinic to a handful of round-trips.
        """
        if not card_guids:
            return 0
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE done_cards
                SET card_data = jsonb_set(
                        card_data, '{Прием}',
                        (card_data -> 'Прием')
                          || jsonb_build_object('Врач', %(name)s::text,
                                                'Врач_код', %(code)s::text))
                WHERE card_guid = ANY(%(guids)s)
                  AND jsonb_typeof(card_data -> 'Прием') = 'object'
                """,
                {"guids": card_guids, "code": code, "name": name},
            )
            return cur.rowcount

    async def clear_doctor_on_cards(self, *, card_guids: list[str]) -> int:
        """Drop Врач and Врач_код from the Прием block of many cards.

        Both keys are empty on Alenka's live data, so removing them puts a card
        back exactly as 1C sent it. Callers pick the cards by demo code
        (list_cards_with_doctor_codes) — this method itself does not check.
        """
        if not card_guids:
            return 0
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE done_cards
                SET card_data = jsonb_set(
                        card_data, '{Прием}',
                        (card_data -> 'Прием') - 'Врач' - 'Врач_код')
                WHERE card_guid = ANY(%(guids)s)
                  AND jsonb_typeof(card_data -> 'Прием') = 'object'
                """,
                {"guids": card_guids},
            )
            return cur.rowcount

    async def list_audited_by_visit_date(
        self, *, organization_id: str, visit_date: date
    ) -> list[dict]:
        """Audited cards of one org on one visit date, with per-kind issue counts.

        Selection matches the pull API's (reporting/api_formatter.py): audited
        cards only — ignored and broken ones carry no results, and pending ones
        have not been through the pipeline yet. The date comes from
        medkard_visit_date(Прием.DATE), which reads both 1C's DD.MM.YYYY and ISO
        (migration 026); a card whose date is unparseable is simply not on any
        date and drops out.

        Counts, not the results themselves: callers rank cards by how much the
        audit found, and a date's worth of full result arrays is a lot of JSON
        to move for a number. A NULL result column counts as zero — "the checker
        never ran" and "it found nothing" are different, but neither puts a line
        in a report.
        """
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT card_guid,
                       card_data -> 'Прием' ->> 'Врач_код'          AS doctor_code,
                       COALESCE(jsonb_array_length(formal_result), 0)    AS formal_n,
                       COALESCE(jsonb_array_length(diag_result), 0)      AS diag_n,
                       COALESCE(jsonb_array_length(icd_check_result), 0) AS icd_n
                FROM done_cards
                WHERE organization_id = %(org_id)s::uuid
                  AND ignored = FALSE
                  AND broken = FALSE
                  AND status = 'done'
                  AND medkard_visit_date(card_data -> 'Прием' ->> 'DATE') = %(date)s::date
                ORDER BY card_guid
                """,
                {"org_id": organization_id, "date": visit_date},
            )
            return [dict(row) for row in await cur.fetchall()]
