"""
Integration tests for push_log (migration 027): the trigger that logs every
push over an existing done_cards row, dated, with whether it overrode a
completed audit result. Hits the real configured Postgres.

The trigger fires on done -> pending updates, which in production means
DoneCardsStorage.upsert_pending — so these tests drive the storage layer
rather than issuing raw UPDATEs, exercising the same path POST /visits/push
takes.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest

from storage.done_cards_storage import DoneCardsStorage
from storage.models.result import FormalFinding, FormalStructureResult
from storage.organizations_storage import OrganizationsStorage


def _card(guid: str, version: int = 1) -> str:
    return json.dumps(
        {"Прием": {"GUID": guid, "DATE": "01.08.2026"}, "Пациент": {"ФИО": "Тест Тестов"}, "v": version},
        ensure_ascii=False,
    )


def _formal_with_findings() -> FormalStructureResult:
    return FormalStructureResult(
        findings=[
            FormalFinding(
                flag="missing_section",
                issue="Отсутствует раздел «Жалобы»",
                source="ДанныеОсмотра",
                comment="pytest fixture",
            )
        ]
    )


async def _audit_card(storage: DoneCardsStorage, guid: str, org_id: str | None) -> None:
    """Put a card into the audited (done, results present) state."""
    now = datetime.now(timezone.utc)
    await storage.upsert(
        card_guid=guid,
        card_data=_card(guid),
        formal=_formal_with_findings(),
        diagnosis=[],
        icd_check=[],
        token_count=1234,
        time_ms=5678,
        started_at=now,
        finished_at=now,
        organization_id=org_id,
    )


async def _push_log_rows(storage: DoneCardsStorage, guid: str) -> list[dict]:
    async with storage._pool.connection() as conn:
        cur = await conn.execute(
            "SELECT * FROM push_log WHERE card_guid = %(g)s ORDER BY pushed_at",
            {"g": guid},
        )
        return await cur.fetchall()


async def _cleanup(guid: str) -> None:
    async with DoneCardsStorage() as storage:
        async with storage._pool.connection() as conn:
            await conn.execute("DELETE FROM done_cards WHERE card_guid = %(g)s", {"g": guid})
            await conn.execute("DELETE FROM push_log WHERE card_guid = %(g)s", {"g": guid})


@pytest.fixture
async def alenka_org_id() -> str:
    async with OrganizationsStorage() as organizations:
        return await organizations.get_id_by_name("Alenka")


@pytest.fixture
async def mds_org_id() -> str:
    async with OrganizationsStorage() as organizations:
        return await organizations.get_id_by_name("MDS")


@pytest.mark.asyncio
async def test_push_over_audited_card_logs_overrode_audit_true(alenka_org_id: str):
    """The headline case: a push destroys audit output — logged as overriding."""
    guid = f"pytest-pushlog-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            await _audit_card(storage, guid, alenka_org_id)

            await storage.upsert_pending(
                card_guid=guid, card_data=_card(guid, version=2), organization_id=alenka_org_id
            )

            rows = await _push_log_rows(storage, guid)
            assert len(rows) == 1, "expected exactly one push_log row for one push"
            assert rows[0]["overrode_audit"] is True
            assert str(rows[0]["organization_id"]) == alenka_org_id
            assert rows[0]["card_data"] is None, "payload column is reserved, not yet populated"
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_push_over_pending_card_logs_overrode_audit_false(alenka_org_id: str):
    """A re-push over a not-yet-audited card loses nothing: logged as non-overriding."""
    guid = f"pytest-pushlog-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            await storage.upsert_pending(
                card_guid=guid, card_data=_card(guid), organization_id=alenka_org_id
            )
            await storage.upsert_pending(
                card_guid=guid, card_data=_card(guid, version=2), organization_id=alenka_org_id
            )

            rows = await _push_log_rows(storage, guid)
            assert len(rows) == 1, "the first upsert_pending is an INSERT, not an overwrite"
            assert rows[0]["overrode_audit"] is False
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_push_over_ignored_card_logs_overrode_audit_false(alenka_org_id: str):
    """'ignored' rows carry no audit output, so nothing is lost."""
    guid = f"pytest-pushlog-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            await storage.upsert_ignored(
                card_guid=guid, card_data=_card(guid), organization_id=alenka_org_id
            )
            await storage.upsert_pending(
                card_guid=guid, card_data=_card(guid, version=2), organization_id=alenka_org_id
            )

            rows = await _push_log_rows(storage, guid)
            assert len(rows) == 1
            assert rows[0]["overrode_audit"] is False
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_push_over_broken_card_logs_overrode_audit_false(alenka_org_id: str):
    """'broken' rows carry only a stacktrace, no audit output, so nothing is lost."""
    guid = f"pytest-pushlog-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            await storage.upsert_broken(
                card_guid=guid,
                card_data=_card(guid),
                stacktrace="Traceback (most recent call last):\n  pytest fixture",
                started_at=datetime.now(timezone.utc),
                organization_id=alenka_org_id,
            )
            await storage.upsert_pending(
                card_guid=guid, card_data=_card(guid, version=2), organization_id=alenka_org_id
            )

            rows = await _push_log_rows(storage, guid)
            assert len(rows) == 1
            assert rows[0]["overrode_audit"] is False
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_reaudit_of_done_card_does_not_log(alenka_org_id: str):
    """done -> done replaces results with fresh ones; not a push, not logged."""
    guid = f"pytest-pushlog-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            await _audit_card(storage, guid, alenka_org_id)
            await _audit_card(storage, guid, alenka_org_id)

            assert await _push_log_rows(storage, guid) == []
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_multiple_pushes_in_one_day_aggregate_in_metrics_view(alenka_org_id: str):
    """push_metrics_by_date sums same-day pushes into one row per org/date."""
    overriding_guid = f"pytest-pushlog-{uuid.uuid4()}"
    quiet_guid = f"pytest-pushlog-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            async with storage._pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT pushes_total, pushes_overrode_audit, pushes_no_override "
                    "FROM push_metrics_by_date "
                    "WHERE organization_name = 'Alenka' AND push_date = current_date"
                )
                before = await cur.fetchone() or {
                    "pushes_total": 0, "pushes_overrode_audit": 0, "pushes_no_override": 0
                }

            # One overriding push...
            await _audit_card(storage, overriding_guid, alenka_org_id)
            await storage.upsert_pending(
                card_guid=overriding_guid, card_data=_card(overriding_guid, 2),
                organization_id=alenka_org_id,
            )
            # ...and one quiet push.
            await storage.upsert_pending(
                card_guid=quiet_guid, card_data=_card(quiet_guid), organization_id=alenka_org_id
            )
            await storage.upsert_pending(
                card_guid=quiet_guid, card_data=_card(quiet_guid, 2), organization_id=alenka_org_id
            )

            async with storage._pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT pushes_total, pushes_overrode_audit, pushes_no_override "
                    "FROM push_metrics_by_date "
                    "WHERE organization_name = 'Alenka' AND push_date = current_date"
                )
                after = await cur.fetchone()

        assert after["pushes_total"] - before["pushes_total"] == 2
        assert after["pushes_overrode_audit"] - before["pushes_overrode_audit"] == 1
        assert after["pushes_no_override"] - before["pushes_no_override"] == 1
        assert after["pushes_total"] == after["pushes_overrode_audit"] + after["pushes_no_override"]
    finally:
        await _cleanup(overriding_guid)
        await _cleanup(quiet_guid)


@pytest.mark.asyncio
async def test_replace_priem_on_pending_card_does_not_log(alenka_org_id: str):
    """replace_priem (scripts/backfill-priem.py's write path) never touches
    status or pushed_at, so it must not fire the push_log trigger even when
    it happens to run against a row that is already status='pending' — the
    exact scenario Finding 1 of the final review flagged as a phantom-push
    risk if the trigger only checked NEW.status = 'pending'.
    """
    guid = f"pytest-pushlog-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            # The trigger is BEFORE UPDATE, so the very first upsert_pending
            # (an INSERT — no prior row) never fires it. Push a second time
            # over that pending row so there is one genuine push_log row on
            # the books, then exercise replace_priem against that same
            # already-pending row.
            await storage.upsert_pending(
                card_guid=guid, card_data=_card(guid), organization_id=alenka_org_id
            )
            await storage.upsert_pending(
                card_guid=guid, card_data=_card(guid, version=2), organization_id=alenka_org_id
            )

            rows_after_push = await _push_log_rows(storage, guid)
            assert len(rows_after_push) == 1, "sanity: the second push (pending -> pending) logged exactly one row"

            updated = await storage.replace_priem(
                card_guid=guid,
                priem=json.dumps({"GUID": guid, "DATE": "02.08.2026"}, ensure_ascii=False),
            )
            assert updated is True, "sanity: replace_priem found and updated the row"

            rows_after_replace = await _push_log_rows(storage, guid)
            assert len(rows_after_replace) == 1, (
                "replace_priem on an already-pending row must not add a new "
                "push_log row — it is not a push"
            )
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_metrics_view_does_not_mix_organizations(alenka_org_id: str, mds_org_id: str):
    """A push logged under one org must not inflate another org's same-day count."""
    alenka_guid = f"pytest-pushlog-{uuid.uuid4()}"
    mds_guid = f"pytest-pushlog-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            async with storage._pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT pushes_total FROM push_metrics_by_date "
                    "WHERE organization_name = 'MDS' AND push_date = current_date"
                )
                mds_before = await cur.fetchone()
                mds_before_total = mds_before["pushes_total"] if mds_before else 0

            # Push under Alenka only.
            await storage.upsert_pending(
                card_guid=alenka_guid, card_data=_card(alenka_guid), organization_id=alenka_org_id
            )
            await storage.upsert_pending(
                card_guid=alenka_guid, card_data=_card(alenka_guid, 2), organization_id=alenka_org_id
            )

            async with storage._pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT pushes_total FROM push_metrics_by_date "
                    "WHERE organization_name = 'MDS' AND push_date = current_date"
                )
                mds_after = await cur.fetchone()
                mds_after_total = mds_after["pushes_total"] if mds_after else 0

            assert mds_after_total == mds_before_total, "Alenka's push must not show up under MDS"

            # Sanity: an MDS push, meanwhile, does show up under MDS.
            await storage.upsert_pending(
                card_guid=mds_guid, card_data=_card(mds_guid), organization_id=mds_org_id
            )
            await storage.upsert_pending(
                card_guid=mds_guid, card_data=_card(mds_guid, 2), organization_id=mds_org_id
            )

            async with storage._pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT pushes_total FROM push_metrics_by_date "
                    "WHERE organization_name = 'MDS' AND push_date = current_date"
                )
                mds_final = await cur.fetchone()

            assert mds_final["pushes_total"] - mds_before_total == 1
    finally:
        await _cleanup(alenka_guid)
        await _cleanup(mds_guid)
