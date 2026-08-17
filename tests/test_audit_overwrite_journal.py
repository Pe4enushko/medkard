"""
Integration tests for the audit-overwrite journal (migration 027): the trigger
that counts every push over an existing card and archives the ones that destroy
audit results. Hits the real configured Postgres.

The trigger fires on done -> pending updates, which in production means
DoneCardsStorage.upsert_pending — so these tests drive the storage layer rather
than issuing raw UPDATEs, exercising the same path POST /visits/push takes.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

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


async def _row(storage: DoneCardsStorage, guid: str) -> dict | None:
    async with storage._pool.connection() as conn:
        cur = await conn.execute(
            "SELECT status, push_count, formal_result FROM done_cards WHERE card_guid = %(g)s",
            {"g": guid},
        )
        return await cur.fetchone()


async def _journal_rows(storage: DoneCardsStorage, guid: str) -> list[dict]:
    async with storage._pool.connection() as conn:
        cur = await conn.execute(
            "SELECT * FROM audit_overwrite_journal WHERE card_guid = %(g)s ORDER BY overwritten_at",
            {"g": guid},
        )
        return await cur.fetchall()


async def _cleanup(guid: str) -> None:
    async with DoneCardsStorage() as storage:
        async with storage._pool.connection() as conn:
            await conn.execute("DELETE FROM done_cards WHERE card_guid = %(g)s", {"g": guid})
            await conn.execute("DELETE FROM audit_overwrite_journal WHERE card_guid = %(g)s", {"g": guid})


@pytest.fixture
async def alenka_org_id() -> str:
    async with OrganizationsStorage() as organizations:
        return await organizations.get_id_by_name("Alenka")


@pytest.mark.asyncio
async def test_push_over_audited_card_journals_old_results(alenka_org_id: str):
    """The headline case: a push destroys audit output, so it is archived."""
    guid = f"pytest-journal-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            await _audit_card(storage, guid, alenka_org_id)

            await storage.upsert_pending(
                card_guid=guid,
                card_data=_card(guid, version=2),
                organization_id=alenka_org_id,
            )

            journal = await _journal_rows(storage, guid)
            assert len(journal) == 1, "expected exactly one journal row for one destructive push"

            entry = journal[0]
            # The journal holds what was LOST, not what replaced it.
            assert entry["formal_result"] is not None
            assert entry["formal_result"][0]["flag"] == "missing_section"
            assert entry["card_data"]["v"] == 1, "journal must keep the OLD card, not the new one"
            assert entry["token_count"] == 1234
            assert entry["time_ms"] == 5678
            assert str(entry["organization_id"]) == alenka_org_id

            # The live row still got wiped — journalling does not block the reset.
            row = await _row(storage, guid)
            assert row["status"] == "pending"
            assert row["formal_result"] is None
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_push_over_pending_card_counts_but_does_not_journal(alenka_org_id: str):
    """A re-push over a not-yet-audited card loses nothing: counter only."""
    guid = f"pytest-journal-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            await storage.upsert_pending(
                card_guid=guid, card_data=_card(guid), organization_id=alenka_org_id
            )
            await storage.upsert_pending(
                card_guid=guid, card_data=_card(guid, version=2), organization_id=alenka_org_id
            )

            assert await _journal_rows(storage, guid) == []
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_push_over_ignored_card_does_not_journal(alenka_org_id: str):
    """'ignored' rows carry no audit output, so nothing is lost."""
    guid = f"pytest-journal-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            await storage.upsert_ignored(
                card_guid=guid, card_data=_card(guid), organization_id=alenka_org_id
            )
            await storage.upsert_pending(
                card_guid=guid, card_data=_card(guid, version=2), organization_id=alenka_org_id
            )

            assert await _journal_rows(storage, guid) == []
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_reaudit_of_done_card_does_not_journal(alenka_org_id: str):
    """done -> done replaces results with fresh ones; nothing is lost."""
    guid = f"pytest-journal-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            await _audit_card(storage, guid, alenka_org_id)
            await _audit_card(storage, guid, alenka_org_id)

            assert await _journal_rows(storage, guid) == []
            row = await _row(storage, guid)
            assert row["push_count"] == 0, "a re-audit is not a push"
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_push_count_increments_on_every_overwrite(alenka_org_id: str):
    """push_count counts all overwrites — destructive or not."""
    guid = f"pytest-journal-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            # First push creates the row: an INSERT, not an overwrite.
            await storage.upsert_pending(
                card_guid=guid, card_data=_card(guid), organization_id=alenka_org_id
            )
            row = await _row(storage, guid)
            assert row["push_count"] == 0, "creating a card is not overwriting one"

            for version in (2, 3, 4):
                await storage.upsert_pending(
                    card_guid=guid, card_data=_card(guid, version), organization_id=alenka_org_id
                )

            row = await _row(storage, guid)
            assert row["push_count"] == 3
            assert await _journal_rows(storage, guid) == [], "no results existed to lose"
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_journal_disabled_for_org_skips_journal_but_still_counts(alenka_org_id: str):
    """The per-org flag gates the journal only — statistics keep working."""
    guid = f"pytest-journal-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            async with storage._pool.connection() as conn:
                await conn.execute(
                    "UPDATE organizations SET audit_overwrite_journal_enabled = FALSE "
                    "WHERE id = %(id)s",
                    {"id": alenka_org_id},
                )

            await _audit_card(storage, guid, alenka_org_id)
            await storage.upsert_pending(
                card_guid=guid, card_data=_card(guid, version=2), organization_id=alenka_org_id
            )

            assert await _journal_rows(storage, guid) == [], "flag off: nothing archived"
            row = await _row(storage, guid)
            assert row["push_count"] == 1, "flag off must not stop counting"
            assert row["formal_result"] is None, "the wipe still happens"
    finally:
        async with DoneCardsStorage() as storage:
            async with storage._pool.connection() as conn:
                await conn.execute(
                    "UPDATE organizations SET audit_overwrite_journal_enabled = TRUE "
                    "WHERE id = %(id)s",
                    {"id": alenka_org_id},
                )
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_metrics_view_splits_destructive_and_empty_overwrites(alenka_org_id: str):
    """The view's three counters agree: total = with_results + no_results."""
    audited_guid = f"pytest-journal-{uuid.uuid4()}"
    empty_guid = f"pytest-journal-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            async with storage._pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT overwrites_with_results, overwrites_total, overwrites_no_results "
                    "FROM audit_overwrite_metrics WHERE organization_name = 'Alenka'"
                )
                before = await cur.fetchone()

            # One destructive overwrite...
            await _audit_card(storage, audited_guid, alenka_org_id)
            await storage.upsert_pending(
                card_guid=audited_guid, card_data=_card(audited_guid, 2), organization_id=alenka_org_id
            )
            # ...and one that costs nothing.
            await storage.upsert_pending(
                card_guid=empty_guid, card_data=_card(empty_guid), organization_id=alenka_org_id
            )
            await storage.upsert_pending(
                card_guid=empty_guid, card_data=_card(empty_guid, 2), organization_id=alenka_org_id
            )

            async with storage._pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT overwrites_with_results, overwrites_total, overwrites_no_results, journal_enabled "
                    "FROM audit_overwrite_metrics WHERE organization_name = 'Alenka'"
                )
                after = await cur.fetchone()

        assert after["journal_enabled"] is True
        assert after["overwrites_with_results"] - before["overwrites_with_results"] == 1
        assert after["overwrites_total"] - before["overwrites_total"] == 2
        assert after["overwrites_no_results"] - before["overwrites_no_results"] == 1
        assert (
            after["overwrites_total"]
            == after["overwrites_with_results"] + after["overwrites_no_results"]
        )
    finally:
        await _cleanup(audited_guid)
        await _cleanup(empty_guid)
