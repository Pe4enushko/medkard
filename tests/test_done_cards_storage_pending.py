"""
Integration tests for DoneCardsStorage.upsert_pending / get_pending, and for
status transitions on the existing upsert*/get_done_guids methods. Hits the
real configured Postgres.
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
from storage.models.result import FormalStructureResult


async def _cleanup(guid: str) -> None:
    async with DoneCardsStorage() as storage:
        async with storage._pool.connection() as conn:
            await conn.execute("DELETE FROM done_cards WHERE card_guid = %(g)s", {"g": guid})


@pytest.mark.asyncio
async def test_upsert_pending_creates_row_with_pending_status():
    guid = f"pytest-push-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            row_id = await storage.upsert_pending(
                card_guid=guid,
                card_data=json.dumps({"Прием": {"GUID": guid}}, ensure_ascii=False),
                organization_id=None,
            )
            assert row_id

            async with storage._pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT status, card_data, formal_result FROM done_cards WHERE card_guid = %(g)s",
                    {"g": guid},
                )
                row = await cur.fetchone()
        assert row["status"] == "pending"
        assert row["formal_result"] is None
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_upsert_pending_on_existing_done_row_wipes_results_and_flags():
    guid = f"pytest-push-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            now = datetime.now(timezone.utc)
            await storage.upsert(
                card_guid=guid,
                card_data=json.dumps({"Прием": {"GUID": guid}, "v": 1}, ensure_ascii=False),
                formal=FormalStructureResult(findings=[]),
                diagnosis=[],
                icd_check=[],
                token_count=10,
                time_ms=5,
                started_at=now,
                finished_at=now,
                organization_id=None,
            )

            async with storage._pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT status FROM done_cards WHERE card_guid = %(g)s", {"g": guid}
                )
                assert (await cur.fetchone())["status"] == "done"

            await storage.upsert_pending(
                card_guid=guid,
                card_data=json.dumps({"Прием": {"GUID": guid}, "v": 2}, ensure_ascii=False),
                organization_id=None,
            )

            async with storage._pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT status, card_data, formal_result, diag_result, icd_check_result, "
                    "ignored, broken, stacktrace FROM done_cards WHERE card_guid = %(g)s",
                    {"g": guid},
                )
                row = await cur.fetchone()
        assert row["status"] == "pending"
        assert row["card_data"]["v"] == 2
        assert row["formal_result"] is None
        assert row["diag_result"] is None
        assert row["icd_check_result"] is None
        assert row["ignored"] is False
        assert row["broken"] is False
        assert row["stacktrace"] is None
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_get_pending_returns_only_pending_rows_for_org():
    org_id = None  # NULL-scoped org bucket, matches existing tests' style for org-less rows
    pending_guid = f"pytest-pending-{uuid.uuid4()}"
    done_guid = f"pytest-done-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            await storage.upsert_pending(
                card_guid=pending_guid,
                card_data=json.dumps({"Прием": {"GUID": pending_guid}}, ensure_ascii=False),
                organization_id=org_id,
            )
            now = datetime.now(timezone.utc)
            await storage.upsert(
                card_guid=done_guid,
                card_data=json.dumps({"Прием": {"GUID": done_guid}}, ensure_ascii=False),
                formal=FormalStructureResult(findings=[]),
                diagnosis=[],
                icd_check=[],
                token_count=1,
                time_ms=1,
                started_at=now,
                finished_at=now,
                organization_id=org_id,
            )

            pending_rows = await storage.get_pending(organization_id=org_id)
        pending_guids = {r["card_guid"] for r in pending_rows}
        assert pending_guid in pending_guids
        assert done_guid not in pending_guids
    finally:
        await _cleanup(pending_guid)
        await _cleanup(done_guid)


@pytest.mark.asyncio
async def test_get_done_guids_excludes_pending_rows():
    guid = f"pytest-pending-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            await storage.upsert_pending(
                card_guid=guid,
                card_data=json.dumps({"Прием": {"GUID": guid}}, ensure_ascii=False),
                organization_id=None,
            )
            done_guids = await storage.get_done_guids(organization_id=None)
        assert guid not in done_guids
    finally:
        await _cleanup(guid)
