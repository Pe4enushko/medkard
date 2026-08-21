"""Integration tests for replay-related DoneCardsStorage methods."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest

from storage.done_cards_storage import DoneCardsStorage

pytestmark = pytest.mark.integration


async def _cleanup_ids(
    *, guids: tuple[str, ...] = (), row_ids: tuple[str, ...] = ()
) -> None:
    async with DoneCardsStorage() as storage, storage._pool.connection() as conn:
        if guids:
            await conn.execute(
                "DELETE FROM done_cards WHERE card_guid = ANY(%(guids)s)",
                {"guids": list(guids)},
            )
        if row_ids:
            await conn.execute(
                "DELETE FROM done_cards WHERE id = ANY(%(ids)s::uuid[])",
                {"ids": list(row_ids)},
            )


async def _seed_broken(guid: str, organization_id: str | None = None) -> None:
    async with DoneCardsStorage() as storage:
        await storage.upsert_broken(
            card_guid=guid,
            card_data=json.dumps({"Прием": {"GUID": guid}}, ensure_ascii=False),
            stacktrace="Traceback (most recent call last):\nValueError: boom",
            started_at=datetime.now(timezone.utc),
            organization_id=organization_id,
        )


async def test_get_broken_returns_only_replayable_rows() -> None:
    replayable = f"pytest-broken-{uuid.uuid4()}"
    no_data = f"pytest-broken-nodata-{uuid.uuid4()}"
    row_without_guid: str | None = None
    try:
        await _seed_broken(replayable)
        await _seed_broken(no_data)
        async with DoneCardsStorage() as storage:
            async with storage._pool.connection() as conn:
                await conn.execute(
                    "UPDATE done_cards SET card_data = NULL WHERE card_guid = %(guid)s",
                    {"guid": no_data},
                )
            row_without_guid = await storage.upsert_broken(
                card_guid=None,
                card_data=json.dumps({"Прием": {}}, ensure_ascii=False),
                stacktrace="ValueError: boom",
                started_at=datetime.now(timezone.utc),
            )
            rows = await storage.get_broken()

        by_guid = {row["card_guid"]: row for row in rows}
        assert by_guid[replayable]["card_data"]["Прием"]["GUID"] == replayable
        assert no_data not in by_guid
        assert None not in by_guid
    finally:
        await _cleanup_ids(
            guids=(replayable, no_data),
            row_ids=(row_without_guid,) if row_without_guid else (),
        )


async def test_get_broken_scopes_only_when_org_is_given() -> None:
    null_org_guid = f"pytest-broken-null-{uuid.uuid4()}"
    org_guid = f"pytest-broken-org-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage, storage._pool.connection() as conn:
            cur = await conn.execute("SELECT id::text FROM organizations LIMIT 1")
            org = await cur.fetchone()
        if org is None:
            pytest.skip("no organizations configured")

        org_id = org["id"]
        await _seed_broken(null_org_guid)
        await _seed_broken(org_guid, org_id)
        async with DoneCardsStorage() as storage:
            all_guids = {row["card_guid"] for row in await storage.get_broken()}
            scoped_guids = {
                row["card_guid"]
                for row in await storage.get_broken(organization_id=org_id)
            }

        assert {null_org_guid, org_guid} <= all_guids
        assert org_guid in scoped_guids
        assert null_org_guid not in scoped_guids
    finally:
        await _cleanup_ids(guids=(null_org_guid, org_guid))


async def test_get_states_for_guids_reports_flags_and_omits_unknown() -> None:
    guid = f"pytest-state-{uuid.uuid4()}"
    try:
        await _seed_broken(guid)
        async with DoneCardsStorage() as storage:
            states = await storage.get_states_for_guids({guid, "absent"})
            empty = await storage.get_states_for_guids(set())

        assert states[guid]["broken"] is True
        assert states[guid]["ignored"] is False
        assert states[guid]["stacktrace"].endswith("ValueError: boom")
        assert "absent" not in states
        assert empty == {}
    finally:
        await _cleanup_ids(guids=(guid,))
