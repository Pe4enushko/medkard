"""Stand-only integration coverage for replaying broken cards."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest

from audit.filters import CardFilter
from audit.pipeline import AuditPipeline
from storage.done_cards_storage import DoneCardsStorage

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _visit(guid: str) -> dict:
    return {
        "Прием": {"GUID": guid},
        "Пациент": {},
        "Диагнозы": [],
        "Услуги": [],
    }


async def _seed_broken(guid: str, organization_id: str | None = None) -> None:
    async with DoneCardsStorage() as storage:
        await storage.upsert_broken(
            card_guid=guid,
            card_data=json.dumps(_visit(guid), ensure_ascii=False),
            stacktrace="Traceback (most recent call last):\nValueError: seeded",
            started_at=datetime.now(timezone.utc),
            organization_id=organization_id,
        )


async def _cleanup(*guids: str) -> None:
    async with DoneCardsStorage() as storage, storage._pool.connection() as conn:
        await conn.execute(
            "DELETE FROM done_cards WHERE card_guid = ANY(%(guids)s)",
            {"guids": list(guids)},
        )


async def test_replay_clears_broken_and_persists_results() -> None:
    """Runs a real audit and therefore belongs on the configured stand."""
    guid = f"pytest-fix-broken-{uuid.uuid4()}"
    try:
        await _seed_broken(guid)
        async with AuditPipeline(org_id=None, card_filter=CardFilter([])) as pipeline:
            await pipeline.run_batched([_visit(guid)], num_batches=1, done_guids=set())

        async with DoneCardsStorage() as storage, storage._pool.connection() as conn:
            states = await storage.get_states_for_guids({guid})
            cur = await conn.execute(
                "SELECT formal_result FROM done_cards WHERE card_guid = %(guid)s",
                {"guid": guid},
            )
            row = await cur.fetchone()

        assert states[guid]["broken"] is False
        assert row["formal_result"] is not None
    finally:
        await _cleanup(guid)


async def test_normal_dedup_keeps_a_broken_card_frozen() -> None:
    guid = f"pytest-fix-dedup-{uuid.uuid4()}"
    try:
        await _seed_broken(guid)
        async with AuditPipeline(org_id=None, card_filter=CardFilter([])) as pipeline:
            pairs = await pipeline.run_batched([_visit(guid)], num_batches=1)

        assert pairs == []
        async with DoneCardsStorage() as storage:
            assert (await storage.get_states_for_guids({guid}))[guid]["broken"] is True
    finally:
        await _cleanup(guid)


async def test_repeat_failure_keeps_broken_with_fresh_stacktrace(monkeypatch) -> None:
    guid = f"pytest-fix-repeat-{uuid.uuid4()}"
    try:
        await _seed_broken(guid)
        async with AuditPipeline(org_id=None, card_filter=CardFilter([])) as pipeline:

            async def fail(_visit_payload):
                raise RuntimeError("fresh replay failure")

            monkeypatch.setattr(pipeline, "_audit_visit", fail)
            await pipeline.run_batched([_visit(guid)], num_batches=1, done_guids=set())

        async with DoneCardsStorage() as storage:
            state = (await storage.get_states_for_guids({guid}))[guid]
        assert state["broken"] is True
        assert "RuntimeError: fresh replay failure" in state["stacktrace"]
        assert "ValueError: seeded" not in state["stacktrace"]
    finally:
        await _cleanup(guid)


async def test_filtered_card_moves_to_ignored() -> None:
    guid = f"pytest-fix-filtered-{uuid.uuid4()}"

    class SkipEverything:
        def should_skip(self, _visit_payload) -> bool:
            return True

    try:
        await _seed_broken(guid)
        async with AuditPipeline(
            org_id=None,
            card_filter=CardFilter([SkipEverything()]),
        ) as pipeline:
            await pipeline.run_batched([_visit(guid)], num_batches=1, done_guids=set())

        async with DoneCardsStorage() as storage:
            state = (await storage.get_states_for_guids({guid}))[guid]
        assert state["ignored"] is True
        assert state["broken"] is False
    finally:
        await _cleanup(guid)
