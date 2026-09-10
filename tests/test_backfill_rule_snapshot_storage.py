"""Integration tests for the two storage methods behind
scripts/hacks/backfill-rule-snapshot.py. Seeds its own rows and deletes
them afterwards — same shape as tests/test_demo_doctors_storage.py."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from storage.base import BaseStorage
from storage.done_cards_storage import DoneCardsStorage
from storage.organizations_storage import OrganizationsStorage

pytestmark = pytest.mark.integration

OLD = [{"flag": "ОТСУТСТВУЮТ_МЕТАДАННЫЕ_ВИЗИТА", "issue": "нет даты", "source": "274n", "comment": ""}]
NEW = [{**OLD[0], "rule_id": "visit_meta_required", "severity": "критичный", "source_ref": "274н",
        "expectation": "дата", "verified_at": "2026-08-19"}]


class _CardsWriter(BaseStorage):
    async def insert_card(self, guid: str, formal: list | None, org_id: str) -> str:
        card = {"Прием": {"GUID": guid, "DATE": "03.03.2045"}, "Пациент": {"CODE": "Т-1", "AGE": 40}}
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO done_cards (card_guid, card_data, formal_result, status, organization_id)"
                " VALUES (%(guid)s, %(data)s::jsonb, %(formal)s::jsonb, 'done', %(org)s::uuid)"
                " RETURNING id::text",
                {"guid": guid, "org": org_id, "data": json.dumps(card, ensure_ascii=False),
                 "formal": None if formal is None else json.dumps(formal, ensure_ascii=False)},
            )
            return (await cur.fetchone())["id"]

    async def read_formal(self, guid: str) -> list | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT formal_result FROM done_cards WHERE card_guid = %(g)s", {"g": guid})
            row = await cur.fetchone()
        return row["formal_result"] if row else None

    async def delete_cards(self, guids: list[str]) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "DELETE FROM done_cards WHERE card_guid = ANY(%(guids)s)", {"guids": guids})


@pytest.fixture
async def seeded():
    async with OrganizationsStorage() as organizations:
        org_id = await organizations.get_id_by_name("MDS")
    old, new, empty, none = (str(uuid.uuid4()) for _ in range(4))
    async with _CardsWriter() as writer:
        ids = {
            "old": await writer.insert_card(old, OLD, org_id),
            "new": await writer.insert_card(new, NEW, org_id),
            "empty": await writer.insert_card(empty, [], org_id),
            "none": await writer.insert_card(none, None, org_id),
        }
    yield {"ids": ids, "old": old}
    async with _CardsWriter() as writer:
        await writer.delete_cards([old, new, empty, none])


async def test_lists_only_cards_with_a_finding_without_snapshot(seeded):
    async with DoneCardsStorage() as storage:
        rows = await storage.list_formal_results_to_backfill(limit=0, after_id="")
    listed = {row["id"] for row in rows}
    assert seeded["ids"]["old"] in listed
    assert seeded["ids"]["new"] not in listed
    assert seeded["ids"]["empty"] not in listed
    assert seeded["ids"]["none"] not in listed


async def test_listed_row_carries_findings_and_patient(seeded):
    async with DoneCardsStorage() as storage:
        rows = await storage.list_formal_results_to_backfill(limit=0, after_id="")
    row = next(r for r in rows if r["id"] == seeded["ids"]["old"])
    assert row["formal_result"] == OLD
    assert row["patient"]["AGE"] == 40


async def test_set_formal_result_rewrites_the_column(seeded):
    async with DoneCardsStorage() as storage:
        written = await storage.set_formal_result(
            card_id=seeded["ids"]["old"], formal_json=json.dumps(NEW, ensure_ascii=False))
    assert written == 1
    async with _CardsWriter() as writer:
        assert await writer.read_formal(seeded["old"]) == NEW
