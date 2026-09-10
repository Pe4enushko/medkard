"""Integration tests for the two storage methods behind
scripts/hacks/backfill-alenka-doctors.py. Seeds its own rows and deletes
them afterwards — same shape as tests/test_demo_doctors_storage.py."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from storage.base import BaseStorage
from storage.done_cards_storage import DoneCardsStorage
from storage.organizations_storage import OrganizationsStorage

pytestmark = pytest.mark.integration


class _CardsWriter(BaseStorage):
    async def insert_card(self, guid: str, card: dict[str, Any] | None, org_id: str) -> str:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO done_cards (card_guid, card_data, status, organization_id)"
                " VALUES (%(guid)s, %(data)s::jsonb, 'done', %(org)s::uuid) RETURNING id::text",
                {"guid": guid, "org": org_id,
                 "data": None if card is None else json.dumps(card, ensure_ascii=False)},
            )
            return (await cur.fetchone())["id"]

    async def read_card(self, guid: str) -> dict | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT card_data FROM done_cards WHERE card_guid = %(g)s", {"g": guid})
            row = await cur.fetchone()
        return row["card_data"] if row else None

    async def delete_cards(self, guids: list[str]) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "DELETE FROM done_cards WHERE card_guid = ANY(%(guids)s)", {"guids": guids})


def _alenka(guid: str) -> dict:
    return {"Прием": {"GUID": guid, "DATE": "03.03.2045"},
            "Врач": {"GUID": "0a99", "FIO": "Правкина И. Г.", "SPECIALIZATION": "Педиатр"},
            "Пациент": {"CODE": "Т-1"}}


def _ours(guid: str) -> dict:
    return {"Прием": {"GUID": guid, "DATE": "03.03.2045", "Врач": "Врач 00012", "Врач_код": "00012"},
            "Врач": {"SPECIALIZATION": "Невролог"}, "Пациент": {"CODE": "Т-2"}}


@pytest.fixture
async def seeded():
    async with OrganizationsStorage() as organizations:
        org_id = await organizations.get_id_by_name("MDS")
    alenka, ours, no_doctor, no_data = (str(uuid.uuid4()) for _ in range(4))
    async with _CardsWriter() as writer:
        ids = {
            "alenka": await writer.insert_card(alenka, _alenka(alenka), org_id),
            "ours": await writer.insert_card(ours, _ours(ours), org_id),
            "no_doctor": await writer.insert_card(no_doctor, {"Прием": {"GUID": no_doctor}}, org_id),
            "no_data": await writer.insert_card(no_data, None, org_id),
        }
    yield {"ids": ids, "alenka": alenka, "ours": ours}
    async with _CardsWriter() as writer:
        await writer.delete_cards([alenka, ours, no_doctor, no_data])


async def test_lists_only_cards_with_doctor_on_top(seeded):
    async with DoneCardsStorage() as storage:
        rows = await storage.list_cards_with_top_level_doctor(limit=0, after_id="")
    listed = {row["id"] for row in rows}
    assert seeded["ids"]["alenka"] in listed
    assert seeded["ids"]["ours"] not in listed
    assert seeded["ids"]["no_doctor"] not in listed
    assert seeded["ids"]["no_data"] not in listed


async def test_listed_row_carries_card_data(seeded):
    async with DoneCardsStorage() as storage:
        rows = await storage.list_cards_with_top_level_doctor(limit=0, after_id="")
    row = next(r for r in rows if r["id"] == seeded["ids"]["alenka"])
    assert row["card_data"]["Врач"]["FIO"] == "Правкина И. Г."


async def test_cursor_skips_rows_up_to_after_id(seeded):
    async with DoneCardsStorage() as storage:
        rows = await storage.list_cards_with_top_level_doctor(
            limit=0, after_id=seeded["ids"]["alenka"])
    assert seeded["ids"]["alenka"] not in {row["id"] for row in rows}


async def test_set_card_data_rewrites_the_card(seeded):
    new_card = {"Прием": {"GUID": seeded["alenka"], "Врач": "X", "Врач_код": "0a99"},
                "Врач": {"SPECIALIZATION": "Педиатр"}}
    async with DoneCardsStorage() as storage:
        written = await storage.set_card_data(
            card_id=seeded["ids"]["alenka"], card_json=json.dumps(new_card, ensure_ascii=False))
    assert written == 1
    async with _CardsWriter() as writer:
        assert await writer.read_card(seeded["alenka"]) == new_card
