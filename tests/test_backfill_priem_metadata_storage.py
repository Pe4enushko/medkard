"""Integration tests for the storage pair behind
scripts/operator/backfill-priem-metadata.py. Seeds its own rows and deletes
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


class _CardsWriter(BaseStorage):
    async def insert_card(self, guid: str, card: dict, org_id: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "INSERT INTO done_cards (card_guid, card_data, status, organization_id)"
                " VALUES (%(guid)s, %(data)s::jsonb, 'done', %(org)s::uuid)",
                {"guid": guid, "org": org_id, "data": json.dumps(card, ensure_ascii=False)},
            )

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


@pytest.fixture
async def seeded():
    async with OrganizationsStorage() as organizations:
        org_id = await organizations.get_id_by_name("MDS")
    with_doctor, without_doctor = str(uuid.uuid4()), str(uuid.uuid4())
    async with _CardsWriter() as writer:
        await writer.insert_card(with_doctor, {
            "Прием": {"GUID": with_doctor, "DATE": "03.03.2045"},
            "Врач": {"SPECIALIZATION": "Терапевт"},
            "Пациент": {"CODE": "Т-1"}, "Диагнозы": [{"КодМКБ": "J06.9"}],
        }, org_id)
        await writer.insert_card(without_doctor, {
            "Прием": {"GUID": without_doctor, "DATE": "03.03.2045"}, "Пациент": {"CODE": "Т-2"},
        }, org_id)
    yield {"with_doctor": with_doctor, "without_doctor": without_doctor}
    async with _CardsWriter() as writer:
        await writer.delete_cards([with_doctor, without_doctor])


async def test_get_visit_metadata_returns_both_blocks(seeded):
    async with DoneCardsStorage() as storage:
        meta = await storage.get_visit_metadata(seeded["with_doctor"])
    assert meta["Прием"]["DATE"] == "03.03.2045"
    assert meta["Врач"] == {"SPECIALIZATION": "Терапевт"}


async def test_get_visit_metadata_missing_doctor_block_is_none(seeded):
    async with DoneCardsStorage() as storage:
        meta = await storage.get_visit_metadata(seeded["without_doctor"])
    assert meta["Врач"] is None


async def test_get_visit_metadata_unknown_guid_is_none():
    async with DoneCardsStorage() as storage:
        assert await storage.get_visit_metadata(str(uuid.uuid4())) is None


async def test_replace_both_blocks_leaves_the_rest(seeded):
    guid = seeded["with_doctor"]
    priem = {"GUID": guid, "DATE": "03.03.2045", "Врач": "Иванов", "Врач_код": "0a99"}
    async with DoneCardsStorage() as storage:
        assert await storage.replace_visit_metadata(
            card_guid=guid, priem=json.dumps(priem, ensure_ascii=False),
            doctor=json.dumps({"SPECIALIZATION": "Педиатр"}, ensure_ascii=False))
    async with _CardsWriter() as writer:
        card = await writer.read_card(guid)
    assert card["Прием"] == priem
    assert card["Врач"] == {"SPECIALIZATION": "Педиатр"}
    assert card["Диагнозы"] == [{"КодМКБ": "J06.9"}]


async def test_replace_without_doctor_keeps_the_stored_block(seeded):
    guid = seeded["with_doctor"]
    async with DoneCardsStorage() as storage:
        await storage.replace_visit_metadata(
            card_guid=guid, priem=json.dumps({"GUID": guid, "DATE": "04.03.2045"}), doctor=None)
    async with _CardsWriter() as writer:
        card = await writer.read_card(guid)
    assert card["Прием"]["DATE"] == "04.03.2045"
    assert card["Врач"] == {"SPECIALIZATION": "Терапевт"}


async def test_replace_adds_doctor_block_where_there_was_none(seeded):
    guid = seeded["without_doctor"]
    async with DoneCardsStorage() as storage:
        await storage.replace_visit_metadata(
            card_guid=guid, priem=json.dumps({"GUID": guid, "DATE": "03.03.2045"}),
            doctor=json.dumps({"SPECIALIZATION": "Педиатр"}, ensure_ascii=False))
    async with _CardsWriter() as writer:
        card = await writer.read_card(guid)
    assert card["Врач"] == {"SPECIALIZATION": "Педиатр"}
