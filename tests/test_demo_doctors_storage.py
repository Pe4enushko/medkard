"""Integration tests for the bulk doctor stamp behind scripts/hacks/backfill-demo-doctors.py.

Seeds its own done_cards rows on a far-future date (2045-03-03) so stand data
can't collide, and deletes them afterwards — same shape as
tests/test_seed_demo_doctor_storage.py.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import date
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

DEMO_CODES = ["90001", "90002"]


class _CardsWriter(BaseStorage):
    async def insert_card(self, guid: str, card: dict[str, Any] | None, org_id: str,
                          status: str = "done") -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "INSERT INTO done_cards (card_guid, card_data, formal_result, diag_result,"
                " status, organization_id)"
                " VALUES (%(guid)s, %(data)s::jsonb, '[]'::jsonb, '[]'::jsonb,"
                " %(status)s, %(org)s::uuid)",
                {"guid": guid, "org": org_id, "status": status,
                 "data": None if card is None else json.dumps(card, ensure_ascii=False)},
            )

    async def read_priem(self, guid: str) -> dict | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT card_data -> 'Прием' AS priem FROM done_cards WHERE card_guid = %(g)s",
                {"g": guid})
            row = await cur.fetchone()
        return row["priem"] if row else None

    async def delete_cards(self, guids: list[str]) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "DELETE FROM done_cards WHERE card_guid = ANY(%(guids)s)", {"guids": guids})


def _card(guid: str, doctor_code: str | None = None,
          date_raw: str = "03.03.2045") -> dict[str, Any]:
    priem: dict[str, Any] = {"GUID": guid, "DATE": date_raw}
    if doctor_code is not None:
        priem["Врач_код"] = doctor_code
        priem["Врач"] = f"Врач {doctor_code}"
    return {"Прием": priem, "Пациент": {"CODE": "Т-000003"}, "Диагнозы": []}


@pytest.fixture
async def org_id() -> str:
    async with OrganizationsStorage() as organizations:
        return await organizations.get_id_by_name("MDS")


@pytest.fixture
async def seeded(org_id: str):
    free = str(uuid.uuid4())        # без врача
    blank = str(uuid.uuid4())       # Врач_код есть, но пустой
    ours = str(uuid.uuid4())        # уже с демо-врачом
    foreign = str(uuid.uuid4())     # врач от 1С
    pending = str(uuid.uuid4())     # ещё не аудирована — бэкфилл берёт и такие
    no_data = str(uuid.uuid4())     # card_data IS NULL
    no_priem = str(uuid.uuid4())    # карта без блока Прием
    old_day = str(uuid.uuid4())     # без врача, но визит давний
    bad_date = str(uuid.uuid4())    # без врача, дата визита нечитаемая

    async with _CardsWriter() as writer:
        await writer.insert_card(free, _card(free), org_id)
        await writer.insert_card(blank, _card(blank, ""), org_id)
        await writer.insert_card(ours, _card(ours, "90002"), org_id)
        await writer.insert_card(foreign, _card(foreign, "00012"), org_id)
        await writer.insert_card(pending, _card(pending), org_id, status="pending")
        await writer.insert_card(no_data, None, org_id)
        await writer.insert_card(no_priem, {"Пациент": {"CODE": "Т-1"}}, org_id)
        await writer.insert_card(old_day, _card(old_day, date_raw="03.03.2035"), org_id)
        await writer.insert_card(bad_date, _card(bad_date, date_raw="позавчера"), org_id)

    guids = {"org_id": org_id, "free": free, "blank": blank, "ours": ours,
             "foreign": foreign, "pending": pending, "no_data": no_data,
             "no_priem": no_priem, "old_day": old_day, "bad_date": bad_date}
    yield guids

    async with _CardsWriter() as writer:
        await writer.delete_cards([v for k, v in guids.items() if k != "org_id"])


# ── выборка ──────────────────────────────────────────────────────────────────

async def test_lists_cards_without_a_doctor(seeded):
    async with DoneCardsStorage() as storage:
        guids = await storage.list_cards_without_doctor(organization_id=seeded["org_id"])
    assert seeded["free"] in guids
    assert seeded["blank"] in guids       # пустая строка — тот же «врача нет»
    assert seeded["pending"] in guids     # статус аудита бэкфилл не смотрит


async def test_skips_cards_that_already_have_a_doctor_or_have_no_priem(seeded):
    async with DoneCardsStorage() as storage:
        guids = await storage.list_cards_without_doctor(organization_id=seeded["org_id"])
    for key in ("ours", "foreign", "no_data", "no_priem"):
        assert seeded[key] not in guids, key


async def test_limit_caps_the_batch(seeded):
    async with DoneCardsStorage() as storage:
        guids = await storage.list_cards_without_doctor(
            organization_id=seeded["org_id"], limit=1)
    assert len(guids) == 1


async def test_lists_cards_carrying_our_codes_only(seeded):
    async with DoneCardsStorage() as storage:
        guids = await storage.list_cards_with_doctor_codes(
            organization_id=seeded["org_id"], codes=DEMO_CODES)
    assert seeded["ours"] in guids
    assert seeded["foreign"] not in guids     # чужого врача ревёрт не трогает
    assert seeded["free"] not in guids


# ── штамп и снятие ───────────────────────────────────────────────────────────

async def test_stamps_the_named_doctor(seeded):
    async with DoneCardsStorage() as storage:
        n = await storage.set_doctor_on_cards(
            card_guids=[seeded["free"]], code="90007", name="Врач 90007")
    assert n == 1
    async with _CardsWriter() as writer:
        priem = await writer.read_priem(seeded["free"])
    assert priem["Врач_код"] == "90007"
    assert priem["Врач"] == "Врач 90007"


async def test_stamp_keeps_the_rest_of_the_priem_block(seeded):
    async with DoneCardsStorage() as storage:
        await storage.set_doctor_on_cards(
            card_guids=[seeded["free"]], code="90007", name="Врач 90007")
    async with _CardsWriter() as writer:
        priem = await writer.read_priem(seeded["free"])
    assert priem["DATE"] == "03.03.2045"
    assert priem["GUID"] == seeded["free"]


async def test_clear_removes_both_keys(seeded):
    async with DoneCardsStorage() as storage:
        n = await storage.clear_doctor_on_cards(card_guids=[seeded["ours"]])
    assert n == 1
    async with _CardsWriter() as writer:
        priem = await writer.read_priem(seeded["ours"])
    assert "Врач_код" not in priem and "Врач" not in priem
    assert priem["DATE"] == "03.03.2045"


async def test_empty_batch_writes_nothing(seeded):
    async with DoneCardsStorage() as storage:
        assert await storage.set_doctor_on_cards(card_guids=[], code="9", name="Н") == 0
        assert await storage.clear_doctor_on_cards(card_guids=[]) == 0


# ── граница по дате визита ───────────────────────────────────────────────────

async def test_since_drops_visits_older_than_the_boundary(seeded):
    async with DoneCardsStorage() as storage:
        guids = await storage.list_cards_without_doctor(
            organization_id=seeded["org_id"], since=date(2045, 1, 1))
    assert seeded["free"] in guids
    assert seeded["old_day"] not in guids


async def test_card_with_an_unreadable_date_drops_out_under_a_boundary(seeded):
    # medkard_visit_date не разобрала дату — карта не лежит ни на одной дате,
    # и под границей её просто нет. Без границы она в выборке остаётся.
    async with DoneCardsStorage() as storage:
        bounded = await storage.list_cards_without_doctor(
            organization_id=seeded["org_id"], since=date(2045, 1, 1))
        unbounded = await storage.list_cards_without_doctor(
            organization_id=seeded["org_id"])
    assert seeded["bad_date"] not in bounded
    assert seeded["bad_date"] in unbounded


async def test_revert_honours_the_same_boundary(seeded):
    async with DoneCardsStorage() as storage:
        near = await storage.list_cards_with_doctor_codes(
            organization_id=seeded["org_id"], codes=DEMO_CODES, since=date(2045, 1, 1))
        far = await storage.list_cards_with_doctor_codes(
            organization_id=seeded["org_id"], codes=DEMO_CODES, since=date(2046, 1, 1))
    assert seeded["ours"] in near
    assert seeded["ours"] not in far
