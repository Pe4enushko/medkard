"""Integration tests for the doctor_code filter and /visits/doctors.

Seeds its own done_cards rows for MDS on a far-future date (2044-01-01) so
existing stand data can't collide, and deletes them afterwards.
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

from reporting.api_formatter import ApiFormatter
from storage.base import BaseStorage
from storage.organizations_storage import OrganizationsStorage

FIXTURE_DATE = date(2044, 1, 1)          # DD.MM.YYYY в карте: 01.01.2044
DOC_A = "00001"
DOC_B = "00002"


class _CardsWriter(BaseStorage):
    async def insert_card(self, guid: str, card_data: dict[str, Any], org_id: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "INSERT INTO done_cards (card_guid, card_data, formal_result, diag_result,"
                " icd_check_result, ignored, broken, status, organization_id)"
                " VALUES (%(guid)s, %(data)s::jsonb, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb,"
                " FALSE, FALSE, 'done', %(org)s::uuid)",
                {"guid": guid, "data": json.dumps(card_data, ensure_ascii=False), "org": org_id},
            )

    async def delete_cards(self, guids: list[str]) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "DELETE FROM done_cards WHERE card_guid = ANY(%(guids)s)", {"guids": guids}
            )


def _card(doctor_code: str | None, doctor_name: str | None) -> dict[str, Any]:
    priem: dict[str, Any] = {"GUID": str(uuid.uuid4()), "DATE": "01.01.2044"}
    if doctor_name is not None:
        priem["Врач"] = doctor_name
    if doctor_code is not None:
        priem["Врач_код"] = doctor_code
    return {
        "Прием": priem,
        "Врач": {"SPECIALIZATION": "Невролог"},
        "Пациент": {"CODE": "Т-000001", "GENDER": "Мужской", "AGE": 40},
        "Диагнозы": [],
    }


@pytest.fixture
async def mds_org_id() -> str:
    async with OrganizationsStorage() as organizations:
        return await organizations.get_id_by_name("MDS")


@pytest.fixture
async def seeded_cards(mds_org_id: str):
    """2 карты врача 00001, 1 карта 00002, 1 без Врач_код — все на 2044-01-01."""
    cards = [
        _card(DOC_A, "Иванов Иван Иванович"),
        _card(DOC_A, "Иванов Иван Иванович"),
        _card(DOC_B, "Петрова Анна Сергеевна"),
        _card(None, None),
    ]
    guids = [c["Прием"]["GUID"].lower() for c in cards]
    async with _CardsWriter() as writer:
        for guid, card in zip(guids, cards):
            await writer.insert_card(guid, card, mds_org_id)
    yield mds_org_id
    async with _CardsWriter() as writer:
        await writer.delete_cards(guids)


async def test_check_counts_only_that_doctor(seeded_cards: str):
    async with ApiFormatter() as formatter:
        assert await formatter.check(FIXTURE_DATE, seeded_cards, DOC_A) == 2
        assert await formatter.check(FIXTURE_DATE, seeded_cards, DOC_B) == 1
        assert await formatter.check(FIXTURE_DATE, seeded_cards, "99999") == 0


async def test_check_without_filter_counts_all(seeded_cards: str):
    async with ApiFormatter() as formatter:
        assert await formatter.check(FIXTURE_DATE, seeded_cards) == 4


async def test_make_xlsx_filters_rows(seeded_cards: str):
    import io
    import openpyxl

    async with ApiFormatter() as formatter:
        content = await formatter.make_xlsx(FIXTURE_DATE, seeded_cards, DOC_A)
    ws = openpyxl.load_workbook(io.BytesIO(content)).active
    assert ws.max_row - 1 == 2  # минус заголовок
