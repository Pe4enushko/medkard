"""Integration test for DoneCardsStorage.list_audited_by_visit_date.

Seeds its own done_cards rows on a far-future date (2044-02-02) so stand data
can't collide, and deletes them afterwards — same shape as
tests/test_visits_doctor_filter.py.
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

FIXTURE_DATE = date(2044, 2, 2)          # DD.MM.YYYY в карте: 02.02.2044


class _CardsWriter(BaseStorage):
    async def insert_card(self, guid: str, card: dict[str, Any], org_id: str, *,
                          formal: str, diag: str, icd: str | None,
                          status: str = "done", ignored: bool = False,
                          broken: bool = False) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "INSERT INTO done_cards (card_guid, card_data, formal_result, diag_result,"
                " icd_check_result, ignored, broken, status, organization_id)"
                " VALUES (%(guid)s, %(data)s::jsonb, %(formal)s::jsonb, %(diag)s::jsonb,"
                " %(icd)s::jsonb, %(ignored)s, %(broken)s, %(status)s, %(org)s::uuid)",
                {"guid": guid, "data": json.dumps(card, ensure_ascii=False),
                 "formal": formal, "diag": diag, "icd": icd,
                 "ignored": ignored, "broken": broken, "status": status, "org": org_id},
            )

    async def delete_cards(self, guids: list[str]) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "DELETE FROM done_cards WHERE card_guid = ANY(%(guids)s)", {"guids": guids})


def _card(guid: str, doctor_code: str | None, date_raw: str = "02.02.2044") -> dict[str, Any]:
    priem: dict[str, Any] = {"GUID": guid, "DATE": date_raw}
    if doctor_code is not None:
        priem["Врач_код"] = doctor_code
    return {"Прием": priem, "Пациент": {"CODE": "Т-000002"}, "Диагнозы": []}


def _issues(n: int) -> str:
    return json.dumps([{"issue": f"замечание {i}"} for i in range(n)], ensure_ascii=False)


@pytest.fixture
async def org_id() -> str:
    async with OrganizationsStorage() as organizations:
        return await organizations.get_id_by_name("MDS")


@pytest.fixture
async def seeded(org_id: str):
    """Пять карт на фикстурную дату + шум, который выборка обязана отбросить."""
    plain = str(uuid.uuid4())          # без врача, 2 замечания
    rich = str(uuid.uuid4())           # без врача, 5 замечаний
    mine = str(uuid.uuid4())           # уже наш демо-врач
    other = str(uuid.uuid4())          # чужой врач
    ignored = str(uuid.uuid4())        # ignored — не аудирована
    pending = str(uuid.uuid4())        # ещё не аудирована
    other_day = str(uuid.uuid4())      # соседняя дата

    async with _CardsWriter() as writer:
        await writer.insert_card(plain, _card(plain, None), org_id,
                                 formal=_issues(1), diag=_issues(1), icd=_issues(0))
        await writer.insert_card(rich, _card(rich, ""), org_id,
                                 formal=_issues(2), diag=_issues(2), icd=_issues(1))
        await writer.insert_card(mine, _card(mine, "90001"), org_id,
                                 formal=_issues(0), diag=_issues(0), icd=None)
        await writer.insert_card(other, _card(other, "00012"), org_id,
                                 formal=_issues(3), diag=_issues(0), icd=_issues(0))
        await writer.insert_card(ignored, _card(ignored, None), org_id,
                                 formal="[]", diag="[]", icd=None, ignored=True)
        await writer.insert_card(pending, _card(pending, None), org_id,
                                 formal="[]", diag="[]", icd=None, status="pending")
        await writer.insert_card(other_day, _card(other_day, None, "03.02.2044"), org_id,
                                 formal="[]", diag="[]", icd=None)

    yield {"org_id": org_id, "plain": plain, "rich": rich, "mine": mine,
           "other": other, "ignored": ignored, "pending": pending, "other_day": other_day}

    async with _CardsWriter() as writer:
        await writer.delete_cards([plain, rich, mine, other, ignored, pending, other_day])


async def _fetch(org_id: str) -> dict[str, dict]:
    async with DoneCardsStorage() as storage:
        rows = await storage.list_audited_by_visit_date(
            organization_id=org_id, visit_date=FIXTURE_DATE)
    return {r["card_guid"]: r for r in rows}


async def test_returns_audited_cards_of_that_date(seeded):
    rows = await _fetch(seeded["org_id"])
    assert seeded["plain"] in rows and seeded["rich"] in rows


async def test_excludes_ignored_pending_and_other_dates(seeded):
    rows = await _fetch(seeded["org_id"])
    for key in ("ignored", "pending", "other_day"):
        assert seeded[key] not in rows, key


async def test_reports_the_doctor_code_each_card_carries(seeded):
    rows = await _fetch(seeded["org_id"])
    assert rows[seeded["mine"]]["doctor_code"] == "90001"
    assert rows[seeded["other"]]["doctor_code"] == "00012"
    assert rows[seeded["plain"]]["doctor_code"] is None
    assert rows[seeded["rich"]]["doctor_code"] == ""


async def test_counts_findings_per_result_kind(seeded):
    rows = await _fetch(seeded["org_id"])
    rich = rows[seeded["rich"]]
    assert (rich["formal_n"], rich["diag_n"], rich["icd_n"]) == (2, 2, 1)


async def test_counts_a_null_result_as_zero_findings(seeded):
    # icd_check_result IS NULL means "the checker never ran", not "no issues" —
    # either way there is nothing to show in a report.
    rows = await _fetch(seeded["org_id"])
    assert rows[seeded["mine"]]["icd_n"] == 0
