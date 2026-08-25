"""Integration tests for the demo-doctor stamp inside POST /visits/push.

Hits the real configured Postgres through FastAPI's TestClient, same fixture
pattern as tests/test_cards_push_api.py. The stamp itself is unit-tested in
tests/test_demo_doctors.py — what is checked here is the wiring: the switch
reaches the route, the organization gate holds, and a re-pushed card keeps the
doctor it was given.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from api import demo_doctors
from api.app import create_app
from storage.api_keys_storage import ApiKeysStorage
from storage.done_cards_storage import DoneCardsStorage
from storage.organizations_storage import OrganizationsStorage

pytestmark = pytest.mark.integration

DEMO_CODES = {d["code"] for d in demo_doctors.load_doctors()}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
async def key_for_both_orgs():
    async with OrganizationsStorage() as organizations:
        orgs = [await organizations.get_id_by_name("Alenka"),
                await organizations.get_id_by_name("MDS")]
    raw_key = f"medkard_test_{uuid.uuid4().hex}"
    async with ApiKeysStorage() as api_keys:
        key_id = await api_keys.create_key("pytest-demo-doctors", raw_key, orgs)
    yield raw_key
    async with ApiKeysStorage() as api_keys:
        await api_keys.revoke_key(key_id)


@pytest.fixture
def stamp_alenka(monkeypatch):
    monkeypatch.setenv("DEMO_DOCTOR_STAMP_ORG", "Alenka")
    demo_doctors.load_doctors.cache_clear()
    yield
    demo_doctors.load_doctors.cache_clear()


def _card(guid: str, *, doctor_code: str | None = None) -> dict:
    priem = {"GUID": guid, "DATE": "03.03.2045"}
    if doctor_code is not None:
        priem["Врач_код"] = doctor_code
        priem["Врач"] = "Настоящий врач"
    return {"Прием": priem, "Пациент": {"CODE": "Т-000004"}, "Диагнозы": []}


def _push(client: TestClient, key: str, card: dict, org: str) -> None:
    response = client.post(f"/visits/push?org={org}", json=card,
                           headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200, response.text


def _priem(guid: str) -> dict:
    async def _read():
        async with DoneCardsStorage() as storage:
            return await storage.get_priem(guid)
    # Тот же цикл, что у остальных тестов (как в tests/test_cards_push_api.py):
    # собственный цикл закрылся бы вместе с общим пулом из storage/base.py и
    # уронил следующие файлы сьюта.
    return asyncio.get_event_loop().run_until_complete(_read())


def _cleanup(guid: str) -> None:
    async def _delete():
        async with DoneCardsStorage() as storage:
            async with storage._pool.connection() as conn:
                await conn.execute(
                    "DELETE FROM done_cards WHERE lower(card_guid) = lower(%(g)s)",
                    {"g": guid})
    asyncio.get_event_loop().run_until_complete(_delete())


def test_pushed_card_arrives_with_a_doctor(client, key_for_both_orgs, stamp_alenka):
    guid = str(uuid.uuid4())
    try:
        _push(client, key_for_both_orgs, _card(guid), "Alenka")
        assert _priem(guid)["Врач_код"] in DEMO_CODES
    finally:
        _cleanup(guid)


def test_other_clinic_is_not_stamped(client, key_for_both_orgs, stamp_alenka):
    guid = str(uuid.uuid4())
    try:
        _push(client, key_for_both_orgs, _card(guid), "MDS")
        assert "Врач_код" not in _priem(guid)
    finally:
        _cleanup(guid)


def test_switch_off_means_no_stamp(client, key_for_both_orgs, monkeypatch):
    monkeypatch.delenv("DEMO_DOCTOR_STAMP_ORG", raising=False)
    guid = str(uuid.uuid4())
    try:
        _push(client, key_for_both_orgs, _card(guid), "Alenka")
        assert "Врач_код" not in _priem(guid)
    finally:
        _cleanup(guid)


def test_repush_keeps_the_same_doctor(client, key_for_both_orgs, stamp_alenka):
    # upsert_pending rewrites card_data whole, so without the carry-over the
    # doctor would be re-drawn on every push from 1C.
    guid = str(uuid.uuid4())
    try:
        _push(client, key_for_both_orgs, _card(guid), "Alenka")
        first = _priem(guid)["Врач_код"]
        for _ in range(5):
            _push(client, key_for_both_orgs, _card(guid), "Alenka")
            assert _priem(guid)["Врач_код"] == first
    finally:
        _cleanup(guid)


def test_a_doctor_sent_by_1c_wins(client, key_for_both_orgs, stamp_alenka):
    guid = str(uuid.uuid4())
    try:
        _push(client, key_for_both_orgs, _card(guid, doctor_code="1701"), "Alenka")
        assert _priem(guid)["Врач_код"] == "1701"
    finally:
        _cleanup(guid)
