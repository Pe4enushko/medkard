"""
Integration tests for POST /cards/push — hits the real configured Postgres
via FastAPI's TestClient, same fixture pattern as tests/test_cards_api.py.
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

from api.app import create_app
from storage.api_keys_storage import ApiKeysStorage
from storage.done_cards_storage import DoneCardsStorage
from storage.organizations_storage import OrganizationsStorage


def _unique_key() -> str:
    return f"medkard_test_{uuid.uuid4().hex}"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
async def alenka_org_id() -> str:
    async with OrganizationsStorage() as organizations:
        return await organizations.get_id_by_name("Alenka")


@pytest.fixture
async def test_key(alenka_org_id: str) -> str:
    raw_key = _unique_key()
    async with ApiKeysStorage() as api_keys:
        key_id = await api_keys.create_key("pytest-push", raw_key, [alenka_org_id])
    yield raw_key
    async with ApiKeysStorage() as api_keys:
        await api_keys.revoke_key(key_id)


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _cleanup(guid: str) -> None:
    async def _delete():
        async with DoneCardsStorage() as storage:
            async with storage._pool.connection() as conn:
                await conn.execute("DELETE FROM done_cards WHERE card_guid = %(g)s", {"g": guid})

    asyncio.get_event_loop().run_until_complete(_delete())


def _get_pending(organization_id: str | None) -> list[dict]:
    async def _fetch():
        async with DoneCardsStorage() as storage:
            return await storage.get_pending(organization_id=organization_id)

    return asyncio.get_event_loop().run_until_complete(_fetch())


def test_push_missing_key_is_rejected(client: TestClient):
    guid = str(uuid.uuid4())
    resp = client.post("/cards/push?org=Alenka", json={"Прием": {"GUID": guid}})
    assert resp.status_code in (401, 403)


def test_push_without_guid_is_422(client: TestClient, test_key: str):
    resp = client.post("/cards/push?org=Alenka", json={"Прием": {}}, headers=_auth(test_key))
    assert resp.status_code == 422


def test_push_new_card_creates_pending_row(client: TestClient, test_key: str, alenka_org_id: str):
    guid = str(uuid.uuid4())
    try:
        resp = client.post(
            "/cards/push?org=Alenka",
            json={"Прием": {"GUID": guid}, "Пациент": {"ФИО": "Тест Тестов"}},
            headers=_auth(test_key),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["card_guid"] == guid.lower()
        assert body["status"] == "pending"

        rows = _get_pending(organization_id=alenka_org_id)
        matching = [r for r in rows if r["card_guid"] == guid.lower()]
        assert len(matching) == 1
    finally:
        _cleanup(guid)


def test_push_updates_existing_card_and_resets_to_pending(client: TestClient, test_key: str, alenka_org_id: str):
    guid = str(uuid.uuid4())
    try:
        first = client.post(
            "/cards/push?org=Alenka",
            json={"Прием": {"GUID": guid}, "v": 1},
            headers=_auth(test_key),
        )
        assert first.status_code == 200

        second = client.post(
            "/cards/push?org=Alenka",
            json={"Прием": {"GUID": guid}, "v": 2},
            headers=_auth(test_key),
        )
        assert second.status_code == 200
        assert second.json()["status"] == "pending"

        rows = _get_pending(organization_id=alenka_org_id)
        matching = [r for r in rows if r["card_guid"] == guid.lower()]
        assert len(matching) == 1
        assert matching[0]["card_data"]["v"] == 2
    finally:
        _cleanup(guid)
