"""
Integration tests for GET /stats/storage — hits the real configured Postgres
via FastAPI's TestClient, same fixture pattern as tests/test_cards_push_api.py.
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
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
from storage.models.result import FormalFinding, FormalStructureResult
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
        key_id = await api_keys.create_key("pytest-stats", raw_key, [alenka_org_id])
    yield raw_key
    async with ApiKeysStorage() as api_keys:
        await api_keys.revoke_key(key_id)


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _card(guid: str, version: int = 1) -> str:
    return json.dumps(
        {"Прием": {"GUID": guid, "DATE": "01.08.2026"}, "Пациент": {"ФИО": "Тест Тестов"}, "v": version},
        ensure_ascii=False,
    )


def _cleanup(guid: str) -> None:
    async def _delete():
        async with DoneCardsStorage() as storage:
            async with storage._pool.connection() as conn:
                await conn.execute("DELETE FROM done_cards WHERE card_guid = %(g)s", {"g": guid})
                await conn.execute("DELETE FROM push_log WHERE card_guid = %(g)s", {"g": guid})

    asyncio.get_event_loop().run_until_complete(_delete())


def _overwrite_an_audited_card(guid: str, org_id: str) -> None:
    """Audit a card then push over it, producing one push_log row."""

    async def _run():
        now = datetime.now(timezone.utc)
        async with DoneCardsStorage() as storage:
            await storage.upsert(
                card_guid=guid,
                card_data=_card(guid),
                formal=FormalStructureResult(
                    findings=[FormalFinding(flag="missing_section", issue="Отсутствует раздел «Жалобы»")]
                ),
                diagnosis=[],
                icd_check=[],
                token_count=1234,
                time_ms=5678,
                started_at=now,
                finished_at=now,
                organization_id=org_id,
            )
            await storage.upsert_pending(
                card_guid=guid, card_data=_card(guid, version=2), organization_id=org_id
            )

    asyncio.get_event_loop().run_until_complete(_run())


def test_storage_missing_key_is_rejected(client: TestClient):
    resp = client.get("/stats/storage?org=Alenka")
    assert resp.status_code in (401, 403)


def test_storage_unknown_org_is_404(client: TestClient, test_key: str):
    resp = client.get(
        f"/stats/storage?org=NoSuchOrg{uuid.uuid4().hex}", headers=_auth(test_key)
    )
    assert resp.status_code == 404


def test_storage_returns_kilobytes_for_org(client: TestClient, test_key: str):
    resp = client.get("/stats/storage?org=Alenka", headers=_auth(test_key))
    assert resp.status_code == 200

    body = resp.json()
    assert body["organization"] == "Alenka"
    assert body["done_cards_kb"] > 0, "Alenka has stored cards, so its size must be non-zero"
    assert body["push_log_kb"] >= 0
    assert body["total_kb"] == pytest.approx(
        body["done_cards_kb"] + body["push_log_kb"], abs=0.01
    )


def test_storage_org_slug_is_case_insensitive(client: TestClient, test_key: str):
    """?org= resolves the slug case-insensitively, as the visits routes do."""
    lower = client.get("/stats/storage?org=alenka", headers=_auth(test_key))
    assert lower.status_code == 200
    assert lower.json()["organization"] == "Alenka"


def test_push_log_growth_shows_up_in_storage_stats(client: TestClient, test_key: str, alenka_org_id: str):
    """A push over an audited card increases push_log's reported size."""
    guid = f"pytest-stats-{uuid.uuid4()}"
    try:
        before = client.get("/stats/storage?org=Alenka", headers=_auth(test_key)).json()

        _overwrite_an_audited_card(guid, alenka_org_id)

        after = client.get("/stats/storage?org=Alenka", headers=_auth(test_key)).json()
        assert after["push_log_kb"] > before["push_log_kb"]
    finally:
        _cleanup(guid)
