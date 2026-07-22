"""
Integration tests for GET /cards/check_updates — real Postgres via TestClient with
an api key scoped to Alenka + MDS. Requests carry a `since` bounded to this test's
own rows, so they never scan the whole table. Covers what distinguishes this
endpoint from /cards/export: every status is returned (pending/ignored/broken
included), the `since` boundary is inclusive, and a bare call falls back to a
one-week window rather than all history.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import psycopg
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from api.app import create_app  # noqa: E402
from storage.api_keys_storage import ApiKeysStorage  # noqa: E402
from storage.organizations_storage import OrganizationsStorage  # noqa: E402

_CARD = '{"Прием": {"DATE": "01.07.2026"}}'

_INSERT = (
    "INSERT INTO done_cards "
    "(card_guid, card_data, status, ignored, broken, organization_id) "
    "VALUES (%(g)s, %(d)s::jsonb, %(s)s, %(i)s, %(b)s, %(o)s)"
)


def _conninfo() -> str:
    return (
        f"host={os.environ['POSTGRES_HOST']} port={os.environ.get('POSTGRES_PORT','5432')} "
        f"dbname={os.environ['POSTGRES_DB']} user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
async def alenka_org_id() -> str:
    async with OrganizationsStorage() as orgs:
        return await orgs.get_id_by_name("Alenka")


@pytest.fixture
async def mds_org_id() -> str:
    async with OrganizationsStorage() as orgs:
        return await orgs.get_id_by_name("MDS")


@pytest.fixture
async def test_key(alenka_org_id: str, mds_org_id: str):
    raw = f"medkard_test_{uuid.uuid4().hex}"
    async with ApiKeysStorage() as api_keys:
        key_id = await api_keys.create_key("pytest-check-updates", raw, [alenka_org_id, mds_org_id])
    yield raw
    async with ApiKeysStorage() as api_keys:
        await api_keys.revoke_key(key_id)


@pytest.fixture
async def seeded(alenka_org_id: str):
    """One Alenka card per status, seeded after a captured cutoff so requests
    bounded by that cutoff see only these rows."""
    tag = uuid.uuid4().hex[:8]
    guids = {
        "done": f"pytest-cuapi-{tag}-done",
        "pending": f"pytest-cuapi-{tag}-pend",
        "ignored": f"pytest-cuapi-{tag}-ign",
        "broken": f"pytest-cuapi-{tag}-brk",
    }
    rows = [
        (guids["done"], "done", False, False),
        (guids["pending"], "pending", False, False),
        (guids["ignored"], "done", True, False),
        (guids["broken"], "done", False, True),
    ]
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        cur = await conn.execute("SELECT now()")
        cutoff = (await cur.fetchone())[0]
        for guid, status, ignored, broken in rows:
            await conn.execute(
                _INSERT,
                {"g": guid, "d": _CARD, "s": status, "i": ignored,
                 "b": broken, "o": alenka_org_id},
            )
    yield guids, cutoff.isoformat()
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        await conn.execute(
            "DELETE FROM done_cards WHERE card_guid = ANY(%(gs)s)",
            {"gs": list(guids.values())},
        )


def test_check_updates_requires_key(client: TestClient):
    resp = client.get("/cards/check_updates", params={"org": "Alenka"})
    assert resp.status_code in (401, 403)


def test_check_updates_requires_org(client: TestClient, test_key: str):
    resp = client.get("/cards/check_updates", headers=_auth(test_key))
    assert resp.status_code == 422


def test_check_updates_unknown_org_404(client: TestClient, test_key: str):
    resp = client.get(
        "/cards/check_updates",
        params={"org": f"nope-{uuid.uuid4().hex[:6]}"},
        headers=_auth(test_key),
    )
    assert resp.status_code == 404


def test_check_updates_returns_every_status(client, test_key, seeded):
    """The point of the endpoint: unlike export, pending/ignored/broken cards are
    all returned, with their raw card_data."""
    guids, cutoff = seeded
    resp = client.get(
        "/cards/check_updates",
        params={"org": "Alenka", "since": cutoff},
        headers=_auth(test_key),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    by_guid = {r["card_guid"]: r for r in body}

    assert set(guids.values()) <= set(by_guid)          # nothing filtered out

    assert by_guid[guids["pending"]]["status"] == "pending"
    assert by_guid[guids["ignored"]]["ignored"] is True
    assert by_guid[guids["broken"]]["broken"] is True

    sample = by_guid[guids["pending"]]
    assert isinstance(sample["card_data"], dict)        # native JSONB, raw data present
    assert set(sample.keys()) == {
        "card_guid", "card_data", "status", "ignored", "broken",
        "formal_result", "diag_result", "icd_check_result", "updated_at",
    }
    assert "token_count" not in sample and "organization_id" not in sample


def test_check_updates_boundary_is_inclusive(client, test_key, seeded):
    """`since` equal to a row's own updated_at must still return that row: the
    client derives the value from a clock, so a strict > would drop cards that
    landed exactly on the boundary."""
    guids, cutoff = seeded
    resp = client.get(
        "/cards/check_updates",
        params={"org": "Alenka", "since": cutoff},
        headers=_auth(test_key),
    )
    assert resp.status_code == 200
    target = next(r for r in resp.json() if r["card_guid"] == guids["done"])

    resp = client.get(
        "/cards/check_updates",
        params={"org": "Alenka", "since": target["updated_at"]},
        headers=_auth(test_key),
    )
    assert resp.status_code == 200
    assert guids["done"] in {r["card_guid"] for r in resp.json()}


async def test_check_updates_is_org_scoped(client, test_key, mds_org_id, seeded):
    """A key authorized for both orgs must still not see MDS rows in an Alenka
    call — the organization_id filter has to do the work, not just auth."""
    guids, cutoff = seeded
    mds_guid = f"pytest-cuapi-mds-{uuid.uuid4().hex[:8]}"
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        await conn.execute(
            _INSERT,
            {"g": mds_guid, "d": _CARD, "s": "pending", "i": False,
             "b": False, "o": mds_org_id},
        )
    try:
        resp = client.get(
            "/cards/check_updates",
            params={"org": "Alenka", "since": cutoff},
            headers=_auth(test_key),
        )
        assert resp.status_code == 200
        got = {r["card_guid"] for r in resp.json()}
        assert set(guids.values()) <= got
        assert mds_guid not in got
    finally:
        async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
            await conn.execute("DELETE FROM done_cards WHERE card_guid = %(g)s", {"g": mds_guid})


async def test_check_updates_without_since_defaults_to_a_week(client, test_key, alenka_org_id, seeded):
    """No `since` → the last week only: recent rows come back, an older one does
    not, and the response isn't the whole table."""
    guids, _cutoff = seeded
    old_guid = f"pytest-cuapi-old-{uuid.uuid4().hex[:8]}"
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        await conn.execute(
            _INSERT,
            {"g": old_guid, "d": _CARD, "s": "done", "i": False,
             "b": False, "o": alenka_org_id},
        )
        # done_cards_set_updated_at fires BEFORE UPDATE and unconditionally stamps
        # now(), so a plain UPDATE can't backdate the row — disable it for this one
        # statement, otherwise the test would pass vacuously.
        await conn.execute("ALTER TABLE done_cards DISABLE TRIGGER done_cards_set_updated_at")
        try:
            await conn.execute(
                "UPDATE done_cards SET updated_at = now() - interval '30 days' "
                "WHERE card_guid = %(g)s",
                {"g": old_guid},
            )
        finally:
            await conn.execute("ALTER TABLE done_cards ENABLE TRIGGER done_cards_set_updated_at")
    try:
        resp = client.get(
            "/cards/check_updates",
            params={"org": "Alenka"},
            headers=_auth(test_key),
        )
        assert resp.status_code == 200
        got = {r["card_guid"] for r in resp.json()}
        assert set(guids.values()) <= got      # this week's rows present
        assert old_guid not in got             # 30-day-old row outside the window
    finally:
        async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
            await conn.execute("DELETE FROM done_cards WHERE card_guid = %(g)s", {"g": old_guid})
