"""
Integration tests for GET /visits/export — real Postgres via TestClient with an
api key scoped to Alenka + MDS. Every request is bounded to this test's own rows
via a `since` cutoff, so it never pages the whole production table. Covers: auth
required, native JSONB + trimmed columns, the include_ignored opt-in (broken
rows held back either way), and exhaustive/no-dup cursor paging.
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
        key_id = await api_keys.create_key("pytest-export", raw, [alenka_org_id, mds_org_id])
    yield raw
    async with ApiKeysStorage() as api_keys:
        await api_keys.revoke_key(key_id)


@pytest.fixture
async def seeded(alenka_org_id: str):
    """3 audited + 1 ignored + 1 broken Alenka rows, seeded after a captured cutoff
    so exports here are bounded to just these rows."""
    tag = uuid.uuid4().hex[:8]
    audited = [f"pytest-exapi-{tag}-a{i}" for i in range(3)]
    ignored_guid = f"pytest-exapi-{tag}-ign"
    broken_guid = f"pytest-exapi-{tag}-brk"
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        cur = await conn.execute("SELECT now()")
        cutoff = (await cur.fetchone())[0]
        for g in audited:
            await conn.execute(
                "INSERT INTO done_cards (card_guid, card_data, ignored, broken, organization_id) "
                "VALUES (%(g)s, %(d)s::jsonb, FALSE, FALSE, %(o)s)",
                {"g": g, "d": _CARD, "o": alenka_org_id},
            )
        await conn.execute(
            "INSERT INTO done_cards (card_guid, card_data, status, ignored, broken, organization_id) "
            "VALUES (%(g)s, %(d)s::jsonb, 'done', TRUE, FALSE, %(o)s)",   # ignored -> exported as 'ignored'
            {"g": ignored_guid, "d": _CARD, "o": alenka_org_id},
        )
        await conn.execute(
            "INSERT INTO done_cards (card_guid, card_data, ignored, broken, organization_id) "
            "VALUES (%(g)s, %(d)s::jsonb, FALSE, TRUE, %(o)s)",           # broken -> excluded
            {"g": broken_guid, "d": _CARD, "o": alenka_org_id},
        )
    yield audited, ignored_guid, broken_guid, cutoff.isoformat()
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        await conn.execute(
            "DELETE FROM done_cards WHERE card_guid = ANY(%(gs)s)",
            {"gs": audited + [ignored_guid, broken_guid]},
        )


def test_export_requires_key(client: TestClient):
    resp = client.get("/visits/export", params={"org": "Alenka"})
    assert resp.status_code in (401, 403)


def test_export_returns_rows_native_jsonb_trimmed(client, test_key, seeded):
    """Default call: audited rows only, ignored and broken both absent."""
    audited, ign, brk, cutoff = seeded
    resp = client.get(
        "/visits/export",
        params={"org": "Alenka", "since": cutoff},
        headers=_auth(test_key),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    by_guid = {r["card_guid"]: r for r in body}
    assert set(audited) <= set(by_guid)                        # audited cards present
    assert ign not in by_guid and brk not in by_guid           # ignored/broken excluded
    sample = by_guid[audited[0]]
    assert isinstance(sample["card_data"], dict)               # native JSONB in JSON response
    assert set(sample.keys()) == {
        "card_guid", "card_data", "status", "formal_result",
        "diag_result", "icd_check_result", "updated_at",
    }                                                          # trimmed to seven columns
    assert "token_count" not in sample and "organization_id" not in sample


def test_export_include_ignored_opts_in(client, test_key, seeded):
    """include_ignored=true adds the skipped cards under their own status. Those
    are a clinic-requested filter, not failures, and they still hold the 1C
    record — but broken rows stay out even here."""
    audited, ign, brk, cutoff = seeded
    resp = client.get(
        "/visits/export",
        params={"org": "Alenka", "since": cutoff, "include_ignored": "true"},
        headers=_auth(test_key),
    )
    assert resp.status_code == 200
    by_guid = {r["card_guid"]: r for r in resp.json()}
    assert set(audited) <= set(by_guid)                        # audited rows still there
    assert ign in by_guid
    assert by_guid[ign]["status"] == "ignored"                 # never labelled 'done'
    assert brk not in by_guid


def test_export_cursor_offset_paging(client, test_key, seeded):
    audited, _ign, _brk, cutoff = seeded
    seen, cursor = [], 0
    while True:
        resp = client.get(
            "/visits/export",
            params={"org": "Alenka", "since": cutoff, "limit": 2, "cursor": cursor},
            headers=_auth(test_key),
        )
        assert resp.status_code == 200
        page = resp.json()
        seen.extend(r["card_guid"] for r in page)
        if len(page) < 2:
            break
        cursor += 2
    assert set(audited) <= set(seen)
    assert len(seen) == len(set(seen))                         # no dup across pages
# --- append this test to tests/test_export_api.py (org-scoping, spec §7) ---


async def test_export_is_org_scoped(client, test_key, mds_org_id, seeded):
    """A row owned by another org (MDS) must never appear in an Alenka export,
    even though the api key is authorized for both orgs — the WHERE
    organization_id filter, not just auth, must exclude it."""
    audited, _ign, _brk, cutoff = seeded
    mds_guid = f"pytest-exapi-mds-{uuid.uuid4().hex[:8]}"
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        await conn.execute(
            "INSERT INTO done_cards (card_guid, card_data, ignored, broken, organization_id) "
            "VALUES (%(g)s, %(d)s::jsonb, FALSE, FALSE, %(o)s)",
            {"g": mds_guid, "d": _CARD, "o": mds_org_id},
        )
    try:
        resp = client.get(
            "/visits/export",
            params={"org": "Alenka", "since": cutoff},
            headers=_auth(test_key),
        )
        assert resp.status_code == 200
        got = {r["card_guid"] for r in resp.json()}
        assert set(audited) <= got             # Alenka's own rows present
        assert mds_guid not in got             # MDS row NOT leaked into an Alenka pull
    finally:
        async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
            await conn.execute("DELETE FROM done_cards WHERE card_guid = %(g)s", {"g": mds_guid})
