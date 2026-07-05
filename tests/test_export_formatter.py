"""
Tests ApiFormatter.export against the real Postgres: since-filtering, limit=0
(no paging), and limit/cursor offset paging. Seeds and cleans up its own rows.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import psycopg
import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from api.app import create_app  # noqa: E402  (ensures src on path)
from reporting.api_formatter import ApiFormatter  # noqa: E402
from storage.organizations_storage import OrganizationsStorage  # noqa: E402


def _conninfo() -> str:
    return (
        f"host={os.environ['POSTGRES_HOST']} port={os.environ.get('POSTGRES_PORT','5432')} "
        f"dbname={os.environ['POSTGRES_DB']} user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


@pytest.fixture
async def alenka_org_id() -> str:
    async with OrganizationsStorage() as orgs:
        return await orgs.get_id_by_name("Alenka")


@pytest.fixture
async def seeded_guids(alenka_org_id: str):
    guids = [f"pytest-export-{uuid.uuid4()}" for _ in range(3)]
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        for g in guids:
            await conn.execute(
                "INSERT INTO done_cards (card_guid, card_data, ignored, organization_id) "
                "VALUES (%(g)s, %(d)s::jsonb, FALSE, %(o)s)",
                {"g": g, "d": '{"Прием": {"DATE": "01.07.2026"}}', "o": alenka_org_id},
            )
    yield guids, alenka_org_id
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        await conn.execute(
            "DELETE FROM done_cards WHERE card_guid = ANY(%(gs)s)", {"gs": guids}
        )


async def test_export_limit_zero_returns_all_with_native_jsonb(seeded_guids):
    guids, org_id = seeded_guids
    async with ApiFormatter() as fmt:
        rows = await fmt.export(org_id, since=None, limit=0, cursor=0)
    got = {r["card_guid"] for r in rows}
    assert set(guids) <= got
    sample = next(r for r in rows if r["card_guid"] in guids)
    assert isinstance(sample["card_data"], dict)             # native JSONB, not str
    assert set(sample.keys()) == {
        "card_guid", "card_data", "formal_result",
        "diag_result", "icd_check_result", "updated_at",
    }                                                        # trimmed, audited-only columns


async def test_export_cursor_offset_paging_is_exhaustive(seeded_guids):
    guids, org_id = seeded_guids
    seen, cursor = [], 0
    while True:
        async with ApiFormatter() as fmt:
            page = await fmt.export(org_id, since=None, limit=2, cursor=cursor)
        seen.extend(r["card_guid"] for r in page)
        if len(page) < 2:
            break
        cursor += 2
    assert set(guids) <= set(seen)
    assert len(seen) == len(set(seen))                       # no dup across pages
