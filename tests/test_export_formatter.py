"""
Tests ApiFormatter.export against the real Postgres. Every export is bounded to
this test's own rows via a `since` cutoff captured *before* seeding, so it never
pages the whole production table. Covers: native JSONB + six-column trim,
status labelling, exclusive `since`, and exhaustive/no-dup cursor paging.
Ignored rows are exported (labelled `status='ignored'`); only broken ones are
held back.
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

_CARD = '{"Прием": {"DATE": "01.07.2026"}}'


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
async def seeded(alenka_org_id: str):
    """Seed (after a captured `cutoff`) 3 audited + 1 ignored + 1 broken row for
    Alenka. Yields (audited_guids, ignored_guid, broken_guid, org_id, cutoff_iso).
    `cutoff` bounds every export in these tests to just these rows, so paging
    never walks the whole production table."""
    tag = uuid.uuid4().hex[:8]
    audited = [f"pytest-export-{tag}-a{i}" for i in range(3)]
    ignored_guid = f"pytest-export-{tag}-ign"
    broken_guid = f"pytest-export-{tag}-brk"
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        cur = await conn.execute("SELECT now()")
        cutoff = (await cur.fetchone())[0]                      # tz-aware datetime, before any seed
        for g in audited:
            await conn.execute(
                "INSERT INTO done_cards (card_guid, card_data, ignored, broken, organization_id) "
                "VALUES (%(g)s, %(d)s::jsonb, FALSE, FALSE, %(o)s)",
                {"g": g, "d": _CARD, "o": alenka_org_id},
            )
        await conn.execute(
            "INSERT INTO done_cards (card_guid, card_data, status, ignored, broken, organization_id) "
            "VALUES (%(g)s, %(d)s::jsonb, 'done', TRUE, FALSE, %(o)s)",   # ignored -> exported, relabelled
            {"g": ignored_guid, "d": _CARD, "o": alenka_org_id},
        )
        await conn.execute(
            "INSERT INTO done_cards (card_guid, card_data, ignored, broken, organization_id) "
            "VALUES (%(g)s, %(d)s::jsonb, FALSE, TRUE, %(o)s)",           # broken -> must be excluded
            {"g": broken_guid, "d": _CARD, "o": alenka_org_id},
        )
    yield audited, ignored_guid, broken_guid, alenka_org_id, cutoff.isoformat()
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        await conn.execute(
            "DELETE FROM done_cards WHERE card_guid = ANY(%(gs)s)",
            {"gs": audited + [ignored_guid, broken_guid]},
        )


async def test_export_trims_to_six_native_jsonb_columns(seeded):
    audited, _ign, _brk, org_id, cutoff = seeded
    async with ApiFormatter() as fmt:
        rows = await fmt.export(org_id, since=cutoff, limit=0, cursor=0)
    got = {r["card_guid"] for r in rows}
    assert set(audited) <= got
    sample = next(r for r in rows if r["card_guid"] in audited)
    assert isinstance(sample["card_data"], dict)               # native JSONB, not str
    assert set(sample.keys()) == {
        "card_guid", "card_data", "status", "formal_result",
        "diag_result", "icd_check_result", "updated_at",
    }                                                          # trimmed to the seven export columns


async def test_export_includes_ignored_but_not_broken(seeded):
    """Ignored cards carry real 1C data — a filter the clinics asked for, not a
    failure — so consumers tracing a patient need them. Broken rows hold a
    stacktrace instead of a visit and stay out."""
    audited, ign, brk, org_id, cutoff = seeded
    async with ApiFormatter() as fmt:
        rows = await fmt.export(org_id, since=cutoff, limit=0, cursor=0)
    by_guid = {r["card_guid"]: r for r in rows}
    assert set(audited) <= set(by_guid)                        # audited cards present
    assert ign in by_guid                                      # ignored exported...
    assert by_guid[ign]["status"] == "ignored"                 # ...and never labelled 'done'
    assert brk not in by_guid                                  # broken still filtered out


async def test_export_since_is_exclusive(seeded):
    audited, _ign, _brk, org_id, cutoff = seeded
    async with ApiFormatter() as fmt:
        all_rows = await fmt.export(org_id, since=cutoff, limit=0, cursor=0)
    first = next(r for r in all_rows if r["card_guid"] == audited[0])
    since2 = first["updated_at"].isoformat()                    # exclusive cutoff at the first row
    async with ApiFormatter() as fmt:
        rows = await fmt.export(org_id, since=since2, limit=0, cursor=0)
    got = {r["card_guid"] for r in rows}
    assert audited[0] not in got                               # its own row excluded (> is exclusive)
    assert audited[-1] in got                                  # a strictly-later row still returned


async def test_export_cursor_paging_exhaustive_no_dup(seeded):
    audited, _ign, _brk, org_id, cutoff = seeded
    seen, cursor = [], 0
    while True:
        async with ApiFormatter() as fmt:
            page = await fmt.export(org_id, since=cutoff, limit=2, cursor=cursor)
        seen.extend(r["card_guid"] for r in page)
        if len(page) < 2:
            break
        cursor += 2
    assert set(audited) <= set(seen)
    assert len(seen) == len(set(seen))                         # no row repeated across pages
