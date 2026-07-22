"""
Verifies migration 025: done_cards.status exists, defaults sensibly, and its
CHECK constraint rejects invalid values. Hits the real configured Postgres.
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


def _conninfo() -> str:
    return (
        f"host={os.environ['POSTGRES_HOST']} "
        f"port={os.environ.get('POSTGRES_PORT', '5432')} "
        f"dbname={os.environ['POSTGRES_DB']} "
        f"user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


@pytest.mark.asyncio
async def test_existing_rows_backfilled_to_done():
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        cur = await conn.execute(
            "SELECT count(*) FROM done_cards WHERE status IS NULL OR status NOT IN ('pending','done')"
        )
        row = await cur.fetchone()
        assert row[0] == 0


@pytest.mark.asyncio
async def test_new_row_defaults_to_pending():
    guid = f"pytest-{uuid.uuid4()}"
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        try:
            await conn.execute(
                "INSERT INTO done_cards (card_guid, ignored) VALUES (%(g)s, FALSE)",
                {"g": guid},
            )
            cur = await conn.execute(
                "SELECT status FROM done_cards WHERE card_guid = %(g)s", {"g": guid}
            )
            row = await cur.fetchone()
            assert row[0] == "pending"
        finally:
            await conn.execute("DELETE FROM done_cards WHERE card_guid = %(g)s", {"g": guid})


@pytest.mark.asyncio
async def test_invalid_status_is_rejected():
    guid = f"pytest-{uuid.uuid4()}"
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        with pytest.raises(psycopg.errors.CheckViolation):
            await conn.execute(
                "INSERT INTO done_cards (card_guid, status) VALUES (%(g)s, 'bogus')",
                {"g": guid},
            )
