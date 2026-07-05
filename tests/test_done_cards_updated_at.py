"""
Verifies the done_cards.updated_at trigger (migration 019): updated_at is set
on insert and advances on update. Hits the real configured Postgres.
"""
from __future__ import annotations

import asyncio
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


async def _fetch_updated_at(conn, guid: str):
    cur = await conn.execute(
        "SELECT updated_at FROM done_cards WHERE card_guid = %(g)s", {"g": guid}
    )
    row = await cur.fetchone()
    return row[0]


@pytest.mark.asyncio
async def test_updated_at_set_on_insert_and_advances_on_update():
    guid = f"pytest-{uuid.uuid4()}"
    async with await psycopg.AsyncConnection.connect(_conninfo(), autocommit=True) as conn:
        try:
            await conn.execute(
                "INSERT INTO done_cards (card_guid, ignored) VALUES (%(g)s, FALSE)",
                {"g": guid},
            )
            first = await _fetch_updated_at(conn, guid)
            assert first is not None

            await asyncio.sleep(0.01)
            await conn.execute(
                "UPDATE done_cards SET token_count = 1 WHERE card_guid = %(g)s",
                {"g": guid},
            )
            second = await _fetch_updated_at(conn, guid)
            assert second > first
        finally:
            await conn.execute("DELETE FROM done_cards WHERE card_guid = %(g)s", {"g": guid})
