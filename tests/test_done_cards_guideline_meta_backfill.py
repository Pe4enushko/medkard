"""
Integration tests for the two DoneCardsStorage methods the guideline-meta
backfill leans on (scripts/hacks/backfill-guideline-meta.py). Hits the real
configured Postgres. Сам план бэкфилла — tests/test_backfill_guideline_meta.py.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from storage.done_cards_storage import DoneCardsStorage
from storage.models.result import DiagnosisResult, FormalStructureResult

_NOW = datetime.now(timezone.utc)


async def _insert(guid: str, diagnosis: list[DiagnosisResult]) -> str:
    async with DoneCardsStorage() as storage:
        return await storage.upsert(
            card_guid=guid,
            card_data=json.dumps({"Прием": {"GUID": guid}}, ensure_ascii=False),
            formal=FormalStructureResult(),
            diagnosis=diagnosis,
            icd_check=None,
            token_count=0,
            time_ms=0,
            started_at=_NOW,
            finished_at=_NOW,
        )


async def _cleanup(*guids: str) -> None:
    async with DoneCardsStorage() as storage:
        async with storage._pool.connection() as conn:
            await conn.execute(
                "DELETE FROM done_cards WHERE card_guid = ANY(%(g)s)", {"g": list(guids)}
            )


@pytest.mark.asyncio
async def test_lists_only_cards_whose_diagnosis_has_no_snapshot():
    bare = f"pytest-meta-bare-{uuid.uuid4()}"
    stamped = f"pytest-meta-stamped-{uuid.uuid4()}"
    empty = f"pytest-meta-empty-{uuid.uuid4()}"
    try:
        bare_id = await _insert(
            bare, [DiagnosisResult(icd_code="J01", guideline_file_id="file-1")]
        )
        stamped_id = await _insert(
            stamped,
            [DiagnosisResult(
                icd_code="J01",
                guideline_file_id="file-1",
                guideline_meta={"name": "КР", "date": "2024", "age_group": "Взрослые"},
            )],
        )
        # Клинрека не нашлось ещё при аудите: разворачивать нечего, и такая карта
        # не должна крутиться в выборке вечно.
        empty_id = await _insert(empty, [DiagnosisResult(icd_code="Z00")])

        async with DoneCardsStorage() as storage:
            rows = await storage.list_diag_results_to_backfill(limit=0, after_id="")

        found = {row["id"] for row in rows}
        assert bare_id in found
        assert stamped_id not in found
        assert empty_id not in found
    finally:
        await _cleanup(bare, stamped, empty)


@pytest.mark.asyncio
async def test_keyset_paging_does_not_repeat_a_card():
    first = f"pytest-meta-page-a-{uuid.uuid4()}"
    second = f"pytest-meta-page-b-{uuid.uuid4()}"
    try:
        await _insert(first, [DiagnosisResult(icd_code="J01", guideline_file_id="file-1")])
        await _insert(second, [DiagnosisResult(icd_code="J02", guideline_file_id="file-2")])

        async with DoneCardsStorage() as storage:
            page = await storage.list_diag_results_to_backfill(limit=1, after_id="")
            assert len(page) == 1
            nxt = await storage.list_diag_results_to_backfill(
                limit=1, after_id=page[0]["id"]
            )

        assert nxt and nxt[0]["id"] != page[0]["id"]
        assert nxt[0]["id"] > page[0]["id"]
    finally:
        await _cleanup(first, second)


@pytest.mark.asyncio
async def test_set_diag_result_writes_the_snapshot_and_bumps_updated_at():
    guid = f"pytest-meta-write-{uuid.uuid4()}"
    try:
        row_id = await _insert(
            guid, [DiagnosisResult(icd_code="J01", guideline_file_id="file-1")]
        )
        async with DoneCardsStorage() as storage:
            async with storage._pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT updated_at FROM done_cards WHERE id = %(id)s::uuid",
                    {"id": row_id},
                )
                before = (await cur.fetchone())["updated_at"]

            written = await storage.set_diag_result(
                card_id=row_id,
                diag_json=json.dumps(
                    [{
                        "icd_code": "J01",
                        "guideline_file_id": "file-1",
                        "guideline_meta": {"name": "КР", "date": "2024",
                                           "age_group": "Взрослые"},
                    }],
                    ensure_ascii=False,
                ),
            )
            assert written == 1

            async with storage._pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT diag_result, updated_at FROM done_cards WHERE id = %(id)s::uuid",
                    {"id": row_id},
                )
                row = await cur.fetchone()

        assert row["diag_result"][0]["guideline_meta"]["name"] == "КР"
        # Реплика движка забирает карты по updated_at: без сдвига бэкфилл до неё
        # не доедет вовсе.
        assert row["updated_at"] > before
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_a_card_with_our_errors_is_listed_even_with_a_snapshot():
    # Строку надо почистить от errors независимо от того, развёрнут ли снимок.
    guid = f"pytest-meta-degraded-{uuid.uuid4()}"
    try:
        row_id = await _insert(guid, [DiagnosisResult(
            icd_code="I67.9",
            guideline_file_id="file-1",
            guideline_meta={"name": "КР", "date": "2024", "age_group": "Взрослые"},
            errors=["judge_criteria: пусто"],
        )])
        # Аудит пишет errors уже в свою колонку, поэтому строку с ними для теста
        # собираем руками — ровно так она выглядит у карт до этой ветки.
        async with DoneCardsStorage() as storage:
            async with storage._pool.connection() as conn:
                await conn.execute(
                    "UPDATE done_cards SET diag_result = jsonb_set(diag_result, "
                    "'{0,errors}', %(e)s::jsonb), diag_errors = NULL "
                    "WHERE id = %(id)s::uuid",
                    {"id": row_id, "e": json.dumps(["judge_criteria: пусто"])},
                )
            rows = await storage.list_diag_results_to_backfill(limit=0, after_id="")

        assert row_id in {row["id"] for row in rows}
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_set_diag_result_appends_the_degradation_it_took_out():
    guid = f"pytest-meta-append-{uuid.uuid4()}"
    try:
        row_id = await _insert(guid, [DiagnosisResult(
            icd_code="J01", guideline_file_id="file-1", errors=["judge_treatment: таймаут"],
        )])
        async with DoneCardsStorage() as storage:
            await storage.set_diag_result(
                card_id=row_id,
                diag_json=json.dumps([{"icd_code": "J01", "guideline_file_id": "file-1"}]),
                diag_errors=["J01: judge_criteria: пусто"],
            )
            async with storage._pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT diag_errors FROM done_cards WHERE id = %(id)s::uuid",
                    {"id": row_id},
                )
                stored = (await cur.fetchone())["diag_errors"]

        # Аварию, записанную аудитом, бэкфилл дополняет, а не затирает.
        assert stored == ["J01: judge_treatment: таймаут", "J01: judge_criteria: пусто"]
    finally:
        await _cleanup(guid)
