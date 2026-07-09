"""Интеграционные тесты storage.guidelines_storage.GuidelinesStorage.

Требует настроенный Postgres (.env) с применённой миграцией 019.
Запускается на стенде — на dev-машине нет доступа к БД.
Каждый тест чистит вставленные им строки.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from storage.guidelines_storage import GuidelinesStorage
from storage.models.guideline import Guideline


def _g(file_id: str, mkb: list[str], age: list[str], name: str = "test") -> Guideline:
    return Guideline(file_id=file_id, name=name, mkb=mkb, age_category=age)


async def _cleanup(storage: GuidelinesStorage, file_ids: list[str]) -> None:
    async with storage._pool.connection() as conn:
        await conn.execute(
            "DELETE FROM guidelines WHERE file_id = ANY(%(ids)s)", {"ids": file_ids}
        )


async def test_upsert_and_get_roundtrip():
    fid = f"test_{uuid.uuid4().hex}"
    async with GuidelinesStorage() as storage:
        try:
            n = await storage.upsert_many([_g(fid, ["J20.0", "J20.1"], ["Взрослые"])])
            assert n == 1
            got = await storage.get(fid)
            assert got is not None
            assert got.mkb == ["J20.0", "J20.1"]
            assert got.age_category == ["Взрослые"]
        finally:
            await _cleanup(storage, [fid])


async def test_upsert_is_idempotent_by_file_id():
    fid = f"test_{uuid.uuid4().hex}"
    async with GuidelinesStorage() as storage:
        try:
            await storage.upsert_many([_g(fid, ["A15"], ["Дети"], name="one")])
            await storage.upsert_many([_g(fid, ["A16"], ["Взрослые"], name="two")])
            got = await storage.get(fid)
            assert got.name == "two"
            assert got.mkb == ["A16"]
        finally:
            await _cleanup(storage, [fid])


async def test_find_by_code_matches_array_member():
    fid = f"test_{uuid.uuid4().hex}"
    async with GuidelinesStorage() as storage:
        try:
            await storage.upsert_many([_g(fid, ["J20.0", "J20.1"], ["Взрослые"])])
            found = await storage.find_by_code("j20.1")  # нижний регистр — нормализуется
            assert fid in {g.file_id for g in found}
            assert not any(g.file_id == fid for g in await storage.find_by_code("Z99.9"))
        finally:
            await _cleanup(storage, [fid])


async def test_find_by_prefix_strips_subcategory():
    fid = f"test_{uuid.uuid4().hex}"
    async with GuidelinesStorage() as storage:
        try:
            await storage.upsert_many([_g(fid, ["J20.9"], ["Взрослые"])])
            found = await storage.find_by_prefix("J20")
            assert fid in {g.file_id for g in found}
        finally:
            await _cleanup(storage, [fid])
