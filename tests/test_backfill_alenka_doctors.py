"""scripts/hacks/backfill-alenka-doctors.py — the pass over stored cards, not the DB."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "backfill_alenka_doctors",
    Path(__file__).resolve().parent.parent / "scripts" / "hacks" / "backfill-alenka-doctors.py")
backfill = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = backfill
_spec.loader.exec_module(backfill)

ALENKA = {
    "Прием": {"GUID": "g1", "DATE": "09.09.2026"},
    "Врач": {"GUID": "0a99", "FIO": "Правкина И. Г.", "SPECIALIZATION": "Педиатр"},
    "Пациент": {"AGE": "1"},
}


class _FakeStorage:
    def __init__(self, rows):
        self.rows = rows
        self.written: dict[str, dict] = {}

    async def list_cards_with_top_level_doctor(self, *, limit, after_id):
        rows = [r for r in self.rows if r["id"] > after_id]
        return rows[:limit]

    async def set_card_data(self, *, card_id, card_json):
        self.written[card_id] = json.loads(card_json)
        return 1


def test_card_is_rewritten_in_our_form():
    storage = _FakeStorage([{"id": "a", "card_data": ALENKA}])
    totals = asyncio.run(backfill._run(storage, limit=0, batch=10, apply=True))
    assert totals["cards"] == 1
    assert storage.written["a"]["Прием"]["Врач_код"] == "0a99"
    assert storage.written["a"]["Врач"] == {"SPECIALIZATION": "Педиатр"}


def test_dry_run_writes_nothing():
    storage = _FakeStorage([{"id": "a", "card_data": ALENKA}])
    totals = asyncio.run(backfill._run(storage, limit=0, batch=10, apply=False))
    assert totals["cards"] == 1
    assert storage.written == {}


def test_pages_by_id_cursor():
    rows = [{"id": f"{i:02d}", "card_data": ALENKA} for i in range(5)]
    storage = _FakeStorage(rows)
    totals = asyncio.run(backfill._run(storage, limit=0, batch=2, apply=True))
    assert totals["cards"] == 5
    assert sorted(storage.written) == [r["id"] for r in rows]


def test_limit_caps_the_pass():
    rows = [{"id": f"{i:02d}", "card_data": ALENKA} for i in range(5)]
    storage = _FakeStorage(rows)
    totals = asyncio.run(backfill._run(storage, limit=3, batch=2, apply=True))
    assert totals["cards"] == 3
