"""Unit tests for scripts/backfill-demo-doctors.py — the plan, not the DB.

The storage side is covered by tests/test_demo_doctors_storage.py, the stamp
itself by tests/test_demo_doctors.py.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "backfill_demo_doctors",
    Path(__file__).resolve().parent.parent / "scripts" / "backfill-demo-doctors.py")
backfill = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = backfill
_spec.loader.exec_module(backfill)

DOCTORS = [
    {"code": "90001", "name": "Врач 90001"},
    {"code": "90002", "name": "Врач 90002"},
    {"code": "90003", "name": "Врач 90003"},
]


def test_every_card_gets_a_doctor():
    guids = [f"g{i}" for i in range(20)]
    assigned = backfill.assign(guids, DOCTORS)
    assert sorted(g for batch in assigned.values() for g in batch) == sorted(guids)


def test_no_card_is_assigned_twice():
    guids = [f"g{i}" for i in range(20)]
    batches = list(backfill.assign(guids, DOCTORS).values())
    flat = [g for batch in batches for g in batch]
    assert len(flat) == len(set(flat))


def test_only_doctors_from_the_file_are_used():
    assigned = backfill.assign([f"g{i}" for i in range(20)], DOCTORS)
    assert set(assigned) <= {d["code"] for d in DOCTORS}


def test_cards_spread_over_the_doctors():
    # One doctor taking the whole clinic would make the personal-report demo
    # pointless: 40 draws landing on one of three names is ~2e-19.
    assigned = backfill.assign([f"g{i}" for i in range(40)], DOCTORS)
    assert len(assigned) == 3


def test_no_cards_no_batches():
    assert backfill.assign([], DOCTORS) == {}


def test_no_doctors_is_an_error():
    # An empty or unreadable file must stop the backfill, not silently write
    # nothing: the operator asked for a stamp and has to hear that it did not
    # happen.
    with pytest.raises(SystemExit):
        backfill.assign(["g1"], [])
