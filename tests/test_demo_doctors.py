"""Unit tests for src/api/demo_doctors.py — the temporary Alenka doctor stamp.

Pure functions only: no DB. The storage side (bulk stamp and revert) is
covered by tests/test_backfill_demo_doctors_storage.py.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from api import demo_doctors

DOCTORS = [
    {"code": "90001", "name": "Врач 90001"},
    {"code": "90002", "name": "Врач 90002"},
]


@pytest.fixture(autouse=True)
def clean_cache():
    demo_doctors.load_doctors.cache_clear()
    yield
    demo_doctors.load_doctors.cache_clear()


def _card(priem: dict | None = None) -> dict:
    card = {"Пациент": {"Код": "к0138172"}, "Диагнозы": []}
    if priem is not None:
        card["Прием"] = priem
    return card


def _write(tmp_path: Path, payload) -> str:
    path = tmp_path / "demo_doctors.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


# ── the switch ───────────────────────────────────────────────────────────────

def test_disabled_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("DEMO_DOCTOR_STAMP_ORG", raising=False)
    assert demo_doctors.enabled_for("Alenka") is False


def test_enabled_only_for_the_named_org(monkeypatch):
    monkeypatch.setenv("DEMO_DOCTOR_STAMP_ORG", "Alenka")
    assert demo_doctors.enabled_for("Alenka") is True
    assert demo_doctors.enabled_for("MDS") is False


def test_org_name_matches_case_insensitively(monkeypatch):
    # ?org= is resolved case-insensitively by require_org_access, and the
    # canonical name comes back from the DB — a lowercase .env must still match.
    monkeypatch.setenv("DEMO_DOCTOR_STAMP_ORG", "alenka")
    assert demo_doctors.enabled_for("Alenka") is True


def test_blank_env_is_off(monkeypatch):
    monkeypatch.setenv("DEMO_DOCTOR_STAMP_ORG", "   ")
    assert demo_doctors.enabled_for("Alenka") is False


# ── the file ─────────────────────────────────────────────────────────────────

def test_loads_doctors_from_json(tmp_path):
    assert demo_doctors.load_doctors(_write(tmp_path, DOCTORS)) == DOCTORS


def test_missing_file_yields_no_doctors(tmp_path):
    assert demo_doctors.load_doctors(str(tmp_path / "nope.json")) == []


def test_entries_without_code_or_name_are_dropped(tmp_path):
    path = _write(tmp_path, [{"code": "90001"}, {"name": "Без кода"},
                             {"code": "90002", "name": "Целая строка"}])
    assert demo_doctors.load_doctors(path) == [{"code": "90002", "name": "Целая строка"}]


def test_broken_json_yields_no_doctors(tmp_path):
    path = tmp_path / "demo_doctors.json"
    path.write_text("{не json", encoding="utf-8")
    assert demo_doctors.load_doctors(str(path)) == []


# ── the stamp ────────────────────────────────────────────────────────────────

def test_stamps_a_doctor_from_the_file():
    card = demo_doctors.stamp(_card({"GUID": "g1"}), previous=None, doctors=DOCTORS)
    priem = card["Прием"]
    assert (priem["Врач_код"], priem["Врач"]) in [(d["code"], d["name"]) for d in DOCTORS]


def test_keeps_the_doctor_the_card_already_carries():
    # 1C started sending doctors: real data outranks the hack, and the day that
    # happens the stamp turns into a no-op instead of overwriting live values.
    card = demo_doctors.stamp(
        _card({"GUID": "g1", "Врач_код": "1701", "Врач": "Настоящий врач"}),
        previous=None, doctors=DOCTORS)
    assert card["Прием"]["Врач_код"] == "1701"
    assert card["Прием"]["Врач"] == "Настоящий врач"


def test_carries_the_doctor_over_from_the_stored_card():
    # 1C re-pushes the same visit and upsert_pending rewrites card_data whole.
    # Without this the doctor would be re-drawn on every push, and the same
    # visit would answer with a different doctor mid-demo.
    stored = {"GUID": "g1", "Врач_код": "90002", "Врач": "Врач 90002"}
    card = demo_doctors.stamp(_card({"GUID": "g1"}), previous=stored, doctors=DOCTORS)
    assert card["Прием"]["Врач_код"] == "90002"
    assert card["Прием"]["Врач"] == "Врач 90002"


def test_a_stored_card_without_a_doctor_does_not_block_the_stamp():
    card = demo_doctors.stamp(_card({"GUID": "g1"}), previous={"GUID": "g1"},
                              doctors=DOCTORS)
    assert card["Прием"]["Врач_код"]


def test_card_without_priem_is_left_alone():
    card = _card()
    assert demo_doctors.stamp(card, previous=None, doctors=DOCTORS) == card


def test_empty_doctor_list_leaves_the_card_alone():
    # A missing or unreadable file must not cost the clinic its ingest.
    card = _card({"GUID": "g1"})
    assert demo_doctors.stamp(card, previous=None, doctors=[]) == card


def test_the_original_card_is_not_mutated():
    card = _card({"GUID": "g1"})
    demo_doctors.stamp(card, previous=None, doctors=DOCTORS)
    assert "Врач_код" not in card["Прием"]


def test_the_rest_of_the_priem_block_survives():
    card = demo_doctors.stamp(_card({"GUID": "g1", "DATE": "20.08.2026"}),
                              previous=None, doctors=DOCTORS)
    assert card["Прием"]["DATE"] == "20.08.2026"


def test_draws_more_than_one_doctor_over_many_cards():
    # Every card getting the same doctor would make the personal-report demo
    # meaningless: 30 draws from two names collide with probability 2**-29.
    seen = {demo_doctors.stamp(_card({"GUID": f"g{i}"}), previous=None,
                               doctors=DOCTORS)["Прием"]["Врач_код"]
            for i in range(30)}
    assert len(seen) == 2
