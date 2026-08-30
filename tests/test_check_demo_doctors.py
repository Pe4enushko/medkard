"""Unit tests for scripts/hacks/check-demo-doctors.py — the checks, not the DB.

Pure functions only: the SQL side is what the script prints from a live base.
"""

import importlib.util
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "check_demo_doctors",
    Path(__file__).resolve().parent.parent / "scripts" / "hacks" / "check-demo-doctors.py")
check = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = check
_spec.loader.exec_module(check)


def _write(tmp_path: Path, payload) -> str:
    path = tmp_path / "demo_doctors.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


# ── переключатель ────────────────────────────────────────────────────────────

def test_empty_switch_is_a_problem():
    problems = check.check_switch("", "Alenka", ["Alenka", "MDS"])
    assert problems and any("DEMO_DOCTOR_STAMP_ORG" in p for p in problems)


def test_switch_naming_an_unknown_org_is_a_problem():
    # Опечатка в .env выглядит как включённый костыль, а штампа нет.
    problems = check.check_switch("Alenkaa", "Alenkaa", ["Alenka", "MDS"])
    assert problems and any("Alenkaa" in p for p in problems)


def test_switch_matching_the_org_case_insensitively_is_clean():
    assert check.check_switch("alenka", "Alenka", ["Alenka", "MDS"]) == []


def test_switch_pointing_at_another_org_than_the_one_checked():
    problems = check.check_switch("MDS", "Alenka", ["Alenka", "MDS"])
    assert problems and any("MDS" in p and "Alenka" in p for p in problems)


# ── файл врачей ──────────────────────────────────────────────────────────────

def test_missing_file_is_a_problem(tmp_path):
    doctors, problems = check.check_doctors_file(str(tmp_path / "нет.json"))
    assert doctors == [] and problems


def test_empty_list_is_a_problem(tmp_path):
    doctors, problems = check.check_doctors_file(_write(tmp_path, []))
    assert doctors == [] and problems


def test_template_placeholders_are_a_problem(tmp_path):
    # Шаблон из репозитория: штамп сработает, но у всех врачей одно ФИО, и
    # движок, который сопоставляет врачей по ФИО, сведёт их в одного.
    path = _write(tmp_path, [
        {"code": "90001", "name": "Фамилия Имя Отчество"},
        {"code": "90002", "name": "Фамилия Имя Отчество"},
    ])
    _, problems = check.check_doctors_file(path)
    assert any("шаблон" in p.lower() for p in problems)


def test_duplicate_names_are_a_problem(tmp_path):
    path = _write(tmp_path, [
        {"code": "90001", "name": "Иванов И. И."},
        {"code": "90002", "name": "Иванов И. И."},
    ])
    _, problems = check.check_doctors_file(path)
    assert any("ФИО" in p for p in problems)


def test_duplicate_codes_are_a_problem(tmp_path):
    path = _write(tmp_path, [
        {"code": "90001", "name": "Иванов И. И."},
        {"code": "90001", "name": "Петров П. П."},
    ])
    _, problems = check.check_doctors_file(path)
    assert any("код" in p.lower() for p in problems)


def test_a_healthy_file_is_clean(tmp_path):
    path = _write(tmp_path, [
        {"code": "90001", "name": "Иванов И. И."},
        {"code": "90002", "name": "Петров П. П."},
    ])
    doctors, problems = check.check_doctors_file(path)
    assert [d["code"] for d in doctors] == ["90001", "90002"]
    assert problems == []


# ── вывод про происхождение карт без врача ───────────────────────────────────

def test_pulled_cards_without_a_doctor_explain_the_crutch_gap():
    """Карта без pushed_at пришла ночным пулом, а штамп висит только на пуше."""
    verdict = check.explain_missing(pushed=0, pulled=42)
    assert "ночн" in verdict.lower()
    assert "backfill-demo-doctors" in verdict


def test_pushed_cards_without_a_doctor_point_at_the_switch():
    verdict = check.explain_missing(pushed=42, pulled=0)
    assert "пуш" in verdict            # кириллица: латинское "push" тут не встречается
    assert "DEMO_DOCTOR_STAMP_ORG" in verdict
    assert "backfill-demo-doctors" in verdict


def test_no_cards_without_a_doctor_needs_no_advice():
    assert check.explain_missing(pushed=0, pulled=0) == ""
