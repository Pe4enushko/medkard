"""Синонимические ряды контролируемых показателей 168н."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.deterministic.indicators import (
    _codes_of,
    _rows,
    group_name,
    groups_in_text,
    required_groups,
)

_INDICATORS = json.loads((ROOT / "resources" / "168n_indicators.json").read_text(encoding="utf-8"))
# Заключения врача: проверить их сравнением нельзя, ряды на них не строятся.
_CONCLUSIONS = (
    "отсутствие", "исключение", "достижение", "признаки атипии",
    "контроль лабораторных", "цитологическая/мо", "оценка размеров",
    "при гастроэзофагеальном",
)


def _measurable() -> list[tuple[str, int]]:
    path = ROOT / "docs" / "superpowers" / "research" / "2026-08-31-npa-sweep" / "168n-indicators.csv"
    with open(path, encoding="utf-8", newline="") as f:
        return [
            (r["показатель"], int(r["вхождений"]))
            for r in csv.DictReader(f)
            if not r["показатель"].lower().startswith(_CONCLUSIONS)
        ]


# ── файл рядов ────────────────────────────────────────────────────────────────

def test_group_ids_are_unique():
    ids = [g["id"] for g in _INDICATORS["groups"]]
    assert len(ids) == len(set(ids)), ids


def test_every_group_has_a_name_and_patterns():
    for g in _INDICATORS["groups"]:
        assert g["name"], g["id"]
        assert g["patterns"], g["id"]


def test_measurable_indicators_of_the_order_are_covered():
    """Формулировка приказа, не попавшая ни в один ряд, — дыра в правиле.

    Два несовпадения известны и оставлены: это обрывки ячеек, разорванных
    между страницами при выгрузке приложений, а не показатели.
    """
    uncovered = [t for t, _ in _measurable() if not groups_in_text(t)]
    assert len(uncovered) <= 2, uncovered


def test_abbreviation_and_full_form_land_in_one_group():
    """Приказ пишет показатель и сокращённо, и полностью — это один показатель."""
    pairs = [
        ("АД, ЧСС", "артериальное давление, частота сердечных сокращений"),
        ("ХС-ЛПНП", "холестерин-липопротеины низкой плотности"),
        ("Вес (ИМТ), окружность талии", "Вес (индекс массы тела), окружность талии"),
        ("пациентам при терапии варфарином - МНО",
         "пациентам при терапии варфарином - международное нормализованное отношение"),
        ("скорость клубочковой фильтрации", "с расчетом СКФ"),
        ("NT-proBNP", "уровень N-концевого пропептида натрийуретического гормона (B-типа)"),
    ]
    for short, full in pairs:
        assert groups_in_text(short) == groups_in_text(full), (short, full)


def test_free_text_of_a_doctor_matches_the_same_groups():
    """Ряд обязан ловить и то, как пишет врач, а не только формулировку приказа."""
    found = groups_in_text("Вес 82 кг, ИМТ 27,1. АД 130/80 мм рт.ст., пульс 72 в мин.")
    assert {"weight_bmi", "blood_pressure", "heart_rate"} <= found


def test_smoking_is_recognised_however_the_doctor_puts_it():
    """Приказ пишет «статус курения», врач — «не курит» или «некурящий»."""
    for text in ("статус курения", "Не курит", "курит 10 лет", "Некурящий",
                 "отрицает курение", "курильщик со стажем"):
        assert "smoking" in groups_in_text(text), text


# ── коды МКБ ──────────────────────────────────────────────────────────────────

def test_every_row_of_the_appendices_yields_codes():
    for row in _rows():
        assert _codes_of(row["код_мкб"]), row


def test_ranges_expand_to_every_rubric():
    assert _codes_of("I05 - I09") == ["I05", "I06", "I07", "I08", "I09"]


def test_subrubric_matches_its_rubric_range():
    """I25.2 попадает в диапазон I20 - I25 — сравнение идёт по префиксу."""
    assert required_groups("I25.2")


def test_diagnosis_outside_the_appendices_requires_nothing():
    """ОРВИ в перечнях диспансерного наблюдения нет."""
    assert required_groups("J06.9") == set()


def test_stenosis_row_requires_what_the_order_names():
    """I65.2: вес, талия, курение, АД, ЧСС, ХС-ЛПНП и дуплексное сканирование."""
    got = required_groups("I65.2")
    assert {"weight_bmi", "waist", "smoking", "blood_pressure",
            "heart_rate", "ldl", "bca_duplex"} <= got


def test_group_name_is_human_readable():
    assert group_name("blood_pressure") == "Артериальное давление"
