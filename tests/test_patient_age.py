"""Единый разбор возраста пациента из карты 1С (parsers.json_parser.patient_age)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.diagnosis.clinic_recs import _is_age_eligible
from parsers.json_parser import patient_age
from storage.models.guideline import Guideline


@pytest.mark.parametrize("patient,expected", [
    ({"AGE": 67}, 67),
    ({"AGE": "67"}, 67),
    # Детям до года 1С ставит 0. Прежнее `AGE or Возраст` считало это
    # отсутствием данных и возвращало None для всех младенцев.
    ({"AGE": 0}, 0),
    ({"AGE": "0"}, 0),
    ({"Возраст": 5}, 5),
    ({"AGE": None, "Возраст": 5}, 5),
    # Строгий разбор: непонятное значение — не возраст. Вызывающий обязан
    # трактовать None в сторону сужения набора правил.
    ({"AGE": "66 лет"}, None),
    ({"AGE": ""}, None),
    ({"AGE": "—"}, None),
    ({}, None),
])
def test_patient_age(patient, expected):
    assert patient_age(patient) == expected


def test_infant_still_matches_a_children_guideline():
    """AGE=0 не должен обесточивать возрастной фильтр клинреков."""
    children_only = Guideline(file_id="x", age_category=["Дети"])
    adults_only = Guideline(file_id="y", age_category=["Взрослые"])

    assert _is_age_eligible(children_only, patient_age({"AGE": 0})) is True
    assert _is_age_eligible(adults_only, patient_age({"AGE": 0})) is False


@pytest.mark.parametrize("age,expected_child", [(17, True), (18, False)])
def test_guideline_age_boundary_is_eighteen(age, expected_child):
    """Рубрикатор относит к детям 0–17; прежние 15 лет отсекали 16–17-летних."""
    children_only = Guideline(file_id="x", age_category=["Дети"])
    assert _is_age_eligible(children_only, age) is expected_child
