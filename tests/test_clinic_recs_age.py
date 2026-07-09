"""Юнит-тесты clinic_recs._is_age_eligible (по Guideline)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.diagnosis.clinic_recs import _is_age_eligible
from storage.models.guideline import Guideline


def _g(age_category: list[str]) -> Guideline:
    return Guideline(file_id="x", age_category=age_category)


@pytest.mark.parametrize("age,cats,expected", [
    (None, ["Дети"], True),
    (10, ["Дети"], True),
    (30, ["Дети"], False),
    (30, ["Взрослые"], True),
    (10, ["Взрослые"], False),
    (10, ["Взрослые", "Дети"], True),
    (30, ["Взрослые", "дети"], True),
    (10, [], True),
    (10, ["дети"], True),
])
def test_is_age_eligible(age, cats, expected):
    assert _is_age_eligible(_g(cats), age) is expected
