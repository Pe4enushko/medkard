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


def _manifest_guidelines() -> list[Guideline]:
    import csv

    with open(ROOT / "resources" / "manifest.csv", encoding="utf-8") as fh:
        return [
            Guideline(
                file_id=row["ID"],
                name=row["Наименование"],
                mkb=[c.strip().upper() for c in row["МКБ-10"].split(",") if c.strip()],
                age_category=[a.strip() for a in row["Возрастная категория"].split(",") if a.strip()],
            )
            for row in csv.DictReader(fh)
        ]


def test_children_get_their_own_slice_of_the_real_manifest():
    """Детские клинреки доезжают до чекеров, и это не тот же набор, что у взрослых.

    Возрастная категория приходит одной ячейкой манифеста («Взрослые, дети»),
    и если бы она не разбиралась на элементы, фильтр не узнал бы ни «дети», ни
    «взрослые» и пропускал бы всё подряд — детская карта получала бы взрослые
    рекомендации, а разницы в числах не было бы видно.
    """
    rows = _manifest_guidelines()
    for_child = [g for g in rows if _is_age_eligible(g, 5)]
    for_adult = [g for g in rows if _is_age_eligible(g, 44)]

    assert for_child, "детская карта осталась без клинреков"
    assert len(for_child) < len(rows)
    assert len(for_adult) < len(rows)
    assert {g.file_id for g in for_child} != {g.file_id for g in for_adult}

    child_only = [g for g in rows if _is_age_eligible(g, 5) and not _is_age_eligible(g, 44)]
    assert child_only, "нет ни одного клинрека только для детей — фильтр не различает"


def test_infant_is_filtered_as_a_child_not_as_unknown_age():
    """Детям до года ставят AGE 0. Falsy-разбор возраста отправлял бы им всё подряд."""
    from audit.diagnosis.clinic_recs import _patient_age

    rows = _manifest_guidelines()
    infant_age = _patient_age({"AGE": 0})
    assert infant_age == 0

    for_infant = {g.file_id for g in rows if _is_age_eligible(g, infant_age)}
    for_child = {g.file_id for g in rows if _is_age_eligible(g, 5)}
    assert for_infant == for_child
    assert len(for_infant) < len(rows)
