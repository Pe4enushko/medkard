"""Юнит-тесты storage.models.guideline.Guideline — чистые функции, без БД."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from storage.models.guideline import Guideline


def _row(**overrides) -> dict[str, str]:
    base = {
        "ID": "581_2",
        "Наименование": "Острый бронхит",
        "МКБ-10": "J20.0, J20.1",
        "Возрастная категория": "Взрослые, дети",
        "Разработчик": "Минздрав",
        "Статус одобрения НПС": "Одобрено",
        "Дата размещения": "01.01.2020",
        "Статус применения": "Действует",
    }
    base.update(overrides)
    return base


def test_from_manifest_row_splits_mkb_into_upper_list():
    assert Guideline.from_manifest_row(_row()).mkb == ["J20.0", "J20.1"]


def test_from_manifest_row_uppercases_and_strips_mkb():
    g = Guideline.from_manifest_row(_row(**{"МКБ-10": " j20.0 ,j20.1 "}))
    assert g.mkb == ["J20.0", "J20.1"]


def test_from_manifest_row_splits_age_category_verbatim():
    assert Guideline.from_manifest_row(_row()).age_category == ["Взрослые", "дети"]


def test_from_manifest_row_single_values():
    g = Guideline.from_manifest_row(_row(**{"МКБ-10": "A15", "Возрастная категория": "Дети"}))
    assert g.mkb == ["A15"]
    assert g.age_category == ["Дети"]


def test_from_manifest_row_empty_cells_become_empty_lists():
    g = Guideline.from_manifest_row(_row(**{"МКБ-10": "", "Возрастная категория": ""}))
    assert g.mkb == []
    assert g.age_category == []


def test_from_manifest_row_maps_all_scalar_fields():
    g = Guideline.from_manifest_row(_row())
    assert g.file_id == "581_2"
    assert g.name == "Острый бронхит"
    assert g.developer == "Минздрав"
    assert g.nps_status == "Одобрено"
    assert g.published_at == "01.01.2020"
    assert g.usage_status == "Действует"
