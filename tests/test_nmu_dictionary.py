"""Справочник 804н хранит факты приказа; тип визита выводит валидатор.

Смысл этих тестов — держать границу: в JSON не должно протекать наше
представление о типах визита, а в валидаторе не должно оказаться вида услуги
без сопоставления.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.formal_structure.validator import (
    _NMU_DOC,
    _NMU_KIND_TO_VISIT_TYPE,
    _NMU_VISIT_TYPES,
    VisitType,
)

_PATH = ROOT / "src" / "audit" / "formal_structure" / "nmu_services.json"


def test_file_holds_order_categories_not_our_visit_types():
    doc = json.loads(_PATH.read_text(encoding="utf-8"))
    assert set(doc["kinds"]) == set(_NMU_KIND_TO_VISIT_TYPE)
    for code, entry in doc["codes"].items():
        assert set(entry) == {"kind", "name"}, code
        assert entry["kind"] in doc["kinds"], code
        # Наши имена типов визита в справочник не просачиваются.
        assert entry["kind"] not in VisitType.__members__, code


def test_every_kind_has_a_visit_type():
    kinds = {entry["kind"] for entry in _NMU_DOC["codes"].values()}
    assert kinds <= set(_NMU_KIND_TO_VISIT_TYPE)


def test_dispensary_and_prophylactic_stay_distinct_in_the_file():
    """Приказ различает 168н/192н и 404н — различие обязано пережить генерацию.

    Пока обе категории сводятся в PROPHYLACTIC (docs/tech-debt.md), но факт
    хранится, чтобы разведение стоило одной строки в валидаторе.
    """
    codes = _NMU_DOC["codes"]
    assert codes["B04.047.001"]["kind"] == "dispensary"
    assert codes["B04.047.002"]["kind"] == "prophylactic"
    assert _NMU_VISIT_TYPES["B04.047.001"] is VisitType.PROPHYLACTIC
    assert _NMU_VISIT_TYPES["B04.047.002"] is VisitType.PROPHYLACTIC


def test_dictionary_covers_ordinary_clinic_appointments():
    codes = _NMU_DOC["codes"]
    for code, kind in [
        ("B01.047.001", "appointment_primary"),   # терапевт
        ("B01.047.002", "appointment_repeat"),
        ("B01.023.001", "appointment_primary"),   # невролог
        ("B01.015.001", "appointment_primary"),   # кардиолог
        ("B01.031.002", "appointment_repeat"),    # педиатр
    ]:
        assert codes[code]["kind"] == kind, code


def test_non_appointments_are_absent():
    """Освидетельствование, патронаж и ежедневный осмотр приёмом не считаются."""
    codes = _NMU_DOC["codes"]
    for code in ("B01.070.001", "B01.070.011", "B01.047.009", "B01.001.006"):
        assert code not in codes, code
