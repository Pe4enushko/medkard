from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.formal_structure.required_fields import missing_required_fields

_CORE = [
    "Температура", "ЧСС", "ЧД", "Состояние", "Сознание", "Ф20", "Кожные покровы",
    "Видимые слизистые", "Слизистые ротоглотки", "Миндалины", "Неврологический статус",
    "Опорно-двигательная система", "Сердечно-сосудистая система", "Органы брюшной полости",
    "Стул", "Мочеиспускание", "родственник лвн", "В выдаче листке нетрудоспособности",
]
_ANAMNESIS = ["Эпидемиологический анамнез", "Аллергологический анамнез", "Прививочный анамнез"]


def _inspection(labels):
    return [{"Параметр": label, "Значение": "заполнено"} for label in labels]


def test_primary_visit_without_anamnesis_reports_it():
    """Жалоба главврача: врач не заполнил анамнез, 1С поле не прислала,
    и в отчёте это ничем не отмечено."""
    labels = _CORE + _ANAMNESIS + ["Жалобы на момент осмотра", "Рекомендации и назначения:"]

    missing = missing_required_fields(_inspection(labels), {"primary"})

    assert missing == ["Анамнез заболевания"]


def test_repeat_visit_does_not_need_the_anamnesis_block():
    """На повторном приёме анамнез не переписывают: в 5 днях боевых карт он
    есть у 12-16% повторных против 89-100% первичных."""
    labels = _CORE + ["Рекомендации и назначения:"]

    assert missing_required_fields(_inspection(labels), {"repeat"}) == []


def test_another_template_is_left_alone():
    """Вакцинация, нефролог и «Заметки» — свои шаблоны; на 334 боевых картах
    базовый шаблон приносил либо все 18 полей ядра, либо не больше восьми."""
    labels = ["Жалобы прививочный", "Вакцинация", "объективно вакцинация", "Температура"]

    assert missing_required_fields(_inspection(labels), {"prophylactic"}) == []


def test_label_drift_is_not_a_missing_field():
    labels = list(_CORE) + _ANAMNESIS + [
        "Анамнез заболевания",
        "Жалобы на момент осмотра:",      # 1С иногда добавляет двоеточие
        "Рекомендации и назначения",      # а иногда не добавляет
    ]

    assert missing_required_fields(_inspection(labels), {"primary"}) == []


def test_empty_value_counts_as_missing():
    """Поле пришло, но пустое — для врача это то же самое, что его нет."""
    inspection = _inspection(_CORE + _ANAMNESIS + ["Жалобы на момент осмотра", "Рекомендации и назначения:"])
    inspection.append({"Параметр": "Анамнез заболевания", "Значение": "   "})

    assert missing_required_fields(inspection, {"primary"}) == ["Анамнез заболевания"]


def test_several_visit_types_require_only_what_all_of_them_require():
    labels = _CORE + ["Рекомендации и назначения:"]

    assert missing_required_fields(_inspection(labels), {"primary", "repeat"}) == []


def test_validator_turns_missing_fields_into_one_finding():
    from audit.formal_structure.validator import FormalValidator, VisitType

    visit = {"ДанныеОсмотра": _inspection(_CORE + _ANAMNESIS + ["Рекомендации и назначения:"])}

    finding = FormalValidator()._check_missing_required_fields(visit, {VisitType.PRIMARY})

    assert finding is not None
    assert finding["flag"] == "НЕЗАПОЛНЕНЫ_ПОЛЯ_ШАБЛОНА"
    assert "Анамнез заболевания" in finding["issue"]
    assert "Жалобы на момент осмотра" in finding["issue"]


def test_validator_stays_quiet_when_the_record_is_complete():
    from audit.formal_structure.validator import FormalValidator, VisitType

    visit = {"ДанныеОсмотра": _inspection(_CORE + ["Рекомендации и назначения:"])}

    assert FormalValidator()._check_missing_required_fields(visit, {VisitType.REPEAT}) is None


def test_one_slot_may_arrive_under_either_name():
    """«Рекомендации и назначения» и голая «Рекомендации» — один слот:
    в 10 боевых картах из 13 текст лежал во втором поле."""
    labels = _CORE + _ANAMNESIS + ["Жалобы на момент осмотра", "Анамнез заболевания", "Рекомендации"]

    assert missing_required_fields(_inspection(labels), {"primary"}) == []
