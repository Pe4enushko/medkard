"""Метрики прогона по выгруженному отчёту: разбор колонок и сами доли."""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "report_metrics", ROOT / "scripts" / "report-metrics.py"
)
metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metrics)

_OLD_HEADERS = [
    "Специализация", "Дата приема", "Данные карты", "Данные осмотра", "Услуги",
    "Диагнозы", "Проверка по приказам МЗ", "Проверка по клин.рекоммендациям",
    "Проверка кодирования МКБ",
]
# Раскладка отчётов, выгруженных с 20.08 по 26.08.2026: между этими датами в
# шаблоне жила колонка «Источники КР», сдвигавшая проверку МКБ вправо. Её убрали
# (дублировала хвост колонки H), но файлы той недели лежат у клиник, и метрики
# обязаны читать их наравне с нынешними.
_NEW_HEADERS = _OLD_HEADERS[:8] + ["Источники КР", "Проверка кодирования МКБ"]

_CARD = "Пациент:\n  AGE: 3\n  GENDER: Мужской\n\nВрач:\n  SPECIALIZATION: Педиатр\n\nПрием:\n  DATE: 24.08.2026\n  GUID: g-1"
_SERVICES = "1.\n  КодЕГИСЗ: B01.031.001\n  Наименование: Прием (осмотр, консультация) врача-педиатра первичный"
_INSPECTION = (
    "1.\n  Значение: кашель\n  Параметр: Жалобы на момент осмотра\n"
    "2.\n  Значение: в мин.\n  Параметр: ЧСС"
)


def _row(headers, **cells):
    return tuple(cells.get(name, "") for name in headers)


def test_columns_are_found_by_name_not_by_position():
    """«Источники КР» появилась между прогонами, сдвинула проверку МКБ вправо, а
    через неделю исчезла: по номеру колонки метрика читала бы соседнюю ячейку —
    и на файлах той недели, и на нынешних."""
    old = _row(_OLD_HEADERS, **{"Проверка кодирования МКБ": "[ОШИБКА КОДИРОВАНИЯ МКБ]"})
    new = _row(_NEW_HEADERS, **{"Проверка кодирования МКБ": "[РЕКОМЕНДАЦИЯ ПО КОДУ МКБ]"})

    assert metrics._code_markers(_OLD_HEADERS, [old])["icd_label"] == "error"
    assert metrics._code_markers(_NEW_HEADERS, [new])["icd_label"] == "recommendation"


def test_a_missing_column_reads_as_empty_not_as_a_crash():
    assert metrics._cell(_OLD_HEADERS, _row(_OLD_HEADERS), "Источники КР") == ""


def test_repeated_flag_in_one_card_is_counted():
    """Повтор одного флага в карте невозможен с тех пор, как правило даёт
    один вердикт вместо массива замечаний — по нему отличается ревизия."""
    formal = "[ОТСУТСТВУЕТ_ОБЪЕКТИВНЫЙ_ОСМОТР] раз\n[ОТСУТСТВУЕТ_ОБЪЕКТИВНЫЙ_ОСМОТР] два"
    rows = [_row(_NEW_HEADERS, **{"Проверка по приказам МЗ": formal})]

    assert metrics._code_markers(_NEW_HEADERS, rows)["cards_with_a_repeated_flag"] == 1


def test_inspection_fields_keep_report_order():
    assert metrics._inspection_fields(_INSPECTION) == [
        ("Жалобы на момент осмотра", "кашель"),
        ("ЧСС", "в мин."),
    ]


def test_visit_is_restored_well_enough_for_the_validator():
    row = _row(_NEW_HEADERS, **{
        "Данные карты": _CARD, "Услуги": _SERVICES, "Данные осмотра": _INSPECTION,
        "Диагнозы": "1.\n  КодМКБ: J06.9",
    })

    visit = metrics._visit(_NEW_HEADERS, row)

    assert visit["Пациент"]["AGE"] == 3
    assert visit["Услуги"][0]["КодЕГИСЗ"] == "B01.031.001"
    assert visit["Диагнозы"] == [{"КодМКБ": "J06.9"}]
    assert visit["ДанныеОсмотра"][1] == {"Параметр": "ЧСС", "Значение": "в мин."}


def test_a_finding_counts_as_naming_a_field_only_if_it_quotes_one():
    labels = ["Жалобы на момент осмотра", "ЧСС"]

    assert metrics._names_a_field("В поле «Жалобы на момент осмотра» пусто", labels)
    assert not metrics._names_a_field("Жалобы не собраны", labels)


def test_metrics_count_shares_over_a_two_card_report():
    formal_named = "[ОБНАРУЖЕНЫ_ЗАГЛУШКИ] В параметре ЧСС стоит «в мин.»\n    [Наблюдения]: ЧСС: в мин.\n    [Источник: 274n]"
    formal_vague = "[ОРФОГРАФИЧЕСКИЕ_ОШИБКИ] Опечатка в тексте\n    [Источник: 274n]"
    rows = [
        _row(_NEW_HEADERS, **{
            "Данные карты": _CARD, "Услуги": _SERVICES, "Данные осмотра": _INSPECTION,
            "Проверка по приказам МЗ": formal_named,
        }),
        _row(_NEW_HEADERS, **{
            "Данные карты": _CARD, "Услуги": _SERVICES, "Данные осмотра": _INSPECTION,
            "Проверка по приказам МЗ": formal_vague,
        }),
    ]

    out = asyncio.run(metrics._metrics(_NEW_HEADERS, rows))

    assert out["cards"] == 2
    assert out["field_named"]["formal_issue"] == {"named": 1, "total": 2}
    # заглушка «в мин.» есть в обеих картах, флаг — только в первой
    assert out["placeholders"] == {"cards": 2, "flagged": 1}
    assert out["flags"]["ОБНАРУЖЕНЫ_ЗАГЛУШКИ"] == 1


def test_a_short_field_name_still_counts_but_only_as_a_whole_word():
    labels = ["ЧСС", "Ф20", "Рекомендации и назначения:"]

    assert metrics._names_a_field("В параметре ЧСС стоит «в мин.»", labels)
    assert metrics._names_a_field("поле «Рекомендации и назначения» пусто", labels)
    # подстрока внутри чужого слова ссылкой на поле не является
    assert not metrics._names_a_field("осмотр проведён", ["ЧС", "мот"])
