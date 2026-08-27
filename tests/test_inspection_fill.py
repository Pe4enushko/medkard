"""Дозаполнение пустых полей шаблона в отчёте.

1С не присылает поле, которое врач не заполнил: его нет в записи ни пустым
значением, ни ключом. Отчёт поэтому дорисовывает недостающие поля шаблона сам —
иначе врач не отличит «поле не заполнено» от «такого поля в шаблоне нет».
"""

import json

import pytest

from parsers.inspection_fill import PLACEHOLDER, fill_missing_fields, match_format
from parsers.inspection_order import load_inspection_formats


def _field(label, value="x"):
    return {"Значение": value, "Параметр": label}


def _labels(items):
    return [item["Параметр"] for item in items]


def _formats(tmp_path, data):
    path = tmp_path / "inspection_formats.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return load_inspection_formats("Alenka", path=path)


_SIMPLE = {
    "Alenka": {
        "standard": {
            "order": ["жалобы", "анамнез", "температура, чсс"],
            "signature": ["жалобы", "температура"],
            "min_signature_match": 2,
        }
    }
}


# ── опознание шаблона ────────────────────────────────────────────────────────


def test_record_below_the_threshold_matches_nothing(tmp_path):
    formats = _formats(tmp_path, _SIMPLE)
    assert match_format([_field("жалобы")], formats) is None


def test_record_at_the_threshold_matches(tmp_path):
    formats = _formats(tmp_path, _SIMPLE)
    matched = match_format([_field("жалобы"), _field("температура")], formats)
    assert matched is not None and matched.name == "standard"


def test_first_matching_format_wins(tmp_path):
    """Порядок форматов в файле — это приоритет опознания: у 17 боевых карт из
    261 прививочные поля приписаны поверх полного базового осмотра, и такая
    карта обязана остаться базовой."""
    formats = _formats(
        tmp_path,
        {
            "Alenka": {
                "standard": {
                    "order": ["жалобы", "анамнез"],
                    "signature": ["жалобы", "анамнез"],
                    "min_signature_match": 2,
                },
                "vaccination": {
                    "order": ["прививка"],
                    "signature": ["прививка"],
                    "min_signature_match": 1,
                },
            }
        },
    )
    both = [_field("жалобы"), _field("анамнез"), _field("прививка")]
    assert match_format(both, formats).name == "standard"


def test_format_without_a_signature_is_not_recognised(tmp_path):
    formats = _formats(tmp_path, {"Alenka": {"standard": ["жалобы", "анамнез"]}})
    assert formats == []


# ── дозаполнение ─────────────────────────────────────────────────────────────


def test_missing_fields_are_added_in_template_order(tmp_path):
    formats = _formats(tmp_path, _SIMPLE)
    out = fill_missing_fields([_field("чсс", "80"), _field("жалобы", "кашель")], formats[0])
    assert _labels(out) == ["жалобы", "анамнез", "температура", "чсс"]
    assert [item["Значение"] for item in out] == ["кашель", PLACEHOLDER, PLACEHOLDER, "80"]


def test_a_slot_that_came_under_another_name_is_not_redrawn(tmp_path):
    """«На приеме пациент с» 1С шлёт как «родственник лвн» — дорисовать его
    пустым значило бы показать врачу дефект в заполненном поле."""
    formats = _formats(
        tmp_path,
        {
            "Alenka": {
                "standard": {
                    "order": [["на приеме пациент с", "родственник лвн"], "жалобы"],
                    "signature": ["жалобы"],
                    "min_signature_match": 1,
                }
            }
        },
    )
    out = fill_missing_fields([_field("родственник лвн", "мамой"), _field("жалобы")], formats[0])
    assert _labels(out) == ["родственник лвн", "жалобы"]
    assert PLACEHOLDER not in [item["Значение"] for item in out]


def test_a_missing_multi_name_slot_is_drawn_under_the_first_name(tmp_path):
    formats = _formats(
        tmp_path,
        {
            "Alenka": {
                "standard": {
                    "order": [["на приеме пациент с", "родственник лвн"], "жалобы"],
                    "signature": ["жалобы"],
                    "min_signature_match": 1,
                }
            }
        },
    )
    out = fill_missing_fields([_field("жалобы")], formats[0])
    assert _labels(out) == ["на приеме пациент с", "жалобы"]


def test_fields_outside_the_template_keep_their_order_at_the_tail(tmp_path):
    formats = _formats(tmp_path, _SIMPLE)
    data = [_field("Динамика заболевания"), _field("жалобы"), _field("Заметки")]
    out = fill_missing_fields(data, formats[0])
    assert _labels(out)[-2:] == ["Динамика заболевания", "Заметки"]


def test_nothing_is_lost_or_duplicated(tmp_path):
    formats = _formats(tmp_path, _SIMPLE)
    data = [_field("чсс"), _field("Заметки"), _field("анамнез")]
    out = fill_missing_fields(data, formats[0])
    assert [item for item in data if item in out] == data
    assert len(out) == 4 + 1  # 4 слота шаблона + поле вне шаблона


def test_the_drawn_field_repeats_the_key_order_of_the_record(tmp_path):
    """В отчёте блок поля печатается ключ за ключом; дорисованное поле не должно
    выделяться перевёрнутым порядком «Параметр/Значение»."""
    formats = _formats(tmp_path, _SIMPLE)
    out = fill_missing_fields([{"Параметр": "жалобы", "Значение": "кашель"}], formats[0])
    drawn = next(item for item in out if item["Значение"] == PLACEHOLDER)
    assert list(drawn) == ["Параметр", "Значение"]

    out = fill_missing_fields([{"Значение": "кашель", "Параметр": "жалобы"}], formats[0])
    drawn = next(item for item in out if item["Значение"] == PLACEHOLDER)
    assert list(drawn) == ["Значение", "Параметр"]


def test_an_empty_record_is_left_alone(tmp_path):
    formats = _formats(tmp_path, _SIMPLE)
    assert fill_missing_fields([], formats[0]) == []


# ── боевой конфиг Алёнки ─────────────────────────────────────────────────────


def test_alenka_standard_card_gets_the_dead_slots_drawn():
    """«Пациент нуждается в уходе» и «Диагноз» не пришли ни в одной из 334 карт
    (диагноз едет отдельным блоком карты). Решение главврача — рисовать шаблон
    как есть, поэтому в отчёте они появляются пустыми у каждой карты."""
    formats = load_inspection_formats("Alenka")
    card = [
        _field(label)
        for label in (
            "Температура", "ЧСС", "ЧД", "Состояние", "Сознание", "Ф20", "Кожные покровы",
            "Видимые слизистые", "Слизистые ротоглотки", "Миндалины", "Неврологический статус",
            "Опорно-двигательная система", "Сердечно-сосудистая система",
            "Органы брюшной полости", "Стул", "Мочеиспускание",
        )
    ]
    matched = match_format(card, formats)
    assert matched.name == "standard"

    out = fill_missing_fields(card, matched)
    drawn = [item["Параметр"] for item in out if item["Значение"] == PLACEHOLDER]
    assert "Пациент нуждается в уходе" in drawn
    assert "Диагноз" in drawn
    assert _labels(out)[0] == "На приеме пациент с"


def test_alenka_vaccination_card_is_not_filled_by_the_standard_template():
    """Чужой шаблон: базовый набор к нему не применяется, иначе карта получит
    два десятка пустых строк. Опознаётся своими полями, дорисовывается своим
    ядром."""
    formats = load_inspection_formats("Alenka")
    card = [
        _field(label)
        for label in (
            "Эпидемиологический анамнез", "Температура", "ЧСС", "ЧД", "Ф20", "Стул",
            "объективно вакцинация", "Жалобы прививочный", "Кожные прививочные",
            "Лимфатические узлы", "Аускультативно дыхание", "Комментарий к вакцинации",
            "Предоставленные документы",
        )
    ]
    matched = match_format(card, formats)
    assert matched.name == "vaccination"

    drawn = [
        item["Параметр"]
        for item in fill_missing_fields(card, matched)
        if item["Значение"] == PLACEHOLDER
    ]
    assert drawn == ["Прививочный анамнез"]


def test_alenka_foreign_card_matches_no_format():
    """Туберкулинодиагностика — 16 карт из одного поля; нефролог — своё ядро.
    Ни то, ни другое дорисовывать нечем."""
    formats = load_inspection_formats("Alenka")
    assert match_format([_field("Комментарий к вакцинации")], formats) is None
    assert match_format(
        [
            _field("Жалобы нефролог"),
            _field("Объективный статус нефролог"),
            _field("An. morbi"),
            _field("родственник лвн"),
            _field("Рекомендации"),
        ],
        formats,
    ) is None


@pytest.mark.parametrize("clinic", ["MDS", "нет такой"])
def test_a_clinic_without_formats_raises(clinic):
    with pytest.raises(ValueError):
        load_inspection_formats(clinic)
