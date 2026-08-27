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


def test_never_drawn_slot_keeps_its_place_but_is_not_drawn(tmp_path):
    """Поле из never_drawn упорядочивается как обычно, но пустым не рисуется."""
    formats = _formats(
        tmp_path,
        {
            "Alenka": {
                "standard": {
                    "order": ["жалобы", "диагноз", "анамнез"],
                    "never_drawn": ["диагноз"],
                    "signature": ["жалобы"],
                    "min_signature_match": 1,
                }
            }
        },
    )
    assert _labels(fill_missing_fields([_field("жалобы")], formats[0])) == ["жалобы", "анамнез"]
    # пришёл — встал на своё место шаблона
    out = fill_missing_fields([_field("анамнез"), _field("диагноз", "J06.9")], formats[0])
    assert _labels(out) == ["жалобы", "диагноз", "анамнез"]
    assert out[1]["Значение"] == "J06.9"


def test_alenka_standard_card_draws_care_but_not_diagnosis():
    """«Пациент нуждается в уходе» и «Диагноз» не пришли ни в одной из 334 карт.
    Диагноз при этом в карте есть всегда — своим блоком, поэтому пустым он не
    рисуется; поля про уход в выгрузке нет вовсе, и оно рисуется."""
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
    assert "Диагноз" not in drawn
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


def test_alenka_specialist_cards_get_their_own_templates():
    """У нефролога и гастроэнтеролога свои шаблоны: базовый закрывает у них 4 и 17
    слотов из 37, натянуть его значило бы выдать 19–32 пустые строки на карту."""
    formats = load_inspection_formats("Alenka")

    nephrologist = [
        _field(label)
        for label in ("родственник лвн", "Рекомендации", "Наследственный анамнез",
                      "Жалобы нефролог", "An. morbi", "Объективный статус нефролог", "an_vitae")
    ]
    matched = match_format(nephrologist, formats)
    assert matched.name == "nephrologist"
    out = fill_missing_fields(nephrologist, matched)
    # карта полная — дорисовывать нечего, поменялся только порядок
    assert PLACEHOLDER not in [item["Значение"] for item in out]
    assert _labels(out)[:3] == ["родственник лвн", "Жалобы нефролог", "An. morbi"]

    gastro = [
        _field(label)
        for label in ("Status praesens аллерголог", "Жалобы на момент обращения",
                      "Мочевыделительная система", "Живот", "Язык", "Селезенка")
    ]
    assert match_format(gastro, formats).name == "gastroenterologist"


def test_alenka_card_without_a_template_is_left_alone():
    """Туберкулинодиагностика — 16 карт из одного поля, карты из одних «Заметок» —
    ещё 15. Шаблона у них нет, дорисовывать нечем."""
    formats = load_inspection_formats("Alenka")
    assert match_format([_field("Комментарий к вакцинации")], formats) is None
    assert match_format([_field("Заметки")], formats) is None


def test_all_alenka_templates_share_one_order():
    """Врач читает отчёт по всем приёмам подряд: жалобы, анамнез, витальные, осмотр,
    рекомендации должны идти в одном и том же порядке в каждом шаблоне клиники."""
    spine = ["Жалоб", "анамнез", "Температура", "Стул", "Рекомендаци"]
    for fmt in load_inspection_formats("Alenka"):
        names = [slot[0] for slot in fmt.slots]
        positions = [
            next((i for i, name in enumerate(names) if part.lower() in name.lower()), None)
            for part in spine
        ]
        seen = [p for p in positions if p is not None]
        assert seen == sorted(seen), f"{fmt.name}: {names}"


def test_a_field_name_with_a_comma_stays_one_slot():
    """Запятая в строке манифеста делит её на поля, поэтому имя с запятой внутри
    задаётся списком: «При температуре 37,5  С прием» — одно поле, а не два."""
    vaccination = next(f for f in load_inspection_formats("Alenka") if f.name == "vaccination")
    assert ("При температуре 37,5  С прием",) in vaccination.slots


@pytest.mark.parametrize("clinic", ["MDS", "нет такой"])
def test_a_clinic_without_formats_raises(clinic):
    with pytest.raises(ValueError):
        load_inspection_formats(clinic)
