"""Уровни сопоставления с реестром — на реалистичных парах МНН, без базы."""
from __future__ import annotations

import pytest

from grls.match import (FUZZY_THRESHOLD, MatchKind, classify, contains,
                        discriminator_tokens, discriminators_agree)


@pytest.mark.parametrize("query,candidate", [
    ("Метопролол", "Метопролола сукцинат"),
    ("Амлодипин", "Амлодипин + Аторвастатин"),
    ("Амоксициллин", "Амоксициллин + Клавулановая кислота"),
    ("Левотироксин натрия", "Левотироксин"),
    ("Инсулин", "Инсулин гларгин"),
    ("Тиамин", "Тиамина хлорид"),
])
def test_added_or_dropped_word_is_the_same_medicine(query, candidate):
    """Солевые формы и комбинации — то же вещество, и триграммы их теряли.

    «Метопролол» против «Метопролола сукцинат» — это 0.455, ниже порога 0.6:
    поиск по МНН оставался пустым, запрос проваливался в торговые наименования,
    а оттуда — в реестр БАД.
    """
    assert classify(query, candidate) is MatchKind.CONTAINS


@pytest.mark.parametrize("query,candidate", [
    ("Витамин D", "Витамин Е"),
    ("Витамин В12", "Витамин В6"),
    ("Витамин В1", "Витамин В12"),
    ("Витамин В6", "Витамин В1"),
])
def test_replaced_discriminator_is_another_medicine(query, candidate):
    """Различитель заменён — это другой препарат, как бы ни были похожи строки.

    «Витамин В1» против «Витамин В12» — 0.750, и прежняя охрана его пропускала:
    она искала токен подстрокой, а «в1» лежит внутри «в12». Тиамин находился
    как цианокобаламин.
    """
    assert classify(query, candidate) is None


def test_similar_spelling_stays_a_guess():
    """Похоже — не значит то же. Такое попадание обязано быть помечено.

    Разводить эти пары порогом бессмысленно: правильный «Левотироксин» (0.650)
    лежит ниже неправильного «Преднизона» (0.692).
    """
    assert classify("Преднизолон", "Преднизон") is MatchKind.FUZZY


def test_extended_stem_is_reported_as_containment_not_identity():
    """Известная цена вхождения: «Эналаприл» ⊂ «Эналаприлат», а это разные МНН.

    Отличить это от «Метопролол» ⊂ «Метопролола сукцинат» без словаря веществ
    нельзя, поэтому уровень не скрывается: ответ говорит «входит в МНН
    „Эналаприлат“» и показывает найденное название, а не утверждает тождество.
    """
    assert classify("Эналаприл", "Эналаприлат") is MatchKind.CONTAINS


def test_numbered_form_is_not_the_bare_vitamin():
    """«Витамин D» и «Витамин D3» — разные записи реестра, и различитель разный.

    Осознанный перекос в осторожность: запрос уйдёт искать по торговым
    наименованиям, а не получит чужую регистрацию под видом своей.
    """
    assert classify("Витамин D", "Витамин D3") is None
    assert classify("Витамин D3", "Витамин D") is None


def test_exact_match_wins_over_everything():
    assert classify("Эналаприл", "эналаприл") is MatchKind.EXACT
    assert classify("  ЭНАЛАПРИЛ  ", "«Эналаприл»") is MatchKind.EXACT


@pytest.mark.parametrize("query,candidate", [
    ("Цефтриаксон", "Цефотаксим"),
    ("Кальция глюконат", "Кальция глицерофосфат"),
    ("Инсулин гларгин", "Инсулин глулизин"),
])
def test_unrelated_medicines_do_not_match(query, candidate):
    assert classify(query, candidate) is None


def test_discriminators_are_short_or_numbered_tokens():
    assert discriminator_tokens("Витамин D") == ["d"]
    assert discriminator_tokens("Витамин D3") == ["d3"]
    assert discriminator_tokens("Витамин В12") == ["в12"]
    assert discriminator_tokens("амоксициллин + клавулановая кислота") == []


def test_discriminator_must_be_a_whole_word():
    """Сравнение подстрокой — то, из-за чего «в1» пролезало в «в12»."""
    assert discriminators_agree("Витамин В1", "Витамин В1") is True
    assert discriminators_agree("Витамин В1", "Витамин В12") is False


def test_containment_ignores_fragments():
    """Обрывок короче основы не должен цеплять пол-реестра."""
    assert contains("Инсулин", "Инсулин гларгин") is True
    assert contains("ин", "Инсулин") is False
    assert contains("", "Инсулин") is False


def test_empty_query_matches_nothing():
    assert classify("", "Инсулин") is None
    assert classify("~", "Инсулин") is None


def test_fuzzy_threshold_is_the_documented_one():
    assert FUZZY_THRESHOLD == 0.6
