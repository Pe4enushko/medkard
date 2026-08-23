"""Нормализация упоминания препарата: что написал врач → запросы к реестру.

Строки взяты из прогона 2026-08-23 по боевым картам МДС — не выдуманы.
"""
from __future__ import annotations

import pytest

from grls.mention import search_candidates


@pytest.mark.parametrize(
    ("as_written", "expected"),
    [
        # Обычный случай: один кандидат, лишних обращений к базе нет.
        ("Цефиксим", ["цефиксим"]),
        ("Метронидазол", ["метронидазол"]),
        # Регистр и ® снимаются — это делает normalize_query.
        ("Клайра®", ["клайра"]),
        ("капотен", ["капотен"]),
        # Цифра — часть названия, а не доза. Правило «выкинуть токены с цифрами»
        # убило бы оба.
        ("Омега 3", ["омега 3"]),
        ("Мабелль-плюс", ["мабелль-плюс"]),
        # Форма выпуска и дозировка приклеены к названию.
        ("Далацин гель 1%", ["далацин", "далацин гель 1%"]),
        ("Скиноклир 15% гель", ["скиноклир", "скиноклир 15% гель"]),
        ("Липобейз крем", ["липобейз", "липобейз крем"]),
        ("Seveki НУФ крем №2", ["seveki нуф", "seveki нуф крем №2"]),
        # Вид изделия — не название препарата.
        ("ВМС «Мирена»", ["мирена", "вмс мирена"]),
    ],
)
def test_candidates_on_real_mentions(as_written: str, expected: list[str]) -> None:
    assert search_candidates(as_written) == expected


@pytest.mark.parametrize(
    ("as_written", "first", "second"),
    [
        ("Моксонидин (Физиотенз)", "моксонидин", "физиотенз"),
        ("Нитроглицерин (Нитроспрей)", "нитроглицерин", "нитроспрей"),
        ("Витамин Д (Вигантол)", "витамин д", "вигантол"),
        ("Адапален 0.1% крем (Дифферин)", "адапален", "дифферин"),
    ],
)
def test_parentheses_hold_a_second_name_not_junk(
    as_written: str, first: str, second: str
) -> None:
    """В скобках врач пишет второе имя того же препарата, а не мусор.

    Выбрасывать скобки нельзя: «Дифферин» ищется ничуть не хуже «адапалена», а
    на неполном реестре может оказаться единственным, что найдётся.
    """
    candidates = search_candidates(as_written)
    assert candidates[0] == first
    assert candidates[1] == second


def test_full_string_stays_last_for_the_contains_tier() -> None:
    """Исходная строка нужна слою «вхождение»: он ищет название реестра внутри
    запроса и потому переживает мусор вокруг названия."""
    candidates = search_candidates("Липобейз Биоактив пенка для умывания")
    assert candidates[0] == "липобейз биоактив"
    assert candidates[-1] == "липобейз биоактив пенка для умывания"


def test_no_candidates_for_a_string_without_a_name() -> None:
    assert search_candidates("1%") == []
    assert search_candidates("   ") == []


def test_candidates_are_unique() -> None:
    """Кандидаты идут в базу по очереди: дубль — лишний запрос на слабой машине."""
    for mention in ("Цефиксим", "Далацин гель 1%", "Моксонидин (Физиотенз)"):
        candidates = search_candidates(mention)
        assert len(candidates) == len(set(candidates))
