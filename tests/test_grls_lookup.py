"""Какой запрос до какого реестра доходит."""
from __future__ import annotations

from grls.lookup import _supplement_queries
from grls.mention import search_candidates


def test_abbreviation_does_not_reach_the_supplement_registry() -> None:
    """«АСК (кардиомиагнил 75мг)» с прогона 2026-08-23.

    В ГРЛС не нашлось ничего: врач переставил буквы в «Кардиомагниле». Дальше
    запрос «аск» вытащил из реестра БАД польский коллаген для спортсменов —
    в конце его названия стоит обрубленное «L-аск», — и судья получил карточку
    коллагена как справку об ацетилсалициловой кислоте.
    """
    candidates = search_candidates("АСК (кардиомиагнил 75мг)")

    assert candidates[0] == "аск"
    assert "аск" not in _supplement_queries(candidates)
    assert "кардиомиагнил" in _supplement_queries(candidates)


def test_real_supplement_names_still_get_through() -> None:
    """Порог отсекает аббревиатуры, а не БАДы: их названия длинные."""
    for mention in ("Омега 3", "Вит Д", "Когнитив комплекс", "Армолипид"):
        assert _supplement_queries(search_candidates(mention))


def test_nothing_left_means_the_registry_is_not_asked_at_all() -> None:
    """Пустой список — цикл поиска просто не выполнится ни разу."""
    assert _supplement_queries(["кок", "вмс"]) == []
