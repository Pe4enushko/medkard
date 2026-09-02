"""Контролируемые показатели 168н: что требуется по коду МКБ и что нашлось в записи.

Приказ 168н — единственный найденный источник требований, привязанных к
**диагнозу**, а не к виду приёма. Приложения 1-3 перечисляют по каждому коду
МКБ колонку «Контролируемые показатели состояния здоровья».

Два файла в ``resources/``:

* ``168n_appendices.csv`` — 102 строки перечней, как в приказе;
* ``168n_indicators.json`` — синонимические ряды. Нужны потому, что приказ
  пишет один и тот же показатель и сокращённо, и полностью: «АД, ЧСС» и
  «артериальное давление, частота сердечных сокращений», «ХС-ЛПНП» и
  «холестерин-липопротеины низкой плотности». Врач в записи волен написать
  третьим способом — ряды покрывают и это.

Чего здесь нет: показателей вида «отсутствие данных о ЗНО по результатам
биопсии». Это заключение врача, а не поле; проверить его сравнением нельзя.
Из 156 формулировок приказа таких 84 — почти весь третий перечень.
"""

from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path

_RESOURCES = Path(__file__).resolve().parents[3] / "resources"
_APPENDICES = _RESOURCES / "168n_appendices.csv"
_INDICATORS = _RESOURCES / "168n_indicators.json"

# Коды в перечнях записаны и поштучно, и диапазонами: «I05 - I09», «I10 - I15».
_RANGE_RE = re.compile(r"([A-ZА-Я]\d{2})\s*-\s*([A-ZА-Я]\d{2})")
_CODE_RE = re.compile(r"[A-ZА-Я]\d{2}(?:\.\d+)?")


@lru_cache(maxsize=1)
def _groups() -> tuple[tuple[str, str, tuple[re.Pattern[str], ...]], ...]:
    doc = json.loads(_INDICATORS.read_text(encoding="utf-8"))
    return tuple(
        (g["id"], g["name"], tuple(re.compile(p, re.I) for p in g["patterns"]))
        for g in doc["groups"]
    )


def groups_in_text(text: str) -> set[str]:
    """Идентификаторы рядов, чьи формулировки встретились в тексте."""
    return {gid for gid, _, pats in _groups() if any(p.search(text) for p in pats)}


def group_name(group_id: str) -> str:
    for gid, name, _ in _groups():
        if gid == group_id:
            return name
    return group_id


@lru_cache(maxsize=1)
def _rows() -> tuple[dict[str, str], ...]:
    with open(_APPENDICES, encoding="utf-8", newline="") as f:
        return tuple(csv.DictReader(f))


def _codes_of(cell: str) -> list[str]:
    """Коды строки перечня, диапазоны раскрываются в рубрики-концы.

    Диапазон «I05 - I09» разворачивается в I05, I06, I07, I08, I09 — сравнение
    потом идёт по префиксу, поэтому подрубрики вроде I05.1 совпадут сами.
    """
    codes: list[str] = []
    rest = cell
    for start, end in _RANGE_RE.findall(cell):
        letter, lo, hi = start[0], int(start[1:]), int(end[1:])
        codes += [f"{letter}{n:02d}" for n in range(lo, hi + 1)]
    rest = _RANGE_RE.sub(" ", rest)
    codes += _CODE_RE.findall(rest)
    return codes


def required_groups(icd_code: str) -> set[str]:
    """Ряды показателей, которых 168н требует при этом коде МКБ.

    Пункт 9, последний абзац: при нескольких заболеваниях перечень «должен
    включать все параметры, соответствующие каждому заболеванию», — поэтому
    вызывающий берёт объединение по всем диагнозам карты, а не первый код.
    """
    code = icd_code.strip().upper()
    if not code:
        return set()
    required: set[str] = set()
    for row in _rows():
        if any(code.startswith(c) for c in _codes_of(row["код_мкб"])):
            required |= groups_in_text(row["контролируемые_показатели"])
    return required
