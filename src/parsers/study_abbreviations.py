"""
study_abbreviations.py — аббревиатуры исследований, которые не являются препаратами.

Врач пишет план обследования сокращениями: «Рекомендовано: 1. Контроль ЭКГ
2. КАК 3. ОАМ». Модель читает такой пункт как назначение лекарства, и на карте
ЦДЗ-00867066 (24.09.2026) выдала «в назначении 'КАК' отсутствуют наименование
препарата, дозировка, способ введения» — КАК это клинический анализ крови.
Просить модель этого не делать бесполезно: аббревиатуру от торгового названия
не отличить без справочника, а справочник есть только у кода.

Список лежит в ``resources/study_abbreviations.json``. В него идут только
сокращения, которых не может носить препарат: «АСК» (ацетилсалициловая кислота),
«Д3», «В12», «Mg B6» в нём быть не должны — иначе фильтр снимет замечание о
настоящем назначении.

Читают: ``audit/formal_structure/validator.py`` (снимает замечания о назначении,
у которых предмет — аббревиатура), ``LLM/graphs/diagnosis_nodes.py`` (не ищет
такие упоминания в ГРЛС).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "resources" / "study_abbreviations.json"

_INSIGNIFICANT = re.compile(r"[^0-9a-zа-яё]+")

_normalised: frozenset[str] | None = None


def _normalise(text: str) -> str:
    """«кл.ан.кр.» и «ЭХО-КГ» — одно и то же написание с точностью до разделителей."""
    return _INSIGNIFICANT.sub("", text.strip().casefold())


def _load(path: str | Path = _DEFAULT_PATH) -> frozenset[str]:
    global _normalised
    if _normalised is None:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)["abbreviations"]
        _normalised = frozenset(_normalise(item) for item in raw)
    return _normalised


def is_study_abbreviation(text: str) -> bool:
    """Обозначает ли текст исследование, а не лекарственный препарат.

    Сверяется целое значение: «КАК» — исследование, «КАК и Дона 1500 мг» — нет,
    потому что в такой строке препарат тоже назван.
    """
    normalised = _normalise(text)
    return bool(normalised) and normalised in _load()
