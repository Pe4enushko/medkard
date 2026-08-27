"""
inspection_fill.py — дорисовка полей шаблона, которых нет в записи.

1С не присылает поле, которое врач не заполнил: незаполненного анамнеза в
записи не «пусто», его там нет вовсе (по 334 боевым картам Алёнки поле «пришло»
и поле «пришло непустым» совпадают до единицы). Читающий отчёт врач поэтому не
отличает «поле не заполнено» от «такого поля в шаблоне нет» — отличие знает
только шаблон, и здесь оно в отчёт и возвращается: недостающие поля встают на
свои места в порядке шаблона со значением «не заполнено».

Шаблон опознаётся по полям самой записи, а не по клинике и не по коду услуги:
одна клиника присылает несколько шаблонов (осмотр, вакцинация, нефролог), и
базовый педиатрический приходит и первичным приёмом, и повторным, и
профилактическим. Запись, не совпавшую ни с одним шаблоном, дорисовывать нечем,
и она остаётся как есть.
"""

from __future__ import annotations

from typing import Any

from parsers.inspection_order import (
    InspectionFormat,
    claim_nearest,
    labels_match,
    normalized_labels,
)

PLACEHOLDER = "не заполнено"

_PARAM_KEY = "Параметр"
_VALUE_KEY = "Значение"


def match_format(
    inspection_data: list[dict[str, Any]],
    formats: list[InspectionFormat],
) -> InspectionFormat | None:
    """Шаблон, которым заполнена запись, или ``None``.

    Форматы перебираются в порядке манифеста: он же порядок приоритета. У 17
    боевых карт из 261 прививочные поля приписаны поверх полного базового
    осмотра, и такая карта обязана остаться базовой, а не стать вакцинацией.
    """
    labels = [
        str(item.get(_PARAM_KEY, ""))
        for item in inspection_data
        if isinstance(item, dict)
    ]
    for fmt in formats:
        hits = sum(
            any(labels_match(label, name) for label in labels) for name in fmt.signature
        )
        if hits >= fmt.min_signature_match:
            return fmt
    return None


def fill_missing_fields(
    inspection_data: list[dict[str, Any]],
    fmt: InspectionFormat,
    *,
    placeholder: str = PLACEHOLDER,
    max_distance: int = 2,
) -> list[dict[str, Any]]:
    """Запись в порядке шаблона, где недостающие поля дорисованы *placeholder*.

    Поля, которых в шаблоне нет, сохраняют свой относительный порядок и уходят
    в хвост — как и при обычном переупорядочивании. Ничего не теряется и не
    дублируется: пришедшие элементы попадают в результат теми же объектами.

    Слоты из ``fmt.never_drawn`` пропускаются: место в порядке у них есть, а
    пустой строки быть не должно.
    """
    if not inspection_data:
        return list(inspection_data)

    norm_params = normalized_labels(inspection_data)
    claimed = [False] * len(inspection_data)
    result: list[dict[str, Any]] = []

    for slot in fmt.slots:
        idx = claim_nearest(slot, norm_params, claimed, max_distance=max_distance)
        if idx >= 0:
            claimed[idx] = True
            result.append(inspection_data[idx])
        elif not any(labels_match(slot[0], name) for name in fmt.never_drawn):
            result.append(_drawn(slot[0], inspection_data, placeholder))

    for idx, item in enumerate(inspection_data):
        if not claimed[idx]:
            result.append(item)

    return result


def _drawn(label: str, inspection_data: list[dict[str, Any]], placeholder: str) -> dict[str, str]:
    """Дорисованное поле — с тем же порядком ключей, что у пришедших.

    Отчёт печатает блок поля ключ за ключом, и перевёрнутая пара
    «Параметр/Значение» бросалась бы в глаза сильнее самого пропуска.
    """
    for item in inspection_data:
        keys = list(item) if isinstance(item, dict) else []
        if keys[:2] == [_PARAM_KEY, _VALUE_KEY]:
            return {_PARAM_KEY: label, _VALUE_KEY: placeholder}
        if keys[:2] == [_VALUE_KEY, _PARAM_KEY]:
            return {_VALUE_KEY: placeholder, _PARAM_KEY: label}
    return {_VALUE_KEY: placeholder, _PARAM_KEY: label}
