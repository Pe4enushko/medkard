"""
inspection_labels.py — имена полей осмотра, под которыми их видит врач.

Часть колонок 1С отдаёт внутренними именами, которых нет ни в интерфейсе
врача, ни в русском языке: «родственник лвн» — это поле «На приеме пациент с».

Ярлык модель читает буквально. По имени «родственник лвн» она не признаёт
запись сведениями о сопровождающем и ставит
ОТСУТСТВУЕТ_ИНФОРМАЦИЯ_О_СОПРОВОЖДАЮЩЕМ на карту, где сопровождающий указан:
на прод-отчёте Алёнки за 24.08.2026 таких срабатываний 33 из 53, за 20.08 —
49 из 53. Правило при этом верное, врач заполнил поле верно — не совпадает
только имя.

Переименование делается один раз, на входе визита в аудит, поэтому одно и то
же имя видят и проверяющая модель, и отчёт, который читает врач.
"""

from __future__ import annotations

from typing import Any

from parsers.inspection_order import labels_match

_PARAM_KEY = "Параметр"
_INSPECTION_KEY = "ДанныеОсмотра"

# внутреннее имя 1С -> имя поля в интерфейсе врача
_ALIASES: dict[str, str] = {
    "родственник лвн": "На приеме пациент с",
}


def rename_internal_labels(inspection_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Копия ДанныеОсмотра, где внутренние имена заменены на врачебные.

    Сверка терпимая (``labels_match``): 1С меняет регистр и дописывает
    двоеточия. Порядок полей и всё остальное содержимое не трогаем.
    """
    renamed: list[dict[str, Any]] = []
    for item in inspection_data:
        if not isinstance(item, dict):
            renamed.append(item)
            continue
        label = str(item.get(_PARAM_KEY, ""))
        human = next(
            (name for internal, name in _ALIASES.items() if labels_match(internal, label)),
            None,
        )
        renamed.append({**item, _PARAM_KEY: human} if human else item)
    return renamed


def normalize_visit_labels(visit: dict[str, Any]) -> dict[str, Any]:
    """Копия визита с врачебными именами полей осмотра."""
    inspection = visit.get(_INSPECTION_KEY)
    if not isinstance(inspection, list) or not inspection:
        return visit
    return {**visit, _INSPECTION_KEY: rename_internal_labels(inspection)}
