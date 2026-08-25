"""
required_fields.py — поля осмотра, отсутствие которых в записи считается дефектом.

1С не присылает поля, которые врач не заполнил: незаполненного анамнеза в
записи не «пусто», его там нет вовсе. Модель видит только пришедшие поля и
поэтому пропускает такое: на выгрузке Алёнки за 24.08 отсутствие анамнеза
отмечено в 5 картах из 11, обоснования диагноза — в 20 из 39. Отсутствие
считает код: он единственный, кто знает, каким запись должна была прийти.

Набор обязательных полей лежит в ``resources/required_fields.json`` и считан
по боевым картам — см. комментарий в самом файле.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from parsers.inspection_order import labels_match

_PARAM_KEY = "Параметр"
_VALUE_KEY = "Значение"
_ALL_KEY = "all"

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "resources" / "required_fields.json"

_templates: list[dict[str, Any]] | None = None


def _load(path: str | Path = _DEFAULT_PATH) -> list[dict[str, Any]]:
    global _templates
    if _templates is None:
        with open(path, encoding="utf-8") as f:
            _templates = json.load(f)["templates"]
    return _templates


def _filled_labels(inspection_data: list[dict[str, Any]]) -> list[str]:
    return [
        str(item.get(_PARAM_KEY, "")).strip()
        for item in inspection_data
        if isinstance(item, dict) and str(item.get(_VALUE_KEY, "") or "").strip()
    ]


def _slot(entry: Any) -> tuple[str, ...]:
    """Один слот шаблона: имя поля либо несколько имён одного и того же слота.

    Достаточно любого из имён — 1С присылает один и тот же текст то под
    «Рекомендации и назначения», то под «Рекомендации».
    """
    return (entry,) if isinstance(entry, str) else tuple(entry)


def _match_template(labels: list[str], templates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Шаблон, по которому заполнена запись, — по его собственным полям.

    Клиника здесь не спрашивается намеренно: одна и та же клиника присылает
    несколько шаблонов (осмотр, вакцинация, нефролог), и код услуги их не
    различает — на боевых картах базовый педиатрический шаблон приходил и
    первичным приёмом, и повторным, и профилактическим.
    """
    for template in templates:
        signature: list[str] = template.get("signature") or []
        matched = sum(
            1 for token in signature if any(labels_match(token, label) for label in labels)
        )
        if matched >= int(template.get("min_signature_match", len(signature))):
            return template
    return None


def missing_required_fields(
    inspection_data: list[dict[str, Any]],
    visit_type_keys: set[str],
    *,
    path: str | Path = _DEFAULT_PATH,
) -> list[str]:
    """Обязательные поля, которых в записи нет (или которые пришли пустыми).

    *visit_type_keys* — ключи типов визита из ``rules.json``
    (``primary`` / ``repeat`` / ``prophylactic`` / …). Если типов несколько,
    требуется только то, что требует каждый из них: набор одного типа не
    должен становиться замечанием из-за того, что услуга попала во второй.

    Пустой список — и когда всё на месте, и когда шаблон записи неизвестен.
    """
    labels = _filled_labels(inspection_data)
    if not labels:
        return []

    template = _match_template(labels, _load(path))
    if template is None:
        return []

    required: dict[str, list[Any]] = template.get("required") or {}
    common = [_slot(entry) for entry in (required.get(_ALL_KEY) or [])]
    per_type = [
        {_slot(entry) for entry in (required.get(key) or [])}
        for key in sorted(visit_type_keys)
    ]
    shared = set.intersection(*per_type) if per_type else set()

    expected = common + [
        slot for slot in dict.fromkeys(
            _slot(entry)
            for key in sorted(visit_type_keys)
            for entry in (required.get(key) or [])
        )
        if slot in shared
    ]

    return [
        slot[0] for slot in expected
        if not any(labels_match(name, label) for name in slot for label in labels)
    ]
