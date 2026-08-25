"""
json_parser.py — parse raw JSON payloads from 1C.

Supports two payload shapes:
  - A wrapper dict:  {"appointments": [<visit dict>, ...]}
  - A bare list:     [<visit dict>, ...]
  - A raw JSON string of either shape.

Usage::

    from parsers.json_parser import AppointmentParser, ParsedAppointment

    # Split a multi-visit payload into individual visit dicts
    visits = AppointmentParser.split(raw_json_or_dict_or_list)

    # Parse a single visit dict into typed, named parts
    parsed = AppointmentParser.parse(visit)
    patient    = parsed.patient
    diagnoses  = parsed.diagnoses
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

_APPOINTMENTS_KEY = "appointments"


def patient_age(patient: dict[str, Any]) -> int | None:
    """Возраст пациента из блока «Пациент», или None если его нельзя прочитать.

    Единственное место, где карта 1С превращается в число лет: и формальный
    контур, и подбор клинических рекомендаций обязаны видеть один и тот же
    возраст.

    Две тонкости, обе стоили дефектов:

    * ``AGE = 0`` — это ребёнок до года, а не отсутствие данных. Поэтому
      проверяется ``is None``, а не ложность значения: ``patient.get("AGE") or
      patient.get("Возраст")`` на нуле уходило во вторую ветку и возвращало
      None для всех младенцев.
    * Разбор намеренно строгий: «66 лет» и прочий текст возрастом не считаются.
      Вызывающий обязан трактовать None как «возраст неизвестен» в сторону
      сужения (см. ``FormalValidator.get_rules``), поэтому нераспознанное
      значение безопаснее вольного разбора.
    """
    raw = patient.get("AGE")
    if raw is None:
        raw = patient.get("Возраст")
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return int(raw) if raw >= 0 else None
    text = str(raw).strip()
    return int(text) if text.isdigit() else None


def visit_date(visit_meta: dict[str, Any]) -> date | None:
    """Дата приёма из блока «Прием», или None если её нельзя прочитать.

    1С отдаёт её в двух видах — «25.06.2026» и «2026-06-25T13:10:00», — поэтому
    разбор общий: и подбор статуса препарата в ГРЛС на дату визита, и блок
    «сегодня» для чекеров обязаны видеть один и тот же день.
    """
    raw = visit_meta.get("DATE")
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    value = raw.strip()
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        pass
    try:
        day, month, year = (int(part) for part in value.split("."))
        return date(year, month, day)
    except (TypeError, ValueError):
        return None


@dataclass
class ParsedAppointment:
    """Structured parts of a single raw 1C visit dict."""

    patient: dict[str, Any]
    diagnoses: list[dict[str, Any]]
    inspection_data: list[dict[str, Any]]
    services: list[dict[str, Any]]
    visit_meta: dict[str, Any]       # "Прием" block


class AppointmentParser:
    """Parse raw 1C JSON payloads — both multi-visit arrays and single visit dicts.

    # TODO: refactor DiagnosisValidator and FormalStructureValidator to use
    #       AppointmentParser.parse() instead of accessing visit keys directly.
    """

    @staticmethod
    def split(data: dict | list | str) -> list[dict[str, Any]]:
        """Extract the list of raw visit dicts from any supported payload shape.

        Args:
            data: A wrapper dict with an ``"appointments"`` key, a bare list of
                  visit dicts, or a raw JSON string of either shape.

        Returns:
            List of individual visit dicts.

        Raises:
            KeyError:    If a wrapper dict has no ``"appointments"`` key.
            ValueError:  If *data* is none of the supported types.
        """
        if isinstance(data, str):
            data = json.loads(data)

        if isinstance(data, list):
            return list(data)

        if isinstance(data, dict):
            return list(data[_APPOINTMENTS_KEY])

        raise ValueError(
            f"Cannot extract appointments from input of type {type(data).__name__!r}"
        )

    @staticmethod
    def split_file(path: str | Path) -> list[dict[str, Any]]:
        """Load a JSON file and split its appointments array.

        Args:
            path: Path to the JSON file.

        Returns:
            List of individual visit dicts.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return AppointmentParser.split(data)

    @staticmethod
    def parse(visit: dict[str, Any]) -> ParsedAppointment:
        """Parse a single raw visit dict into typed, named parts.

        Args:
            visit: A single visit dict as produced by :meth:`split`.

        Returns:
            :class:`ParsedAppointment` with all parts extracted.
        """
        return ParsedAppointment(
            patient=visit.get("Пациент") or {},
            diagnoses=visit.get("Диагнозы") or [],
            inspection_data=visit.get("ДанныеОсмотра") or [],
            services=visit.get("Услуги") or [],
            visit_meta=visit.get("Прием") or {},
        )
