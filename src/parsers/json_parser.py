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
from pathlib import Path
from typing import Any

_APPOINTMENTS_KEY = "appointments"


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
