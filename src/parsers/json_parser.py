"""
json_parser.py — parse raw JSON payloads from 1C.

The top-level format is::

    {
        "appointments": [ <visit dict>, <visit dict>, ... ]
    }

Usage::

    from parsers.json_parser import AppointmentsParser

    with open("data.json", encoding="utf-8") as f:
        data = json.load(f)

    visits = AppointmentsParser.split(data)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AppointmentsParser:
    """Parse raw appointment payloads."""

    @staticmethod
    def split(data: dict[str, Any]) -> list[dict[str, Any]]:
        """Split the ``appointments`` array into individual visit dicts.

        Args:
            data: Parsed JSON object with a top-level ``appointments`` key.

        Returns:
            List of individual appointment dicts.

        Raises:
            KeyError: If the ``appointments`` key is missing.
        """
        return list(data["appointments"])

    @staticmethod
    def split_file(path: str | Path) -> list[dict[str, Any]]:
        """Load a JSON file and split its appointments array.

        Args:
            path: Path to the JSON file.

        Returns:
            List of individual appointment dicts.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return AppointmentsParser.split(data)
