"""
parsers/filter_config.py — load per-org CardFilter from filterconfig.json.

Usage::
    from parsers.filter_config import load_card_filter
    card_filter = load_card_filter("Alenka")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit.filters import AnalysisFilter, CardFilter, IcdFilter, KDLFilter, _BaseFilter

_CONFIG_PATH = Path(__file__).parent.parent.parent / "filterconfig.json"

_STRATEGY_REGISTRY: dict[str, type[_BaseFilter]] = {
    "IcdFilter": IcdFilter,
    "KDLFilter": KDLFilter,
    "AnalysisFilter": AnalysisFilter,
}


def _build_strategy(entry: dict[str, Any]) -> _BaseFilter:
    name: str = entry["name"]
    args: dict[str, Any] = entry.get("args") or {}
    cls = _STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown filter strategy {name!r}. "
            f"Available: {sorted(_STRATEGY_REGISTRY)}"
        )
    return cls(**args)


def load_card_filter(org_name: str) -> CardFilter:
    """Read filterconfig.json and return a CardFilter for the given org.

    Returns an empty CardFilter (no strategies) if the org is not listed in the config.
    """
    config: list[dict[str, Any]] = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    for entry in config:
        if entry.get("org") == org_name:
            strategies = [
                _build_strategy(s)
                for s in (entry.get("strategies") or [])
                if s.get("is_active", True)
            ]
            return CardFilter(strategies)
    return CardFilter([])
