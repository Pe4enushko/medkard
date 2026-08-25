"""
inspection_order.py — optional canonical reordering of ДанныеОсмотра fields.

ДанныеОсмотра is a list of {"Параметр": <label>, "Значение": <value>} dicts
arriving from 1C in arbitrary order. Given a flat list of canonical order
tokens (from a per-clinic/per-format manifest), reorder the list so matched
fields follow the manifest order; unmatched fields keep their original
relative order and are appended after the matched ones.

Matching is case-insensitive and fuzzy: labels are normalized (lowercased,
ё→е, whitespace collapsed, leading/trailing punctuation stripped) and compared
by Levenshtein distance with a small threshold (default 2). Names shorter than
_MIN_FUZZY_LEN characters must match exactly — see _distance_budget.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_PARAM_KEY = "Параметр"
_STRIP_CHARS = " .:;,-—"
_MIN_FUZZY_LEN = 4

_DEFAULT_FORMATS_PATH = Path(__file__).resolve().parents[2] / "resources" / "inspection_formats.json"


def _normalize(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(_STRIP_CHARS)


def _levenshtein(a: str, b: str, max_distance: int = 2) -> int:
    """Levenshtein distance with an early-out: if the true distance exceeds
    *max_distance*, returns a value greater than *max_distance* (not necessarily
    the exact distance)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > max_distance:
        return max_distance + 1
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            val = min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + cost)
            cur.append(val)
            if val < row_min:
                row_min = val
        if row_min > max_distance:
            return max_distance + 1
        prev = cur
    return prev[lb]


def _distance_budget(a: str, b: str, max_distance: int) -> int:
    """Distance allowed between normalized names *a* and *b*.

    Abbreviations are 2-3 characters long, and a distance-2 window on such a
    name matches nearly anything: on real vaccination cards «Фсс» landed on the
    «чсс» rank and «ФР»/«ХЗ»/«Р» on «чд». Anything shorter than _MIN_FUZZY_LEN
    therefore has to match exactly.
    """
    if min(len(a), len(b)) < _MIN_FUZZY_LEN:
        return 0
    return max_distance


def labels_match(a: str, b: str, *, max_distance: int = 2) -> bool:
    """Одно ли это имя поля с точностью до дрейфа написания.

    Тот же матчинг, что и в переупорядочивании: 1С добавляет и убирает
    двоеточия, путает «листка»/«листке», а короткие имена сверяются точно.
    """
    na, nb = _normalize(a), _normalize(b)
    budget = _distance_budget(na, nb, max_distance)
    return _levenshtein(na, nb, budget) <= budget


def reorder_inspection_data(
    inspection_data: list[dict[str, Any]],
    order_tokens: list[str],
    *,
    max_distance: int = 2,
) -> list[dict[str, Any]]:
    """Return a new list of the same items reordered to follow *order_tokens*.

    Greedy: each token, in manifest order, claims the nearest not-yet-claimed
    item whose normalized Параметр is within *max_distance* of the normalized
    token (ties broken by earliest original position). Unmatched items are
    appended in their original relative order. Never drops or duplicates items.
    """
    if not order_tokens or not inspection_data:
        return list(inspection_data)

    norm_params = [_normalize(str(item.get(_PARAM_KEY, ""))) for item in inspection_data]
    claimed = [False] * len(inspection_data)
    result: list[dict[str, Any]] = []

    for token in order_tokens:
        nt = _normalize(token)
        best_idx = -1
        best_dist = max_distance + 1
        for idx, np in enumerate(norm_params):
            if claimed[idx]:
                continue
            budget = _distance_budget(nt, np, max_distance)
            d = _levenshtein(nt, np, budget)
            if d > budget:
                continue
            if d < best_dist:
                best_dist = d
                best_idx = idx
                if d == 0:
                    break
        if best_idx >= 0:
            claimed[best_idx] = True
            result.append(inspection_data[best_idx])

    for idx, item in enumerate(inspection_data):
        if not claimed[idx]:
            result.append(item)

    return result


def load_inspection_format(
    clinic: str,
    format_name: str,
    path: str | Path = _DEFAULT_FORMATS_PATH,
) -> list[str]:
    """Load the manifest JSON and return the flat, comma-split token list for
    ``[clinic][format_name]``.

    Raises ValueError with a clear message if the clinic or format is absent.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if clinic not in data:
        raise ValueError(
            f"Clinic {clinic!r} not found in {path} (have: {sorted(data)})"
        )
    formats = data[clinic]
    if format_name not in formats:
        raise ValueError(
            f"Format {format_name!r} not found for clinic {clinic!r} in {path} "
            f"(have: {sorted(formats)})"
        )

    tokens: list[str] = []
    for line in formats[format_name]:
        for tok in str(line).split(","):
            tok = tok.strip()
            if tok:
                tokens.append(tok)
    return tokens
