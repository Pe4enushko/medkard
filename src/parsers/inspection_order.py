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
from dataclasses import dataclass
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


def match_score(a: str, b: str, *, max_distance: int = 2) -> int | None:
    """Расстояние между именами полей или ``None``, если это разные поля.

    Ноль — имена совпали дословно; чем больше, тем дальше написание разошлось.
    Нужно там, где на одно имя шаблона претендуют несколько полей записи:
    выигрывает ближайшее.
    """
    na, nb = _normalize(a), _normalize(b)
    budget = _distance_budget(na, nb, max_distance)
    distance = _levenshtein(na, nb, budget)
    return distance if distance <= budget else None


def labels_match(a: str, b: str, *, max_distance: int = 2) -> bool:
    """Одно ли это имя поля с точностью до дрейфа написания.

    Тот же матчинг, что и в переупорядочивании: 1С добавляет и убирает
    двоеточия, путает «листка»/«листке», а короткие имена сверяются точно.
    """
    return match_score(a, b, max_distance=max_distance) is not None


def normalized_labels(inspection_data: list[dict[str, Any]]) -> list[str]:
    """Нормализованные имена полей записи — в том порядке, в каком они пришли."""
    return [_normalize(str(item.get(_PARAM_KEY, ""))) for item in inspection_data]


def claim_nearest(
    names: list[str] | tuple[str, ...],
    norm_params: list[str],
    claimed: list[bool],
    *,
    max_distance: int = 2,
) -> int:
    """Индекс ближайшего незанятого поля записи, подходящего под любое из *names*,
    или -1. Ничьи решаются в пользу поля, пришедшего раньше."""
    best_idx = -1
    best_dist = max_distance + 1
    for name in names:
        nt = _normalize(name)
        for idx, np in enumerate(norm_params):
            if claimed[idx]:
                continue
            distance = match_score(nt, np, max_distance=max_distance)
            if distance is None or distance >= best_dist:
                continue
            best_dist, best_idx = distance, idx
            if distance == 0:
                return best_idx
    return best_idx


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
        best_idx = claim_nearest([token], norm_params, claimed, max_distance=max_distance)
        if best_idx >= 0:
            claimed[best_idx] = True
            result.append(inspection_data[best_idx])

    for idx, item in enumerate(inspection_data):
        if not claimed[idx]:
            result.append(item)

    return result


@dataclass(frozen=True)
class InspectionFormat:
    """Один шаблон осмотра: порядок полей, слоты и по чему шаблон опознаётся.

    Слот — одно поле записи; несколько имён в слоте значат, что 1С присылает
    его то под одним, то под другим («На приеме пациент с» = «родственник лвн»),
    и заполненным слот считается по любому из них.

    ``never_drawn`` — слоты, которые упорядочиваются наравне с остальными, но
    пустыми не рисуются: они живут в карте вне записи осмотра, и пустая строка
    сказала бы о них неправду.
    """

    name: str
    slots: tuple[tuple[str, ...], ...]
    signature: tuple[str, ...]
    min_signature_match: int
    never_drawn: tuple[str, ...] = ()

    @property
    def order_tokens(self) -> list[str]:
        return [name for slot in self.slots for name in slot]


def _slots(order: list[Any]) -> tuple[tuple[str, ...], ...]:
    """Строка манифеста → слоты. Запятая в строке разделяет разные поля, список
    из нескольких строк — имена одного поля."""
    slots: list[tuple[str, ...]] = []
    for line in order:
        if isinstance(line, (list, tuple)):
            names = tuple(str(name).strip() for name in line if str(name).strip())
            if names:
                slots.append(names)
            continue
        for token in str(line).split(","):
            token = token.strip()
            if token:
                slots.append((token,))
    return tuple(slots)


def _clinic_formats(clinic: str, path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    clinics = {key: value for key, value in data.items() if not key.startswith("_")}
    if clinic not in clinics:
        raise ValueError(
            f"Clinic {clinic!r} not found in {path} (have: {sorted(clinics)})"
        )
    return {
        key: value for key, value in clinics[clinic].items() if not key.startswith("_")
    }


def load_inspection_format(
    clinic: str,
    format_name: str,
    path: str | Path = _DEFAULT_FORMATS_PATH,
) -> list[str]:
    """Load the manifest JSON and return the flat, comma-split token list for
    ``[clinic][format_name]``.

    Accepts both manifest shapes: a bare list of order lines, and the object
    ``{"order": [...], "signature": [...], "min_signature_match": N}`` — only
    the order is returned either way.

    Raises ValueError with a clear message if the clinic or format is absent.
    """
    formats = _clinic_formats(clinic, path)
    if format_name not in formats:
        raise ValueError(
            f"Format {format_name!r} not found for clinic {clinic!r} in {path} "
            f"(have: {sorted(formats)})"
        )
    entry = formats[format_name]
    order = entry["order"] if isinstance(entry, dict) else entry
    return [name for slot in _slots(order) for name in slot]


def load_inspection_formats(
    clinic: str,
    path: str | Path = _DEFAULT_FORMATS_PATH,
) -> list[InspectionFormat]:
    """Все шаблоны клиники, которые можно опознать по самой записи, — в том
    порядке, в каком они лежат в манифесте (он же порядок приоритета).

    Шаблон без ``signature`` не возвращается: опознать его нечем, а применять
    ко всем подряд нельзя — одна клиника присылает несколько шаблонов.
    """
    out: list[InspectionFormat] = []
    for name, entry in _clinic_formats(clinic, path).items():
        if not isinstance(entry, dict) or not entry.get("signature"):
            continue
        out.append(
            InspectionFormat(
                name=name,
                slots=_slots(entry["order"]),
                signature=tuple(entry["signature"]),
                min_signature_match=int(entry.get("min_signature_match", len(entry["signature"]))),
                never_drawn=tuple(entry.get("never_drawn", ())),
            )
        )
    return out
