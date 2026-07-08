# Inspection-Order Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional per-clinic/per-format manifest that reorders each visit's `ДанныеОсмотра` fields into a canonical order in the Excel report, matching field labels case-insensitively and fuzzily (Levenshtein ≤ 2).

**Architecture:** A pure module `src/parsers/inspection_order.py` loads the manifest (`resources/inspection_formats.json`, structured clinic → format → lines) and reorders a `ДанныеОсмотра` list in memory. The token list is threaded as an optional `order_tokens` argument through `ExcelFormatter → AuditExcelWriter → _build_row`, applied just before rendering. Selection is via a new `--format` CLI argument; absence means no reorg (unchanged behaviour).

**Tech Stack:** Python 3.11, pytest (asyncio_mode=auto, pythonpath=src), openpyxl. No new dependencies — Levenshtein is a small local implementation.

## Global Constraints

- `pythonpath = src` (from `pytest.ini`) — imports in tests are `from parsers.inspection_order import ...`, no package prefix.
- **No new dependencies.** Confirmed no `rapidfuzz` / `python-Levenshtein` in requirements; implement Levenshtein locally.
- The reorg must **never drop or duplicate** a `ДанныеОсмотра` item: `len(result) == len(input)` always.
- Original visit dicts must never be mutated; normalization is for comparison only.
- Default behaviour (no `--format`) is byte-for-byte unchanged: `order_tokens` defaults to `None` everywhere.
- Match rule: normalize (`lower()`, `ё→е`, collapse internal whitespace, strip leading/trailing ` .:;,-—`), then Levenshtein ≤ 2, greedy, earliest-position tie-break.

---

### Task 1: Reorder core — normalization, Levenshtein, `reorder_inspection_data`

**Files:**
- Create: `src/parsers/inspection_order.py`
- Test: `tests/test_inspection_order.py`

**Interfaces:**
- Consumes: nothing (leaf module, stdlib only).
- Produces:
  - `reorder_inspection_data(inspection_data: list[dict], order_tokens: list[str], *, max_distance: int = 2) -> list[dict]`
  - `_normalize(s: str) -> str` (private)
  - `_levenshtein(a: str, b: str, max_distance: int = 2) -> int` (private; returns a value `> max_distance` when the true distance exceeds the bound)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_inspection_order.py`:

```python
from parsers.inspection_order import (
    reorder_inspection_data,
    _normalize,
    _levenshtein,
)


def _item(param, value=""):
    return {"Параметр": param, "Значение": value}


def test_normalize_lowercases_dedots_and_strips():
    assert _normalize("  Рекомендации и назначения:  ") == "рекомендации и назначения"
    assert _normalize("Ф20") == "ф20"
    assert _normalize("Жёлчь") == "желчь"          # ё -> е
    assert _normalize("а   б") == "а б"            # whitespace collapse


def test_levenshtein_basic_and_earlyout():
    assert _levenshtein("огол", "огол") == 0
    assert _levenshtein("листка", "листке") == 1
    assert _levenshtein("ого", "огол") == 1
    # length gap exceeds bound -> returns something > max_distance, not the exact distance
    assert _levenshtein("а", "аллергический анамнез", max_distance=2) > 2


def test_reorder_exact_manifest_order():
    data = [_item("Диагноз"), _item("Жалобы на момент осмотра"), _item("Анамнез заболевания")]
    tokens = ["жалобы на момент осмотра", "анамнез заболевания", "диагноз"]
    out = reorder_inspection_data(data, tokens)
    assert [d["Параметр"] for d in out] == [
        "Жалобы на момент осмотра",
        "Анамнез заболевания",
        "Диагноз",
    ]


def test_reorder_fuzzy_one_char_drift():
    data = [_item("В выдаче листке нетрудоспособности"), _item("Огол")]
    tokens = ["огол", "в выдаче листка нетрудоспособности"]
    out = reorder_inspection_data(data, tokens)
    assert [d["Параметр"] for d in out] == [
        "Огол",
        "В выдаче листке нетрудоспособности",
    ]


def test_unmatched_data_goes_to_tail_preserving_order():
    data = [_item("Группа здоровья"), _item("Диагноз"), _item("Заметки")]
    tokens = ["диагноз"]
    out = reorder_inspection_data(data, tokens)
    assert [d["Параметр"] for d in out] == ["Диагноз", "Группа здоровья", "Заметки"]


def test_unmatched_tokens_are_skipped():
    data = [_item("Диагноз")]
    tokens = ["на приеме пациент с", "диагноз", "пациент нуждается в уходе"]
    out = reorder_inspection_data(data, tokens)
    assert [d["Параметр"] for d in out] == ["Диагноз"]


def test_length_invariant_and_no_duplication():
    data = [_item("A"), _item("B"), _item("C")]
    out = reorder_inspection_data(data, ["b"])
    assert len(out) == 3
    assert sorted(d["Параметр"] for d in out) == ["A", "B", "C"]


def test_duplicate_labels_one_claimed_rest_to_tail():
    data = [_item("Рекомендации", "first"), _item("Диагноз"), _item("Рекомендации", "second")]
    tokens = ["рекомендации", "диагноз"]
    out = reorder_inspection_data(data, tokens)
    # first "Рекомендации" claimed by token, then Диагноз, then leftover duplicate in tail
    assert [(d["Параметр"], d["Значение"]) for d in out] == [
        ("Рекомендации", "first"),
        ("Диагноз", ""),
        ("Рекомендации", "second"),
    ]


def test_no_false_match_diagnoz_vs_obosnovanie():
    # "диагноз" must NOT claim "Обоснование диагноза" (distance >> 2)
    data = [_item("Обоснование диагноза")]
    tokens = ["диагноз"]
    out = reorder_inspection_data(data, tokens)
    # single item, unmatched -> tail; still present exactly once
    assert [d["Параметр"] for d in out] == ["Обоснование диагноза"]


def test_empty_inputs():
    assert reorder_inspection_data([], ["диагноз"]) == []
    data = [_item("Диагноз")]
    assert [d["Параметр"] for d in reorder_inspection_data(data, [])] == ["Диагноз"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/okabe/projects/medkard && python -m pytest tests/test_inspection_order.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'parsers.inspection_order'`

- [ ] **Step 3: Write the implementation**

Create `src/parsers/inspection_order.py` (the `load_inspection_format` function is added in Task 2; this task only creates the reorder core):

```python
"""
inspection_order.py — optional canonical reordering of ДанныеОсмотра fields.

ДанныеОсмотра is a list of {"Параметр": <label>, "Значение": <value>} dicts
arriving from 1C in arbitrary order. Given a flat list of canonical order
tokens (from a per-clinic/per-format manifest), reorder the list so matched
fields follow the manifest order; unmatched fields keep their original
relative order and are appended after the matched ones.

Matching is case-insensitive and fuzzy: labels are normalized (lowercased,
ё→е, whitespace collapsed, leading/trailing punctuation stripped) and compared
by Levenshtein distance with a small threshold (default 2).
"""

from __future__ import annotations

import re
from typing import Any

_PARAM_KEY = "Параметр"
_STRIP_CHARS = " .:;,-—"


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
            d = _levenshtein(nt, np, max_distance)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/okabe/projects/medkard && python -m pytest tests/test_inspection_order.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /home/okabe/projects/medkard
git add src/parsers/inspection_order.py tests/test_inspection_order.py
git commit -m "feat: add reorder_inspection_data core with fuzzy field matching"
```

---

### Task 2: Manifest file + `load_inspection_format` loader

**Files:**
- Create: `resources/inspection_formats.json`
- Modify: `src/parsers/inspection_order.py` (add loader function + imports)
- Modify: `tests/test_inspection_order.py` (append loader tests)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `load_inspection_format(clinic: str, format_name: str, path: str | Path = _DEFAULT_FORMATS_PATH) -> list[str]`
  - `_DEFAULT_FORMATS_PATH: Path` pointing at `resources/inspection_formats.json`

- [ ] **Step 1: Create the manifest file**

Create `resources/inspection_formats.json`:

```json
{
  "Alenka": {
    "standard": [
      "жалобы на момент осмотра",
      "анамнез заболевания",
      "эпидемиологический анамнез",
      "прививочный анамнез",
      "Аллергологический анамнез",
      "температура, чсс, чд, вес, рост",
      "огол, ого",
      "состояние, сознание",
      "ф20, кожные покровы",
      "видимые слизистые",
      "слизистые ротоглотки",
      "миндалины",
      "периферические лимфоузлы",
      "неврологический статус",
      "опорно-двигательная система",
      "сердечно-сосудистая система",
      "дыхательная система",
      "органы брюшной полости",
      "стул",
      "мочеиспускание",
      "план обследования",
      "план лечения",
      "обоснование диагноза",
      "диагноз",
      "рекомендации и назначения",
      "рекомендована следующая плановая консультация"
    ]
  }
}
```

- [ ] **Step 2: Write the failing loader tests**

Append to `tests/test_inspection_order.py`:

```python
import json

import pytest

from parsers.inspection_order import load_inspection_format


def _write_formats(tmp_path, data):
    p = tmp_path / "inspection_formats.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def test_load_comma_split_flattens_tokens(tmp_path):
    p = _write_formats(
        tmp_path,
        {"Alenka": {"standard": ["температура, чсс, чд, вес, рост", "диагноз"]}},
    )
    tokens = load_inspection_format("Alenka", "standard", path=p)
    assert tokens == ["температура", "чсс", "чд", "вес", "рост", "диагноз"]


def test_load_real_manifest_default_path():
    # the committed resources/inspection_formats.json must load and split
    tokens = load_inspection_format("Alenka", "standard")
    assert "температура" in tokens and "рост" in tokens
    assert "жалобы на момент осмотра" in tokens


def test_load_missing_clinic_raises(tmp_path):
    p = _write_formats(tmp_path, {"Alenka": {"standard": ["диагноз"]}})
    with pytest.raises((KeyError, ValueError)):
        load_inspection_format("MDS", "standard", path=p)


def test_load_missing_format_raises(tmp_path):
    p = _write_formats(tmp_path, {"Alenka": {"standard": ["диагноз"]}})
    with pytest.raises((KeyError, ValueError)):
        load_inspection_format("Alenka", "typo", path=p)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/okabe/projects/medkard && python -m pytest tests/test_inspection_order.py -k load -v`
Expected: FAIL — `ImportError: cannot import name 'load_inspection_format'`

- [ ] **Step 4: Add the loader to `inspection_order.py`**

Add near the top of `src/parsers/inspection_order.py`, after the existing `from __future__` import add `import json` and `from pathlib import Path`, then add:

```python
_DEFAULT_FORMATS_PATH = Path(__file__).resolve().parents[2] / "resources" / "inspection_formats.json"


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
```

Note: `parents[2]` resolves `src/parsers/inspection_order.py` → repo root; `resources/` sits at repo root.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/okabe/projects/medkard && python -m pytest tests/test_inspection_order.py -v`
Expected: PASS (all tests, including Task 1's)

- [ ] **Step 6: Commit**

```bash
cd /home/okabe/projects/medkard
git add src/parsers/inspection_order.py resources/inspection_formats.json tests/test_inspection_order.py
git commit -m "feat: add inspection_formats.json manifest and load_inspection_format loader"
```

---

### Task 3: Thread `order_tokens` through the Excel writer

**Files:**
- Modify: `src/parsers/excel.py` (`_build_row` at :156, `build_workbook_bytes` at :182, `AuditExcelWriter.__init__` at :222, `AuditExcelWriter.append` at :248)
- Test: `tests/test_excel_reorder.py` (create)

**Interfaces:**
- Consumes: `reorder_inspection_data` from Task 1.
- Produces:
  - `_build_row(visit, formal, diagnosis, icd_check=None, *, legacy=False, order_tokens=None)`
  - `build_workbook_bytes(rows, *, legacy=False, order_tokens=None)`
  - `AuditExcelWriter(path, *, legacy=False, order_tokens=None)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_excel_reorder.py`:

```python
import openpyxl

from audit.models import FormalStructureResult
from parsers.excel import AuditExcelWriter


def _visit():
    return {
        "Врач": {"SPECIALIZATION": "педиатр"},
        "Прием": {"DATE": "25.06.2026"},
        "ДанныеОсмотра": [
            {"Параметр": "Диагноз", "Значение": "ОРВИ"},
            {"Параметр": "Жалобы на момент осмотра", "Значение": "кашель"},
            {"Параметр": "Анамнез заболевания", "Значение": "3 дня"},
        ],
    }


def _read_inspection_cell(path):
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    # column D (index 4) = "Данные осмотра"
    val = ws.cell(row=2, column=4).value
    wb.close()
    return val


def test_writer_without_order_tokens_keeps_source_order(tmp_path):
    path = tmp_path / "r.xlsx"
    writer = AuditExcelWriter(path)
    writer.append(_visit(), FormalStructureResult(findings=[]), diagnosis=[], icd_check=[])
    text = _read_inspection_cell(path)
    assert text.index("Диагноз") < text.index("Жалобы на момент осмотра")


def test_writer_with_order_tokens_reorders(tmp_path):
    path = tmp_path / "r.xlsx"
    tokens = ["жалобы на момент осмотра", "анамнез заболевания", "диагноз"]
    writer = AuditExcelWriter(path, order_tokens=tokens)
    writer.append(_visit(), FormalStructureResult(findings=[]), diagnosis=[], icd_check=[])
    text = _read_inspection_cell(path)
    assert text.index("Жалобы на момент осмотра") < text.index("Анамнез заболевания") < text.index("Диагноз")
```

Note: `FormalStructureResult` is defined in `src/storage/models/result.py` (re-exported from `audit.models`); its only field is `findings: list[FormalFinding]` with a default factory, so `FormalStructureResult(findings=[])` — or even `FormalStructureResult()` — is valid. It only needs to render via `_pretty`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/okabe/projects/medkard && python -m pytest tests/test_excel_reorder.py -v`
Expected: FAIL — `AuditExcelWriter.__init__() got an unexpected keyword argument 'order_tokens'`

- [ ] **Step 3: Edit `src/parsers/excel.py`**

3a. Add the import near the other local imports (after `from audit.models import FormalStructureResult`):

```python
from parsers.inspection_order import reorder_inspection_data
```

3b. Change `_build_row` signature (:156) and the inspection-render line (:173). Current:

```python
def _build_row(
    visit: dict[str, Any],
    formal: FormalStructureResult,
    diagnosis: Any,
    icd_check: Any = None,
    *,
    legacy: bool = False,
) -> list[str]:
    if legacy:
        return [_pretty(visit), _pretty(formal), _pretty(diagnosis)]

    specialization = (visit.get("Врач") or {}).get("SPECIALIZATION") or "—"
    visit_date = (visit.get("Прием") or {}).get("DATE") or "—"
    return [
        specialization,
        visit_date,
        _card_data_text(visit),
        _pretty(visit.get("ДанныеОсмотра") or []),
```

Replace with:

```python
def _build_row(
    visit: dict[str, Any],
    formal: FormalStructureResult,
    diagnosis: Any,
    icd_check: Any = None,
    *,
    legacy: bool = False,
    order_tokens: list[str] | None = None,
) -> list[str]:
    if legacy:
        return [_pretty(visit), _pretty(formal), _pretty(diagnosis)]

    specialization = (visit.get("Врач") or {}).get("SPECIALIZATION") or "—"
    visit_date = (visit.get("Прием") or {}).get("DATE") or "—"
    inspection = visit.get("ДанныеОсмотра") or []
    if order_tokens:
        inspection = reorder_inspection_data(inspection, order_tokens)
    return [
        specialization,
        visit_date,
        _card_data_text(visit),
        _pretty(inspection),
```

(The remaining list elements — `Услуги`, `Диагнозы`, `formal`, `diagnosis`, `icd_check` — stay unchanged.)

3c. Change `build_workbook_bytes` (:182) signature and its `_build_row` call (:203):

```python
def build_workbook_bytes(
    rows: list[tuple[dict[str, Any], FormalStructureResult, Any, Any]],
    *,
    legacy: bool = False,
    order_tokens: list[str] | None = None,
) -> bytes:
```

and the call inside the loop:

```python
        row = _build_row(visit, formal, diagnosis, icd_check, legacy=legacy, order_tokens=order_tokens)
```

3d. Change `AuditExcelWriter.__init__` (:222) to accept and store `order_tokens`:

```python
    def __init__(
        self,
        path: str | Path,
        *,
        legacy: bool = False,
        order_tokens: list[str] | None = None,
    ) -> None:
        self._path = Path(path)
        self._legacy = legacy
        self._order_tokens = order_tokens
```

3e. Change the `_build_row` call inside `AuditExcelWriter.append` (:263):

```python
            row = _build_row(
                visit, formal, diagnosis, icd_check,
                legacy=self._legacy, order_tokens=self._order_tokens,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/okabe/projects/medkard && python -m pytest tests/test_excel_reorder.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
cd /home/okabe/projects/medkard
git add src/parsers/excel.py tests/test_excel_reorder.py
git commit -m "feat: thread optional order_tokens through Excel writer"
```

---

### Task 4: Thread `order_tokens` through `ExcelFormatter`

**Files:**
- Modify: `src/audit/excel_formatter.py` (`ExcelFormatter.__init__` at :130-131)
- Test: `tests/test_excel_reorder.py` (append a formatter-forwarding test)

**Interfaces:**
- Consumes: `AuditExcelWriter(..., order_tokens=...)` from Task 3.
- Produces: `ExcelFormatter(excel_path, *, legacy=False, order_tokens=None)` that forwards `order_tokens` to its `AuditExcelWriter`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_excel_reorder.py`:

```python
from audit.excel_formatter import ExcelFormatter


def test_formatter_forwards_order_tokens_to_writer(tmp_path):
    tokens = ["диагноз"]
    fmt = ExcelFormatter(tmp_path / "r.xlsx", order_tokens=tokens)
    assert fmt._excel._order_tokens == tokens
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/okabe/projects/medkard && python -m pytest tests/test_excel_reorder.py::test_formatter_forwards_order_tokens_to_writer -v`
Expected: FAIL — `ExcelFormatter.__init__() got an unexpected keyword argument 'order_tokens'`

- [ ] **Step 3: Edit `src/audit/excel_formatter.py`**

Change `ExcelFormatter.__init__` (:130-131). Current:

```python
    def __init__(self, excel_path: str | Path, *, legacy: bool = False) -> None:
        self._excel = AuditExcelWriter(excel_path, legacy=legacy)
        self._reader = _DoneCardsReader()
```

Replace with:

```python
    def __init__(
        self,
        excel_path: str | Path,
        *,
        legacy: bool = False,
        order_tokens: list[str] | None = None,
    ) -> None:
        self._excel = AuditExcelWriter(excel_path, legacy=legacy, order_tokens=order_tokens)
        self._reader = _DoneCardsReader()
```

Also update the class docstring `Args:` block to mention `order_tokens` (optional canonical field order for ДанныеОсмотра; ``None`` disables reordering).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/okabe/projects/medkard && python -m pytest tests/test_excel_reorder.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /home/okabe/projects/medkard
git add src/audit/excel_formatter.py tests/test_excel_reorder.py
git commit -m "feat: forward order_tokens through ExcelFormatter"
```

---

### Task 5: `--format` CLI argument + wiring in the audit script

**Files:**
- Modify: `scripts/audit-one-c-period.py` (argparse block near :53, startup region after :54, `ExcelFormatter(...)` call at :158)

**Interfaces:**
- Consumes: `load_inspection_format` (Task 2), `ExcelFormatter(..., order_tokens=...)` (Task 4).
- Produces: a `--format NAME` CLI flag; when set, tokens are loaded at startup (failing fast on a bad name) and passed into `ExcelFormatter`.

This task has no unit test (it's CLI glue over already-tested units); it is verified by running the script's help and a fast failure path.

- [ ] **Step 1: Add the argparse option**

After the `--legacy-report` argument (:53) add:

```python
_parser.add_argument(
    "--format",
    default=None,
    metavar="NAME",
    help="Reorder ДанныеОсмотра fields using resources/inspection_formats.json "
         "[<org>][<NAME>]. Omit to leave field order unchanged.",
)
```

- [ ] **Step 2: Load tokens at startup (fail fast on typo)**

Add the import with the other `parsers` imports at the top of the file:

```python
from parsers.inspection_order import load_inspection_format
```

Immediately after `_args = _parser.parse_args()` (:54), add:

```python
INSPECTION_ORDER = (
    load_inspection_format(_args.org, _args.format) if _args.format else None
)
```

(`_args.org` is the clinic key; `load_inspection_format` raises a clear `ValueError` if the org/format pair is absent, which surfaces immediately at module load — before any 1C fetch.)

- [ ] **Step 3: Pass tokens into ExcelFormatter**

Change the call at :158. Current:

```python
        async with ExcelFormatter(EXCEL_PATH, legacy=_args.legacy_report) as fmt:
```

Replace with:

```python
        async with ExcelFormatter(
            EXCEL_PATH, legacy=_args.legacy_report, order_tokens=INSPECTION_ORDER
        ) as fmt:
```

- [ ] **Step 4: Verify the CLI wiring**

Run help (must list `--format`):

```bash
cd /home/okabe/projects/medkard && python scripts/audit-one-c-period.py --help
```
Expected: usage text includes `--format NAME`.

Verify fail-fast on a bad format name. Because token load happens right after `parse_args()`, a bad `--format` must raise `ValueError` before the confirmation prompt / any network work:

```bash
cd /home/okabe/projects/medkard && python scripts/audit-one-c-period.py Alenka --format nonexistent -y 2>&1 | head -20
```
Expected: `ValueError: Format 'nonexistent' not found for clinic 'Alenka' ...` and a non-zero exit.

Verify the good name loads without error (it will proceed toward the audit; interrupt with Ctrl-C once it gets past format loading — the absence of a ValueError is the pass condition):

```bash
cd /home/okabe/projects/medkard && timeout 5 python scripts/audit-one-c-period.py Alenka --format standard 2>&1 | head -20 || true
```
Expected: no `ValueError` about the format; output shows the run starting (confirmation prompt or period log), proving `standard` loaded.

- [ ] **Step 5: Run the full new-feature suite**

Run: `cd /home/okabe/projects/medkard && python -m pytest tests/test_inspection_order.py tests/test_excel_reorder.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
cd /home/okabe/projects/medkard
git add scripts/audit-one-c-period.py
git commit -m "feat: add --format flag wiring inspection order into the audit run"
```

---

### Task 6: Golden test on real Alenka data + docs

**Files:**
- Test: `tests/test_inspection_order_golden.py` (create)
- Modify: `CLAUDE.md` (Architecture section — note the new module)

**Interfaces:**
- Consumes: `load_inspection_format`, `reorder_inspection_data`.
- Produces: nothing new (verification + docs).

- [ ] **Step 1: Write the golden test**

Create `tests/test_inspection_order_golden.py`:

```python
from parsers.inspection_order import load_inspection_format, reorder_inspection_data


def test_alenka_manifest_orders_a_realistic_fragment():
    tokens = load_inspection_format("Alenka", "standard")
    # a realistic, out-of-order Alenka ДанныеОсмотра fragment (labels as they
    # appear in real exports, incl. the trailing-colon case)
    data = [
        {"Параметр": "Направление к другому специалисту (списком)", "Значение": "Лаборатория"},
        {"Параметр": "Рекомендации и назначения:", "Значение": "..."},
        {"Параметр": "Диагноз", "Значение": "J06.9"},
        {"Параметр": "Жалобы на момент осмотра", "Значение": "кашель"},
        {"Параметр": "Анамнез заболевания", "Значение": "3 дня"},
        {"Параметр": "Температура", "Значение": "37"},
        {"Параметр": "ЧСС", "Значение": "80"},
    ]
    out = reorder_inspection_data(data, tokens)
    labels = [d["Параметр"] for d in out]

    # matched fields follow manifest order
    assert labels.index("Жалобы на момент осмотра") < labels.index("Анамнез заболевания")
    assert labels.index("Анамнез заболевания") < labels.index("Температура")
    assert labels.index("Температура") < labels.index("ЧСС")
    assert labels.index("ЧСС") < labels.index("Диагноз")
    assert labels.index("Диагноз") < labels.index("Рекомендации и назначения:")
    # unmatched field goes to the tail
    assert labels[-1] == "Направление к другому специалисту (списком)"
    # nothing lost
    assert len(out) == len(data)
```

- [ ] **Step 2: Run the golden test**

Run: `cd /home/okabe/projects/medkard && python -m pytest tests/test_inspection_order_golden.py -v`
Expected: PASS

- [ ] **Step 3: Update `CLAUDE.md`**

In the `src/` tree diagram under `parsers/`, add the `inspection_order.py` line so it reads:

```
└── parsers/
    ├── excel.py             # Input Excel parser
    ├── inspection_order.py  # Optional canonical reorder of ДанныеОсмотра (manifest-driven, fuzzy match)
    └── json_parser.py       # Visit JSON normalization
```

- [ ] **Step 4: Run the full new-feature suite**

Run: `cd /home/okabe/projects/medkard && python -m pytest tests/test_inspection_order.py tests/test_inspection_order_golden.py tests/test_excel_reorder.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /home/okabe/projects/medkard
git add tests/test_inspection_order_golden.py CLAUDE.md
git commit -m "test: add golden reorder test on Alenka data; document inspection_order module"
```

---

## Notes for the implementer

- Run tests with `python -m pytest` from the repo root; `pytest.ini` sets `pythonpath = src` and `asyncio_mode=auto`, so `from parsers.inspection_order import ...` resolves without installation.
- The default (no `--format`) path must stay identical to today's output. Every new argument defaults to `None`.
- `FormalStructureResult` (from `audit.models`, defined in `storage/models/result.py`) has a single `findings` field with a default factory; `FormalStructureResult(findings=[])` is the minimal valid construction for Task 3's test. The assertion is about the ДанныеОсмотра cell, not the formal cell.
