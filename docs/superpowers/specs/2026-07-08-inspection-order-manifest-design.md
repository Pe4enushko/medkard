# Design: Optional field-order manifest for `ДанныеОсмотра`

**Date:** 2026-07-08
**Project:** medkard
**Status:** Approved — ready for implementation plan

## Problem

`ДанныеОсмотра` (inspection data) in a 1C visit is a `list[{"Параметр": <field label>, "Значение": <value>}]`.
The fields arrive from 1C in arbitrary order, so the Excel report column D shows sections in an
inconsistent, hard-to-scan order across visits.

We want an **optional** manifest that defines a canonical field order per clinic and per card
format. When enabled, the report parser reorders each visit's `ДанныеОсмотра` list so its fields
follow the manifest order, matching manifest entries to actual `Параметр` labels
**case-insensitively and fuzzily** (Levenshtein distance ≤ 2 after light normalization). When not
enabled, behaviour is unchanged.

## Data findings (verified against real exports)

Measured against `report_Alenka_2026-06-25_to_2026-06-25.xlsx` (80 distinct params) and the
`data_snapshots/` JSON:

- The supplied manifest matches the **Alenka** clinic. MDS params are specialty-suffixed
  ("Жалобы кардиолог", "Диагноз невролог") and will not fuzzy-match — this is why the reorg must
  be optional and must never drop non-matching data.
- Manifest lines like `температура, чсс, чд, вес, рост` and `состояние, сознание` correspond to
  **several separate params**, so each manifest line is split by comma into individual tokens.
- Under light normalization + Levenshtein ≤ 2, **33 of 37 tokens** match Alenka params. An
  experiment comparing strategies confirmed Levenshtein is the right tool:

  | strategy | matched | note |
  |---|---|---|
  | exact (normalized) | 30/37 | misses 1-char drifts: `листка→листке`, `Рекомендована→Рекомендованна`, `ого→Огол` |
  | prefix | 33/37 | brittle on suffixes |
  | substring | 36/37 | **false positives**: `диагноз→Обоснование диагноза`, short tokens → single-letter param `Р` |
  | **Levenshtein ≤ 2** | **33/37** | +3 honest matches over exact, **0 false positives** |

  The 4 unmatched tokens (`На приеме пациент с`, `пациент нуждается в уходе`, `диагноз`,
  `рекомендовано посещение специалистов`) are genuinely absent from this Alenka export — accepted
  as-is (they simply contribute no ordering when the field is absent).

## Scope

- **In scope:** a manifest file format; a pure reorder function; a normalized fuzzy matcher; wiring
  through the audit script → Excel writer; unit + golden tests.
- **Out of scope:** DB/schema changes (this is an in-memory transform), the API/`build_workbook_bytes`
  path (only the CLI audit path gets `--format`; the manifest can be threaded there later if needed),
  aliases/synonym tables (manifest tokens are written to match the clinic's actual labels).

## Manifest format

New file `resources/inspection_formats.json`. Structure is **clinic → format → array of lines**:

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

- **First key = clinic**, identical to the audit script's positional `org` argument (`Alenka` / `MDS`).
- **Second key = format name**, selected via a new `--format` argument.
- **Value = array of lines**. On load, each line is split by comma into tokens; the flattened token
  list (in order) is the canonical field order. Manifest tokens are written to match the clinic's
  actual `Параметр` labels (e.g. `Аллергологический анамнез`, not `аллергический анамнез`).

### Selection & error semantics

- No `--format` given → **no reorg** (current behaviour, `order_tokens = None`).
- `--format X` given but `[clinic][X]` is missing in the JSON → **fail immediately** with a clear
  error naming the clinic and format (a typo must not pass silently).

## Reorder algorithm

New module `src/parsers/inspection_order.py`.

```python
def load_inspection_format(
    clinic: str, format_name: str, path: str | Path = <default resources path>
) -> list[str]:
    """Load resources/inspection_formats.json, select [clinic][format_name],
    comma-split every line into a flat token list. Raise KeyError/ValueError with a
    clear message if the clinic or format is absent."""

def reorder_inspection_data(
    inspection_data: list[dict],   # [{"Параметр": ..., "Значение": ...}, ...]
    order_tokens: list[str],       # flat token list from load_inspection_format
    *, max_distance: int = 2,
) -> list[dict]:
    ...
```

**Normalization** (comparison only; original dicts are never mutated):
`lower()` → `ё→е` → collapse internal whitespace → strip leading/trailing punctuation (` .:;,-—`).

**Matching** — greedy, "each manifest token claims its field":

1. Normalize every data item's `Параметр` once.
2. Walk `order_tokens` **in manifest order**. For each token, search the **not-yet-claimed** data
   items for one whose `levenshtein(norm(token), norm(Параметр)) ≤ max_distance`, choosing the
   minimum distance (ties broken by earliest original position). Mark it claimed and append it to
   the result.
3. Tokens with no match are skipped.

**Tail:** all still-unclaimed data items are appended in their original relative order.

**Invariants:**
- No item is lost or duplicated: `len(result) == len(inspection_data)`.
- Each data item is claimed by at most one token (claimed flag).
- **Duplicate `Параметр` labels in the data:** a token claims one (nearest / earliest); any
  remaining duplicates fall through to the tail.

**Levenshtein:** small local implementation (~10 lines) with an early-out when the length
difference exceeds `max_distance`. **No new dependency** — confirmed no `rapidfuzz` /
`python-Levenshtein` in requirements.

## Wiring

Chain: `scripts/audit-one-c-period.py` → `ExcelFormatter(path, legacy=)` →
`AuditExcelWriter(path, legacy=)` → `_build_row(...)`.

Thread an optional `order_tokens: list[str] | None = None` through each constructor:

- **`src/parsers/excel.py`**
  - `AuditExcelWriter.__init__(..., order_tokens=None)` — store it.
  - `_build_row(..., order_tokens=None)` — before rendering:
    ```python
    insp = visit.get("ДанныеОсмотра") or []
    if order_tokens:
        insp = reorder_inspection_data(insp, order_tokens)
    ... _pretty(insp) ...
    ```
  - `build_workbook_bytes(..., order_tokens=None)` — pass through (kept `None` from the API path
    for now).
- **`src/audit/excel_formatter.py`**
  - `ExcelFormatter.__init__(..., order_tokens=None)` — forward to `AuditExcelWriter`.
- **`scripts/audit-one-c-period.py`**
  - New `--format NAME` argument (default `None`).
  - If set: call `load_inspection_format(org, format, ...)` at startup (fail early on typo), pass the
    resulting tokens into `ExcelFormatter(..., order_tokens=...)`.
  - If unset: pass `order_tokens=None` — behaviour unchanged.

## Testing (TDD)

Unit tests for `reorder_inspection_data`:
- reorders matched fields into manifest order;
- unmatched data fields go to the tail preserving relative order;
- `len(result) == len(input)` invariant; nothing duplicated;
- duplicate `Параметр` labels: one claimed, rest to tail;
- empty input and empty `order_tokens`;
- Levenshtein cases: `в выдаче листка` ↔ `В выдаче листке` (dist 1), `ого` ↔ `Огол` (dist 1);
- no false match for `диагноз` against `Обоснование диагноза`.

Unit tests for `load_inspection_format`:
- comma-split flattening (`температура, чсс, чд, вес, рост` → 5 tokens);
- raises on missing clinic and on missing format.

Golden test: a real Alenka `ДанныеОсмотра` fragment reorders to the expected sequence.

## Files touched

- **new** `src/parsers/inspection_order.py`
- **new** `resources/inspection_formats.json`
- **new** `tests/test_inspection_order.py`
- **edit** `src/parsers/excel.py` (`_build_row`, `AuditExcelWriter`, `build_workbook_bytes`)
- **edit** `src/audit/excel_formatter.py` (`ExcelFormatter.__init__`)
- **edit** `scripts/audit-one-c-period.py` (`--format` arg + wiring)
