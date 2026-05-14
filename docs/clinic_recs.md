# ClinicRecs

**File:** `audit/diagnosis/clinic_recs.py`

Maps an МКБ-10 diagnosis code to a clinical-guideline `file_id` from `manifest.csv`. Implements a three-tier lookup strategy: exact match → BM25 token overlap → LLM decider.

## Usage

```python
recs = ClinicRecs()
file_id = await recs.pick_recs(patient, diagnosis)  # str | None
```

## `pick_recs(patient, diagnosis) -> str | None`

1. Reads `КодМКБ` from the diagnosis dict, normalises to uppercase.
2. Returns `None` immediately if the code is empty or in `_SKIP_CODES` (e.g. `Z00.1`).
3. **Exact match** — scans `manifest.csv` for rows where any comma-separated ICD-10 code in the `МКБ-10` column equals the normalised code.
4. **Prefix fallback** — if no exact match, strips the subcategory (`J20.9` → `J20`) and searches for rows where any code starts with that prefix. Delegates to `IcdPrefixPicker.pick()` to choose among candidates.
5. **Single candidate** — returns its `ID` directly.
6. **Multiple candidates, BM25** — tokenises `НаименованиеМКБ` and each candidate's `Наименование`, computes token overlap. If one candidate scores strictly higher than all others, returns it without an LLM call.
7. **Multiple candidates, all-zero BM25** — falls back to `decide_file_id()` (LLM decider).

## Manifest format

`manifest.csv` must contain at minimum:
- `МКБ-10` — comma-separated ICD-10 codes (e.g. `"J06.0, J06.9"`)
- `ID` — the `file_id` that identifies the guideline PDF
- `Наименование` — guideline title (used for BM25 scoring)
