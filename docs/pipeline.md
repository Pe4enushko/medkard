# AuditPipeline

**File:** `audit/pipeline.py`

Orchestrates the full audit of a batch of ambulatory visits. Handles deduplication, ICD filtering, formal and diagnosis validation, Excel output, and `done_cards` persistence. Used as an async context manager to keep the DB connection pool alive across all visits in a run.

## Usage

```python
async with AuditPipeline(excel_path="audit_results.xlsx") as pipeline:
    results = await pipeline.run(payload, done_guids=done_guids, ignore_icd=["Z00.1"])
```

## Constructor

```python
AuditPipeline(excel_path: str | Path)
```

Opens an `AuditExcelWriter` for the given path. The `DoneCardsStorage` pool is opened on `__aenter__` and closed on `__aexit__`.

## Methods

### `run(raw_input, done_guids=None, ignore_icd=None) -> list[Result]`

Audits all visits sequentially. Returns one `Result` per audited visit.

### `run_batched(raw_input, num_batches, done_guids=None, ignore_icd=None) -> list[Result]`

Same as `run` but processes up to `num_batches` visits concurrently via `asyncio.Semaphore`.

### `_filter_pending_appointments(appointments, done_guids, ignore_icd) -> (pending, skipped_done, skipped_icd)`

Splits the appointment list into pending vs. skipped. Returns separate counts for visits skipped because their GUID was already in `done_guids`, and visits skipped because all their ICD codes were in `ignore_icd`.

**ICD skip logic:** a visit is skipped only if *every* diagnosis in `Диагнозы` has a `КодМКБ` that appears in `ignore_icd`. If even one diagnosis is not ignored, the visit is audited.

### `_audit_visit(visit) -> Result`

Audits a single visit:

1. Runs `FormalValidator.validate(visit)` → `FormalStructureResult`
2. If no diagnoses: writes Excel row and upserts `done_cards`, returns early.
3. For each diagnosis, runs `DiagnosisValidator.validate_diagnosis(diagnosis)`.
4. Writes Excel row via `_append_excel` (serialised under `_excel_lock`).
5. Upserts a `done_cards` row via `_upsert_done_card` with elapsed `time_ms`.
6. Returns a `Result` containing formal findings and per-diagnosis results.

### `_upsert_done_card(...)`

Serialises the visit as JSON and calls `DoneCardsStorage.upsert`. Token count is recorded as `0` (placeholder — LLM token tracking not yet wired).

## Data flow per visit

```
visit dict
  └─ FormalValidator.validate()       → FormalStructureResult
  └─ DiagnosisValidator.validate_diagnosis() × N  → list[DiagnosisResult]
  └─ AuditExcelWriter.append()        → xlsx row
  └─ DoneCardsStorage.upsert()        → done_cards DB row
  └─ Result(input, formal, diagnosis) → returned to caller
```
