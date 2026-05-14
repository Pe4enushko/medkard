# DiagnosisValidator

**File:** `audit/diagnosis/validator.py`

Checks a single diagnosis entry against its matched clinical guideline using three parallel LangChain ReAct agents: anamnesis, inspection, and treatment.

## Usage

```python
validator = DiagnosisValidator(visit)
result = await validator.validate_diagnosis(diagnosis)
# DiagnosisAuditResult(anamnesis_issues, inspection_issues, treatment_issues, ...)
```

## Constructor

```python
DiagnosisValidator(visit: dict)
```

Stores the raw visit dict and instantiates a `ClinicRecs` for guideline lookup.

## `validate_diagnosis(diagnosis) -> DiagnosisAuditResult`

1. Calls `ClinicRecs.pick_recs(patient, diagnosis)` to get the guideline `file_id`.
2. If no `file_id` found → logs a warning, returns an empty `DiagnosisAuditResult`.
3. Builds the user message from:
   - `Пациент` fields (key-value lines)
   - Diagnosis formatted as `КодМКБ / НаименованиеМКБ / Детализация / ВыявленВпервые`
   - `ДанныеОсмотра` parsed as `Параметр: Значение` lines
4. Launches three checker agents **in parallel** via `asyncio.gather`:
   - `anamnesis` — tools: `SearchAnamnesisTool`, `SearchGuidelineTool`
   - `inspection` — tools: `SearchInspectionTool`, `SearchGuidelineTool`
   - `treatment` — tools: `SearchTreatmentTool`, `SearchGuidelineTool`
5. Returns `DiagnosisAuditResult` with issues grouped by checker.

## `_run_checker(system_prompt, tools, human_message, checker_label) -> _CheckerRun`

Creates a `create_checker_agent(system_prompt, tools)`, invokes it with:

```python
await agent.ainvoke({"messages": [("user", human_message)]})
```

Extracts the last message content, logs unexpected `finish_reason`, parses the JSON output via `_parse_issues`.

## `_parse_issues(output) -> list[DiagnisisIssue]`

Strips optional markdown code fences, parses JSON array. Each element must have:
- `issue` (str) — the finding text
- `sources` (list) — each with `doc_title`, optional `section`, optional `cite`

Returns an empty list on any parse error.

## DiagnosisAuditResult

```python
@dataclass
class DiagnosisAuditResult:
    anamnesis_issues:  list[DiagnisisIssue]
    inspection_issues: list[DiagnisisIssue]
    treatment_issues:  list[DiagnisisIssue]
    guideline_file_id: str | None
    icd_code:          str
```

`all_issues` property returns the flat concatenation of all three lists.
