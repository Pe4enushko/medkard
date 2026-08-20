# FormalValidator

**File:** `audit/formal_structure/validator.py`

Validates the formal structure of a single ambulatory visit record against a rule set loaded from `rules.json`. Determines visit type(s) from NMU codes and service names, filters applicable rules, renders a system prompt, and calls the LLM.

## Usage

```python
validator = FormalValidator()
findings = await validator.validate(visit)
# [{"flag": "MISSING_COMPLAINTS", "issue": "..."}, ...]
```

## Visit type detection — `get_visit_types(visit) -> set[VisitType]`

Each service in `Услуги` is classified independently. The result is the union across all services.

**Classification priority per service:**

1. If `Диагноз.Код` is `Z11.1` → always adds `PROPHYLACTIC_TUBERCULIN` to the result set (independent of services).
2. Scan every field value for an NMU code matching `[ABАВ]\d{2}\.\d{3}\.\d{3}(\.\d{3})?`:
   - `A*` prefix → `LAB_RESEARCH_INTERVENTION`
   - `B04.*` → `PROPHYLACTIC`
   - `B01.070.001` → `PRIMARY`
   - `B01.070.011` or `.012` → `REPEAT`
   - `B01` with any other middle/suffix → `OTHER`
3. If no NMU code found, keyword fallback on `Наименование`:
   - `повторн` → `REPEAT`, `первичн` → `PRIMARY`, `профилактическ` → `PROPHYLACTIC`
4. Services that match nothing contribute `OTHER`.

If `Услуги` is absent or empty, returns `{OTHER}`.

## Rule filtering — `get_rules(visit_types, patient_age, icd_codes=None) -> list[dict]`

Loads `rules.json` (done once at module import). Returns rules where:
- `applies_to.visit_types` contains `"all"` or overlaps with the resolved visit type keys.
- `applies_to.age_group` is `"all"`, or matches the derived group (`"child"` if `age < 18`, `"adult"` otherwise). When `patient_age=None`, age filtering is skipped.

Deduplicates by `flag_code` — the first matching rule wins.

Third filter — ICD codes.  A rule carrying `applies_to.icd_prefixes` applies
only when one of the visit's diagnosis codes (`Диагнозы[].КодМКБ`, passed in as
`icd_codes`) starts with one of those prefixes; comparison is upper-case and
whitespace-trimmed.  Without codes such a rule never applies.  Today only
`dispensary_followup_adult` uses it (the 168н list of conditions subject to
dispensary follow-up).

## LLM call — `validate(visit) -> list[dict]`

1. Calls `get_visit_types(visit)` to resolve visit type set.
2. Reads `Пациент.AGE` → `patient_age` (int or None).
3. Calls `get_rules(visit_types, patient_age)` → applicable rules.
4. Renders `LLM/prompts/formal_structure_validator.txt` with `{rules}` injected.
5. Calls `LLM.validations.validate_visit(system_prompt, visit)` → list of `{flag, issue}` dicts.
6. Runs `_check_nmu_keyword_contradiction(visit)` — a deterministic check that appends `NMU_CODE_CONTRADICTION` if a `B01` NMU suffix contradicts the service name (`.001` + «повторный» or `.002` + «первичный»).

Returns the combined findings list. Empty list means no defects detected.

## VisitType enum

| Value | Rule key | Meaning |
|---|---|---|
| `PRIMARY` | `primary` | Первичный приём |
| `REPEAT` | `repeat` | Повторный приём |
| `PROPHYLACTIC` | `prophylactic` | Профилактический |
| `PROPHYLACTIC_TUBERCULIN` | `prophylactic_tuberculin` | Туберкулинодиагностика (Z11.1) |
| `LAB_RESEARCH_INTERVENTION` | `lab_research_intervention` | A-коды: лаб/инстр/вмешательства |
| `OTHER` | `other` | Не удалось определить |

## Rule file format

`rules.json` is an object, not a bare list:

```json
{
  "revised_at": "2026-08-19",
  "sources_doc": "docs/formal-rules-sources.md",
  "rules": [ … ]
}
```

`revised_at` is the date of the last revision of the whole set; every rule
carries `source_ref` (the exact clause of the regulation it rests on) and
`verified_at` (the date that wording was last checked against the primary
source).  On import the module logs
`formal rules revised_at=… rules=N oldest verified_at=…`, so the age of the
regulatory base is visible in the logs.

The regulations themselves — editions, validity periods, planned replacements
and open verification tails — are listed in `docs/formal-rules-sources.md`.
