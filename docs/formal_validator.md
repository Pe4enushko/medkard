# FormalValidator

**File:** `audit/formal_structure/validator.py`

Validates the formal structure of a single ambulatory visit record against a rule set loaded from `rules.json`. Determines visit type(s), filters applicable rules, then checks every selected rule in a separate atomic LLM request.

## Usage

```python
validator = FormalValidator()
findings, tokens = await validator.validate(visit)
# [{"flag": "MISSING_COMPLAINTS", "issue": "..."}, ...]
```

## Visit type detection — `get_visit_types(visit) -> set[VisitType]`

Each service in `Услуги` is classified independently. The result is the union across all services.

**Classification priority per service:**

1. If any diagnosis in `Диагнозы[].КодМКБ` is `Z11.1` (compared upper-case, whitespace-trimmed) → always adds `PROPHYLACTIC_TUBERCULIN` to the result set (independent of services).
2. Scan every field value for an NMU code matching `^[ABАВ]\d{2}\.\d{2,3}\.\d{3}(?:\.\d{3})?$` (the middle segment is 2 digits for A-codes and 3 for B-codes, hence `\d{2,3}`):
   - `A*` prefix → `LAB_RESEARCH_INTERVENTION` (thousands of codes; rules narrow them further through `applies_to.service_code_prefixes`)
   - a code listed in `nmu_services.json` → the visit type its 804н kind maps to
   - any other code → **no verdict**; step 3 decides
3. Keyword fallback on `Наименование` — for every service the code left undecided, not only for services carrying no code at all:
   - `повторн` → `REPEAT`, `первичн` → `PRIMARY`, `профилактическ` → `PROPHYLACTIC`
4. A service neither step decided contributes nothing. `OTHER` is the answer only when no service contributed anything.

### `nmu_services.json` — справочник по 804н

The middle segment of an NMU code is the **doctor's specialty**, not a property
of the appointment: `B01.023.001` is a neurologist, `B01.031.002` a paediatrician,
`B01.015.001` a cardiologist. The suffix carries no global meaning either —
`B01.047.001/.002` are терапевт первичный/повторный while `B01.047.010/.011` are
врач по водолазной медицине первичный/повторный, and 96 codes of the `B01`
section are not appointments at all (ежедневный осмотр, ведение родов,
анестезиологическое пособие, освидетельствование, патронаж).

So the type comes from a lookup table generated from the order itself:
`scripts/build-nmu-dictionary.py <804н.pdf>` writes
`src/audit/formal_structure/nmu_services.json` as `{code: {kind, name}}`.
It deliberately holds only ambulatory appointments — `B01` «Прием … первичный /
повторный» and `B04` «Диспансерный / Профилактический прием» — because that is
what an outpatient clinic bills. Everything else is absent on purpose and falls
through to the service name.

**The file stores the order's categories, not ours.** `kind` is one of
`appointment_primary`, `appointment_repeat`, `dispensary`, `prophylactic` — the
distinctions 804н itself draws. The narrowing to medkard's six `VisitType`
values lives in `validator._NMU_KIND_TO_VISIT_TYPE`, next to
`_VISIT_TYPE_RULE_KEY`, so our categorisation stays in one place instead of
leaking into the generator. A kind without a mapping raises at import;
`tests/test_nmu_dictionary.py` guards the boundary in both directions.

Today `dispensary` and `prophylactic` both map to `PROPHYLACTIC`, as before the
dictionary existed. That is a known compromise, not an oversight: 50 of the 93
`B04` appointments in the order are диспансерные, and on such a visit the four
404н rules about the scope of a ПМО fire while the 168н/192н rules — the ones
actually about dispensary follow-up — do not, because they are declared for
`primary`/`repeat`. Splitting the two needs a new `visit_types` key in
`rules.json` and a decision about which rules follow it; tracked in
`docs/tech-debt.md`.

Until 2026-08-22 the classifier recognised `B01.070.*` only, which is the
«прочее» group (врач по медицинской профилактике, паллиативная помощь, судовой
врач, медицинский психолог, патронаж). Every real card of a normal clinic
therefore resolved to `OTHER` and lost 34 of the 42 rules; `B01.070.001`, mapped
to PRIMARY, is in fact «Медицинское освидетельствование на состояние опьянения».

If `Услуги` is absent or empty, returns `{OTHER}`.

## Rule filtering — `get_rules(visit_types, patient_age, icd_codes=None, visit=None) -> list[dict]`

Loads `rules.json` (done once at module import). Returns rules where:
- `applies_to.visit_types` contains `"all"` or overlaps with the resolved visit type keys.
- `applies_to.age_group` is `"all"`, or matches the derived group (`"child"` if `age < 18`, `"adult"` otherwise — the boundary 404н draws with «граждане в возрасте 18 лет и старше»). When `patient_age=None` the card gave no usable age and **only `age_group="all"` rules are kept**: an unknown age must never widen the rule set into the wrong cohort. `Пациент.AGE` is read by `parsers.json_parser.patient_age`, the single parser shared with the clinical-guideline lookup — `AGE = 0` is an infant, not a missing value.

Deduplicates by `flag_code` — the first matching rule wins. Two rules may share a flag on purpose (`test_flag_codes_are_unique_except_shared_pairs`), so such a pair must stay mutually exclusive through `applies_to`; otherwise the second one is dropped silently.

Third filter — ICD codes.  A rule carrying `applies_to.icd_prefixes` applies
only when one of the visit's diagnosis codes (`Диагнозы[].КодМКБ`, passed in as
`icd_codes`) starts with one of those prefixes; comparison is upper-case and
whitespace-trimmed.  Without codes such a rule never applies.  Today only
`dispensary_followup_adult` uses it (the 168н list of conditions subject to
dispensary follow-up).

Service filters are evaluated deterministically from `visit["Услуги"]` before
an LLM call:

- `applies_to.service_code_prefixes` requires at least one matching NMU code;
- `applies_to.service_name_keywords` requires at least one keyword in a service name.

The filters follow the type split in order 804n. In particular, laboratory
results use `A09.*`, ultrasound `A04.*`, radiology `A06.*`, functional studies
`A05.*`/`A12.*`, ambulatory operations `A16.*`, and drug administration is an
`A11.*` service whose name contains an administration/injection marker. `A02.*`
and `A03.*` are examinations and are not treated as generic injections.
The separate-consent rule is sent only for potentially invasive performed
services (`A03.*`, `A11.*`, `A16.*`); a drug prescription inside an ordinary
`B01.*` consultation does not make that rule applicable.

## LLM call — `validate(visit) -> tuple[list[dict], int]`

1. Calls `get_visit_types(visit)` to resolve visit type set.
2. Reads `Пациент.AGE` through `parsers.json_parser.patient_age` → int or None (an unreadable value is logged).
3. Calls `get_rules(visit_types, patient_age, icd_codes, visit)` → applicable rules.
4. Starts `LLM.validations.validate_rule(...)` concurrently for every rule.
5. Every request has the same cacheable prefix: static system prompt, then the complete visit JSON; only the final user message with one rule differs.
6. The model returns the established findings array: `[]` or `[{"flag", "issue", "comment"}]`. Python attaches the trusted `flag_code` for the current rule and the regulatory `source`.
7. Runs `_check_nmu_keyword_contradiction(visit)` — a deterministic check that appends `NMU_CODE_CONTRADICTION` if a `B01` NMU suffix contradicts the service name (`.001` + «повторный» or `.002` + «первичный»).

Returns the combined findings list and summed token count. An empty findings
list means no defects were detected.

The formal validator does not query GRLS. Its task is to apply the selected
documentation and regulatory rule atomically; drug-registry retrieval belongs
to the diagnosis graph's treatment check, where drug identity and status are
actually part of the evidence.

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
`[formal] formal rules revised_at=… rules=N oldest verified_at=…`, so the age of the
regulatory base is visible in the logs.

The regulations themselves — editions, validity periods, planned replacements
and open verification tails — are listed in `docs/formal-rules-sources.md`.
