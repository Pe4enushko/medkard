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
   - matched by `_CODE_RULES` (begin / middle / end of the code) → its visit type
   - any other code → **no verdict**; step 3 decides
3. Keyword fallback on `Наименование` — for every service the code left undecided, not only for services carrying no code at all:
   - `повторн` → `REPEAT`, `первичн` → `PRIMARY`, `диспансерн` → `DISPENSARY`, `профилактическ` → `PROPHYLACTIC`. «Диспансеризация» под `диспансерн` не подпадает — другая основа, и это как раз ПМО по 404н
4. A service neither step decided contributes nothing. `OTHER` is the answer only when no service contributed anything.

### Таблица `_CODE_RULES` — разбор кода по частям

The middle segment of an NMU code is the **doctor's specialty**, not a property
of the appointment: `B01.023.001` is a neurologist, `B01.031.002` a paediatrician,
`B01.015.001` a cardiologist. Until 2026-08-22 the classifier required the middle
to be `070`, which is the «прочее» group of 804н (освидетельствование на
опьянение, паллиативная помощь, судовой врач, медицинский психолог, патронаж),
so every real card of a normal clinic resolved to `OTHER` and lost 34 of the 42
rules.

Classification is now a small table of segment matches — begin / middle / end,
`None` meaning «any» — checked top to bottom, first match wins:

| begin | middle | end | → |
|---|---|---|---|
| `A` | — | — | `LAB_RESEARCH_INTERVENTION` (wins over the rest of the service) |
| `B01` | `_B01_NOT_A_PAIR` | — | no verdict |
| `B01` | — | `001` / `002` | `PRIMARY` / `REPEAT` |
| `B04` | `_B04_NOT_A_PAIR` | — | no verdict |
| `B04` | — | `001` | `DISPENSARY` (диспансерный приём) |
| `B04` | — | `002` | `PROPHYLACTIC` (профилактический осмотр) |

`B04` splits two different things the order keeps apart: `.001` — диспансерный
приём (168н/192н, наблюдение за хроническим пациентом), `.002` —
профилактический осмотр (404н, ПМО). Пока оба давали `PROPHYLACTIC`, на
диспансерном приёме срабатывали четыре правила 404н про объём ПМО, а правила
про само диспансерное наблюдение — нет: они были объявлены только на первичном
и повторном приёме.

The suffix is only reliable for the first pair of a specialty, and only for
specialties where that pair exists at all. `_B01_NOT_A_PAIR` and
`_B04_NOT_A_PAIR` list the exceptions with the reason for each: `B01.054.001` is
«Осмотр (консультация) врача-физиотерапевта» (a single entry, no primary/repeat
split), `B01.030.002` is «Проведение комплексного аутопсийного исследования»,
`B04.015.001` is «Школа для больных с артериальной гипертензией». Without the
exclusions a pure suffix rule gives 23 wrong verdicts across the order.

Further pairs — участковый, подростковый, детский врач, «беременной»
(`.003/.004`, `.005/.006`) — are deliberately **not** decoded from the code:
in the order they are interleaved with entries that are not appointments, so
guessing by suffix is unsafe. 90 such appointments exist; they are recognised by
the service name instead, where the clinic writes «первичный» / «повторный»
itself.

`scripts/checks/check-nmu-classifier.py <804н.pdf>` audits the table against the order:
it reports contradictions and verdicts the order does not support (both are
errors, non-zero exit) and counts what the table leaves to the name (expected).
The order is the oracle, not a data source — nothing is generated from it into
the repository.

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
6. The model returns one verdict — `{"comment", "violated", "issue"}` — and Python
   attaches the trusted `flag_code` and the regulatory `source`. Атомарный вызов с
   атомарным правилом даёт атомарное решение: массив findings на этом месте позволял
   вернуть несколько замечаний на одно правило, и все получали один и тот же флаг.
   `comment` стоит в схеме первым, чтобы модель выписала факты из карты до решения.
   Применимость правила здесь не спрашивается — её решил шаг 3.
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
| `PROPHYLACTIC` | `prophylactic` | Профилактический осмотр (404н, 211н) |
| `DISPENSARY` | `dispensary` | Диспансерный приём (168н, 192н) |
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
