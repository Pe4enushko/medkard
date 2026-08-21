# Ревизия нормативной базы правил формальной проверки — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Перевести `rules.json` на версионируемый формат с прослеживаемостью до
пункта НПА (`revised_at`/`source_ref`/`verified_at`), переобосновать правила
утратившего силу 203н-2017 на действующие приказы, добавить правила по 1094н,
404н, 168н/192н, 1051н, 1089н и научить `get_rules` фильтру по префиксам МКБ-10.

**Architecture:** Изменения почти целиком декларативные — данные в
`rules.json` и новый справочник `docs/formal-rules-sources.md`. В коде
`validator.py` три точки: загрузка файла (список → объект с ключом `rules`),
новый фильтр `icd_prefixes` в `get_rules` (принимает готовый список кодов
МКБ третьим аргументом), и вызов из `validate`, который эти коды достаёт из
карты. LLM-промпт, формат находок, контракты `validate()` и Excel-выгрузки не
меняются.

**Tech Stack:** Python 3, pytest (`asyncio_mode = auto`, `pythonpath = src`),
JSON-данные без схема-библиотек, стандартная библиотека.

**Spec:** `docs/superpowers/specs/2026-08-17-formal-rules-npa-revision-design.md`

## Global Constraints

- Ветка: отдельная **от `specs-2026-08-17`** (не от `release`) — так решено в
  спеке (шапка) и в HANDOFF §3.
- Контракты не ломать: `FormalValidator.validate(visit) -> (findings, tokens)`,
  формат находки `{"flag": ..., "issue": ..., "source": ...}`, поле
  `formal_result` в БД, колонки Excel. Единственный публичный сигнатурный
  сдвиг — `get_rules`, у которого нет внешних вызывающих (проверено:
  единственный вызов — `validate()` в том же файле).
- `flag_code` существующих правил **не переименовывать**: исторические
  результаты в БД на них ссылаются (спека §4.6 про
  `diagnosis_justification_presence`).
- Severity — только из множества `{критичный, значительный, незначительный}`.
- `visit_types` — только из `{all, primary, repeat, prophylactic,
  prophylactic_tuberculin, lab_research_intervention, other}`.
- `age_group` — только из `{all, child, adult}`.
- Все даты — ISO `YYYY-MM-DD`. `verified_at` каждого правила `<= revised_at`.
- Дата ревизии в этом плане: **`2026-08-19`** (используется как `revised_at` и
  как `verified_at` везде, кроме случаев, указанных в задачах явно).
- Вне скоупа (спека §4.7): 947н, 286н/972н, 29н, 785н; правила для визита
  `OTHER`; КР-статус ≠ «Применяется».
- Тесты запускаются из корня репозитория: `pytest tests/<файл> -v`. БД и LLM
  тестам этого плана не нужны.
- Правила `diagnosis_required`, `diagnosis_justification_presence`,
  `too_general_icd_for_rich_detail` сейчас стоят на `age_group: child`, что
  выглядит наследием, но спека их не трогает — **не менять**, кроме явного
  указания в Задаче 8 (`treatment_dosage_clarity`: `child` → `all`).

---

## Обзор файлов

| Файл | Что с ним |
|---|---|
| `src/audit/formal_structure/rules.json` | Формат список → объект; у всех правил `source_ref`/`verified_at`; смена `source` у 12 правил; 9 новых правил |
| `src/audit/formal_structure/validator.py` | Загрузка `["rules"]` + лог; `icd_prefixes` в `get_rules`; фикс чтения диагнозов для Z11.1; фикс `NMU_RE` для A-кодов |
| `docs/formal-rules-sources.md` | Новый: реестр НПА (Приложение Б спеки) |
| `docs/formal_validator.md` | Обновить разделы «Rule filtering» и формат правил |
| `docs/formal-rules-revision-log.md` | Запись о применённой ревизии в раздел «Нормативка» |
| `tests/test_formal_rules_schema.py` | Новый: инварианты файла правил (без БД/LLM) |
| `tests/test_formal_get_rules.py` | Новый: фильтрация по типу/возрасту/МКБ, рендер промпта |
| `tests/test_formal_validator_meta.py` | Новый: лог загрузки с `revised_at` |
| `tests/test_formal_visit_types.py` | Новый: определение типа визита — Z11.1 и A-коды услуг |

**Порядок задач умышленный:** сначала механика (задачи 1–4), потом данные
правил (5–10), затем два фикса детектора типа визита (11–12), потом
документация (13–14) и стендовый гейт (15). Механика меняет формат файла, и все
последующие задачи с данными опираются на уже зелёные схема-тесты; фиксы
детектора идут после данных, чтобы их эффект на отчёты читался отдельно.

---

### Task 1: Формат `rules.json` — объект вместо списка

Спека §2.1. Меняем только обёртку файла и загрузку; содержимое правил на этом
шаге не трогаем.

**Files:**
- Modify: `src/audit/formal_structure/rules.json` (обёртка вокруг всего списка)
- Modify: `src/audit/formal_structure/validator.py:40` (загрузка `_RULES`)
- Test: `tests/test_formal_rules_schema.py` (создать)

**Interfaces:**
- Consumes: ничего (первая задача).
- Produces: модульные константы в `validator.py` —
  `_RULES_DOC: dict` (весь разобранный файл), `_RULES: list[dict]`
  (`_RULES_DOC["rules"]`), `_REVISED_AT: str`. Задачи 2 и 4 их используют.

- [ ] **Step 1: Написать падающий тест**

Создать `tests/test_formal_rules_schema.py`:

```python
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_RULES_PATH = ROOT / "src" / "audit" / "formal_structure" / "rules.json"

_ISO = __import__("re").compile(r"^\d{4}-\d{2}-\d{2}$")


def _doc() -> dict:
    return json.loads(_RULES_PATH.read_text(encoding="utf-8"))


def test_file_is_object_with_meta():
    doc = _doc()
    assert isinstance(doc, dict), "rules.json должен быть объектом, а не списком"
    assert _ISO.match(doc["revised_at"]), f"revised_at не ISO: {doc['revised_at']!r}"
    assert doc["sources_doc"] == "docs/formal-rules-sources.md"
    assert isinstance(doc["rules"], list) and doc["rules"]
```

- [ ] **Step 2: Прогнать тест и убедиться, что он падает**

Run: `pytest tests/test_formal_rules_schema.py -v`
Expected: FAIL — `AssertionError: rules.json должен быть объектом, а не списком`
(сейчас в файле список).

- [ ] **Step 3: Обернуть `rules.json`**

Выполнить из корня репозитория (скрипт сохраняет порядок правил и русский текст):

```bash
python3 - <<'EOF'
import json
from pathlib import Path

p = Path("src/audit/formal_structure/rules.json")
rules = json.loads(p.read_text(encoding="utf-8"))
assert isinstance(rules, list), "файл уже обёрнут"
doc = {
    "revised_at": "2026-08-19",
    "sources_doc": "docs/formal-rules-sources.md",
    "rules": rules,
}
p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
EOF
```

- [ ] **Step 4: Поправить загрузку в `validator.py`**

Заменить строку 40 (`_RULES: list[dict] = json.loads(...)`) на:

```python
_RULES_DOC: dict = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
_RULES: list[dict] = _RULES_DOC["rules"]
_REVISED_AT: str = _RULES_DOC["revised_at"]
```

- [ ] **Step 5: Прогнать тест — должен пройти**

Run: `pytest tests/test_formal_rules_schema.py -v`
Expected: PASS

- [ ] **Step 6: Убедиться, что модуль импортируется и правила читаются**

Run: `python3 -c "import sys; sys.path.insert(0,'src'); from audit.formal_structure.validator import _RULES, _REVISED_AT; print(len(_RULES), _REVISED_AT)"`
Expected: `31 2026-08-19`

- [ ] **Step 7: Коммит**

```bash
git add src/audit/formal_structure/rules.json src/audit/formal_structure/validator.py tests/test_formal_rules_schema.py
git commit -m "feat(formal): rules.json — объект с revised_at/sources_doc"
```

---

### Task 2: Лог при загрузке правил

Спека §2.1: при загрузке — строка
`formal rules revised_at=… rules=N oldest verified_at=…`.

`verified_at` у правил появится только в задачах 5–10, поэтому «самая старая
дата сверки» считается устойчиво к её отсутствию: правила без `verified_at`
в расчёт не берутся, а если таких дат нет вовсе — в лог идёт `none`.

**Files:**
- Modify: `src/audit/formal_structure/validator.py` (сразу после загрузки правил)
- Test: `tests/test_formal_validator_meta.py` (создать)

**Interfaces:**
- Consumes: `_RULES`, `_REVISED_AT` из Задачи 1.
- Produces: запись в логгер `audit.formal_structure.validator` на уровне INFO
  при импорте модуля.

- [ ] **Step 1: Написать падающий тест**

Создать `tests/test_formal_validator_meta.py`:

```python
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_load_logs_revision_meta(caplog):
    """Модуль пишет revised_at и число правил при загрузке."""
    import audit.formal_structure.validator as v

    with caplog.at_level("INFO", logger="audit.formal_structure.validator"):
        importlib.reload(v)

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "revised_at=2026-08-19" in text
    assert f"rules={len(v._RULES)}" in text
    assert "oldest verified_at=" in text
```

- [ ] **Step 2: Прогнать тест и убедиться, что он падает**

Run: `pytest tests/test_formal_validator_meta.py -v`
Expected: FAIL — `AssertionError` (в логе нет `revised_at=`).

- [ ] **Step 3: Добавить лог в `validator.py`**

Сразу после блока `_FLAG_SOURCE` / `_ALL_FLAGS` вставить:

```python
_VERIFIED_DATES: list[str] = sorted(r["verified_at"] for r in _RULES if r.get("verified_at"))
logger.info(
    "[formal] formal rules revised_at=%s rules=%d oldest verified_at=%s",
    _REVISED_AT,
    len(_RULES),
    _VERIFIED_DATES[0] if _VERIFIED_DATES else "none",
)
```

- [ ] **Step 4: Прогнать тест — должен пройти**

Run: `pytest tests/test_formal_validator_meta.py -v`
Expected: PASS

- [ ] **Step 5: Коммит**

```bash
git add src/audit/formal_structure/validator.py tests/test_formal_validator_meta.py
git commit -m "feat(formal): лог revised_at/rules/oldest verified_at при загрузке"
```

---

### Task 3: Фильтр `icd_prefixes` в `get_rules`

Спека §2.3. Решение по сигнатуре (согласовано с пользователем): `get_rules`
принимает **готовый список кодов МКБ**, разбор формата карты остаётся в
`validate()`. Так `get_rules` не знает про формат карты и тестируется без неё.

**Files:**
- Modify: `src/audit/formal_structure/validator.py` (метод `get_rules`, ~строка 197)
- Modify: `src/audit/formal_structure/validator.py` (вызов в `validate`, ~строка 306)
- Test: `tests/test_formal_get_rules.py` (создать)

**Interfaces:**
- Consumes: `_RULES` из Задачи 1.
- Produces: новая сигнатура
  `FormalValidator.get_rules(visit_types: set[VisitType], patient_age: int | None, icd_codes: list[str] | None = None) -> list[dict]`.
  Задачи 6 и 7 пишут правила, опирающиеся на этот фильтр.
  Семантика: правило с непустым `applies_to.icd_prefixes` проходит, только если
  хотя бы один код из `icd_codes` начинается с одного из префиксов (сравнение
  в верхнем регистре, пробелы обрезаются). `icd_codes=None` или пустой список
  → правило с `icd_prefixes` **не применяется**.

- [ ] **Step 1: Написать падающий тест**

Создать `tests/test_formal_get_rules.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.formal_structure.validator import FormalValidator, VisitType


def _rule(**over):
    """Минимальное правило; поля перекрываются через kwargs."""
    base = {
        "rule_id": "r",
        "flag_code": "FLAG",
        "rule_type": "required_field",
        "applies_to": {"visit_types": ["all"], "specialties": ["all"], "age_group": "all"},
        "expectation": "ожидание",
    }
    base.update(over)
    return base


def _flags(rules):
    return [r["flag_code"] for r in rules]


def test_icd_prefix_matches(monkeypatch):
    import audit.formal_structure.validator as v

    rule = _rule(
        rule_id="dispensary",
        flag_code="ДИСПАНСЕРНОЕ",
        applies_to={
            "visit_types": ["primary"],
            "specialties": ["all"],
            "age_group": "adult",
            "icd_prefixes": ["I10", "I11"],
        },
    )
    monkeypatch.setattr(v, "_RULES", [rule])

    got = FormalValidator().get_rules({VisitType.PRIMARY}, 40, ["I11.9"])
    assert _flags(got) == ["ДИСПАНСЕРНОЕ"]


def test_icd_prefix_does_not_match(monkeypatch):
    import audit.formal_structure.validator as v

    rule = _rule(
        flag_code="ДИСПАНСЕРНОЕ",
        applies_to={
            "visit_types": ["primary"],
            "specialties": ["all"],
            "age_group": "adult",
            "icd_prefixes": ["I10", "I11"],
        },
    )
    monkeypatch.setattr(v, "_RULES", [rule])

    assert FormalValidator().get_rules({VisitType.PRIMARY}, 40, ["J06.9"]) == []


def test_icd_prefix_without_codes_is_skipped(monkeypatch):
    import audit.formal_structure.validator as v

    rule = _rule(
        flag_code="ДИСПАНСЕРНОЕ",
        applies_to={
            "visit_types": ["primary"],
            "specialties": ["all"],
            "age_group": "all",
            "icd_prefixes": ["I10"],
        },
    )
    monkeypatch.setattr(v, "_RULES", [rule])

    assert FormalValidator().get_rules({VisitType.PRIMARY}, 40, None) == []
    assert FormalValidator().get_rules({VisitType.PRIMARY}, 40, []) == []


def test_icd_prefix_is_case_and_space_insensitive(monkeypatch):
    import audit.formal_structure.validator as v

    rule = _rule(
        flag_code="ДИСПАНСЕРНОЕ",
        applies_to={
            "visit_types": ["primary"],
            "specialties": ["all"],
            "age_group": "all",
            "icd_prefixes": ["I11"],
        },
    )
    monkeypatch.setattr(v, "_RULES", [rule])

    assert _flags(FormalValidator().get_rules({VisitType.PRIMARY}, 40, [" i11.9 "])) == ["ДИСПАНСЕРНОЕ"]


def test_rule_without_icd_prefixes_ignores_codes(monkeypatch):
    import audit.formal_structure.validator as v

    monkeypatch.setattr(v, "_RULES", [_rule(flag_code="ОБЫЧНОЕ")])

    assert _flags(FormalValidator().get_rules({VisitType.PRIMARY}, 40, None)) == ["ОБЫЧНОЕ"]
    assert _flags(FormalValidator().get_rules({VisitType.PRIMARY}, 40, ["J06.9"])) == ["ОБЫЧНОЕ"]
```

- [ ] **Step 2: Прогнать тесты и убедиться, что они падают**

Run: `pytest tests/test_formal_get_rules.py -v`
Expected: FAIL — `TypeError: get_rules() takes 3 positional arguments but 4 were given`

- [ ] **Step 3: Реализовать фильтр**

В `validator.py` заменить сигнатуру и тело `get_rules`. Новый вид метода
целиком:

```python
    def get_rules(
        self,
        visit_types: set[VisitType],
        patient_age: int | None,
        icd_codes: list[str] | None = None,
    ) -> list[dict]:
        """Return rules applicable to the given visit types, age and ICD codes.

        Age group matching: a rule passes if its ``age_group`` is ``"all"``,
        or matches the derived group (``"child"`` if age < 18, ``"adult"`` otherwise).
        ``patient_age=None`` skips age filtering (matches any age_group).

        ICD matching: a rule carrying ``applies_to.icd_prefixes`` passes only if
        one of ``icd_codes`` starts with one of those prefixes.  Without codes
        such a rule never applies.
        """
        type_keys = {_VISIT_TYPE_RULE_KEY[vt] for vt in visit_types}
        age_group: str | None = None
        if patient_age is not None:
            age_group = "child" if patient_age < 18 else "adult"

        codes = [c.strip().upper() for c in (icd_codes or []) if c and c.strip()]

        seen: set[str] = set()
        rules: list[dict] = []
        for rule in _RULES:
            applies = rule.get("applies_to", {})
            visit_type_applies = applies.get("visit_types", [])
            if not ("all" in visit_type_applies or type_keys & set(visit_type_applies)):
                continue
            if age_group is not None:
                rule_age = applies.get("age_group", "all")
                if rule_age != "all" and rule_age != age_group:
                    continue
            prefixes = applies.get("icd_prefixes") or []
            if prefixes and not any(c.startswith(p) for c in codes for p in prefixes):
                continue
            fc = rule.get("flag_code", "")
            if fc not in seen:
                seen.add(fc)
                rules.append(rule)
        return rules
```

Префиксы в `rules.json` хранятся уже в верхнем регистре (это проверяет
схема-тест Задачи 4), поэтому нормализуется только сторона кодов визита.

- [ ] **Step 4: Прогнать тесты — должны пройти**

Run: `pytest tests/test_formal_get_rules.py -v`
Expected: PASS (5 тестов)

- [ ] **Step 5: Прокинуть коды из `validate`**

В методе `validate` заменить строку `rules = self.get_rules(visit_types, patient_age)` на:

```python
        icd_codes: list[str] = [
            str(d.get("КодМКБ") or "")
            for d in (visit.get("Диагнозы") or [])
            if isinstance(d, dict)
        ]
        rules = self.get_rules(visit_types, patient_age, icd_codes)
```

- [ ] **Step 6: Убедиться, что весь модуль по-прежнему импортируется**

Run: `pytest tests/test_formal_get_rules.py tests/test_formal_rules_schema.py tests/test_formal_validator_meta.py -v`
Expected: PASS

- [ ] **Step 7: Коммит**

```bash
git add src/audit/formal_structure/validator.py tests/test_formal_get_rules.py
git commit -m "feat(formal): фильтр правил по префиксам МКБ-10 (applies_to.icd_prefixes)"
```

---

### Task 4: Схема-тесты набора правил

Спека §5. Тесты, которые сторожат инварианты всех дальнейших задач с данными.
Снимок количества правил и списка флагов задаётся **текущим** состоянием (31
правило) и обновляется в задачах, которые правила добавляют.

**Files:**
- Modify: `tests/test_formal_rules_schema.py` (дописать к тесту из Задачи 1)

**Interfaces:**
- Consumes: формат файла из Задачи 1, семантику `icd_prefixes` из Задачи 3.
- Produces: константы `EXPECTED_RULE_COUNT` и `EXPECTED_FLAGS` в тестовом файле,
  которые задачи 5–10 обновляют по мере добавления правил.

- [ ] **Step 1: Дописать падающие тесты**

Добавить в `tests/test_formal_rules_schema.py`:

```python
_SEVERITIES = {"критичный", "значительный", "незначительный"}
_AGE_GROUPS = {"all", "child", "adult"}
_VISIT_TYPES = {
    "all", "primary", "repeat", "prophylactic",
    "prophylactic_tuberculin", "lab_research_intervention", "other",
}

# Снимок: обновляется задачами, которые добавляют правила.
EXPECTED_RULE_COUNT = 31

# Флаги, сознательно разделённые между child/adult-парой правил.
_SHARED_FLAGS = {"ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО"}


def test_every_rule_has_source_ref_and_verified_at():
    doc = _doc()
    revised = doc["revised_at"]
    for r in doc["rules"]:
        rid = r["rule_id"]
        assert r.get("source_ref", "").strip(), f"{rid}: пустой source_ref"
        va = r.get("verified_at", "")
        assert _ISO.match(va), f"{rid}: verified_at не ISO: {va!r}"
        assert va <= revised, f"{rid}: verified_at {va} позже revised_at {revised}"


def test_rule_ids_are_unique():
    ids = [r["rule_id"] for r in _doc()["rules"]]
    assert len(ids) == len(set(ids)), f"дубли rule_id: {sorted({i for i in ids if ids.count(i) > 1})}"


def test_flag_codes_are_unique_except_shared_pairs():
    flags = [r["flag_code"] for r in _doc()["rules"]]
    dupes = {f for f in flags if flags.count(f) > 1}
    assert dupes <= _SHARED_FLAGS, f"неожиданные дубли flag_code: {sorted(dupes - _SHARED_FLAGS)}"


def test_enums_are_valid():
    for r in _doc()["rules"]:
        rid = r["rule_id"]
        assert r["severity"] in _SEVERITIES, f"{rid}: severity {r['severity']!r}"
        applies = r["applies_to"]
        assert applies.get("age_group", "all") in _AGE_GROUPS, f"{rid}: age_group"
        assert set(applies["visit_types"]) <= _VISIT_TYPES, f"{rid}: visit_types {applies['visit_types']}"


def test_icd_prefixes_are_upper_nonempty_strings():
    for r in _doc()["rules"]:
        prefixes = r["applies_to"].get("icd_prefixes")
        if prefixes is None:
            continue
        assert isinstance(prefixes, list) and prefixes, f"{r['rule_id']}: пустой icd_prefixes"
        for p in prefixes:
            assert isinstance(p, str) and p.strip(), f"{r['rule_id']}: пустой префикс"
            assert p == p.upper().strip(), f"{r['rule_id']}: префикс {p!r} не в верхнем регистре"


def test_rule_count_snapshot():
    assert len(_doc()["rules"]) == EXPECTED_RULE_COUNT


def test_no_rule_references_retired_203n():
    """203н-2017 утратил силу; ярлык source больше не используется (спека §4.6)."""
    offenders = [r["rule_id"] for r in _doc()["rules"] if r.get("source") == "203n"]
    assert not offenders, f"правила всё ещё на ярлыке 203n: {offenders}"
```

- [ ] **Step 2: Прогнать и убедиться, что падает ровно два теста**

Run: `pytest tests/test_formal_rules_schema.py -v`
Expected: FAIL — `test_every_rule_has_source_ref_and_verified_at` (полей ещё
нет ни у одного правила) и `test_no_rule_references_retired_203n` (7 правил на
`203n`). Остальные тесты PASS.

Эти два падения — красный «список работ» для задач 5–10. Они станут зелёными
в Задаче 10, когда последнее правило получит `source_ref`/`verified_at`.

- [ ] **Step 3: Коммит (тесты фиксируются красными намеренно)**

```bash
git add tests/test_formal_rules_schema.py
git commit -m "test(formal): схема-инварианты rules.json (2 теста красные до ревизии правил)"
```

---

### Task 5: Переобоснование правил, снятых с 203н

Спека §4.6. Семь правил переезжают с утратившего силу 203н-2017 на
действующие НПА. `expectation` **не меняется** ни у одного (спека: они не
опирались на текст 203н). Меняются `source`, `source_ref`, `verified_at` и — у
одного правила — `severity`.

`treatment_dosage_clarity` из этой семёрки обрабатывается отдельно в Задаче 8
(там же меняется его `expectation` и `age_group`) — здесь его не трогаем.

**Files:**
- Modify: `src/audit/formal_structure/rules.json`
- Test: `tests/test_formal_rules_schema.py` (уже написан, Задача 4)

**Interfaces:**
- Consumes: формат файла (Задача 1), схема-тесты (Задача 4).
- Produces: правила `repeat_needs_context_or_dynamics`,
  `diagnosis_should_be_supported`, `management_should_follow_diagnosis`,
  `objective_exam_required`, `followup_needed_for_nontrivial_case`,
  `diagnosis_justification_presence` без ярлыка `203n`.

- [ ] **Step 1: Применить правки скриптом**

Выполнить из корня репозитория:

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("src/audit/formal_structure/rules.json")
doc = json.loads(p.read_text(encoding="utf-8"))
by_id = {r["rule_id"]: r for r in doc["rules"]}

patch = {
    "repeat_needs_context_or_dynamics": {
        "source": "274n",
        "source_ref": (
            "приказ МЗ РФ от 13.05.2025 № 274н, прил. 1 (форма 025/у), раздел "
            "«Медицинское наблюдение в динамике» (строки «Жалобы», «Данные наблюдения "
            "в динамике», «Назначения (исследования, консультации)», «Лекарственные "
            "препараты, физиотерапия»); прил. 2, п. 14, п. 15"
        ),
    },
    "diagnosis_should_be_supported": {
        "source": "323-FZ",
        "source_ref": (
            "Федеральный закон от 21.11.2011 № 323-ФЗ, ст. 70 ч. 5 (диагноз основан "
            "на всестороннем обследовании), ч. 6 (состав диагноза); ст. 2 п. 21"
        ),
    },
    "management_should_follow_diagnosis": {
        "source": "323-FZ",
        "source_ref": (
            "Федеральный закон от 21.11.2011 № 323-ФЗ, ст. 37 ч. 1 п. 3 (клинические "
            "рекомендации), п. 4 (стандарты медицинской помощи); ст. 70 ч. 2"
        ),
    },
    "objective_exam_required": {
        "source": "274n",
        "source_ref": (
            "приказ МЗ РФ от 13.05.2025 № 274н, прил. 1 (форма 025/у), раздел «Записи "
            "врачей-специалистов» (строки «Жалобы», «Анамнез заболевания, жизни», "
            "«Объективные данные», «Диагноз основного заболевания / код по МКБ»); "
            "прил. 2, п. 13"
        ),
    },
    "followup_needed_for_nontrivial_case": {
        "source": "323-FZ",
        "source_ref": (
            "Федеральный закон от 21.11.2011 № 323-ФЗ, ст. 70 ч. 2; приказ МЗ РФ от "
            "15.03.2022 № 168н, п. 4, п. 13; приказ МЗ РФ от 13.05.2025 № 274н, прил. 1 "
            "(форма 025/у), строки «Назначения (исследования, консультации)», "
            "«Диспансерное наблюдение»"
        ),
    },
    "diagnosis_justification_presence": {
        "source": "323-FZ",
        "source_ref": (
            "Федеральный закон от 21.11.2011 № 323-ФЗ, ст. 70 ч. 5 — ближайшая норма; "
            "прямой нормы «оформить обоснование диагноза записью» нет, требование "
            "удержано как внутренний стандарт качества"
        ),
        "severity": "незначительный",
    },
}

for rid, fields in patch.items():
    rule = by_id[rid]
    assert rule["source"] == "203n", f"{rid}: ожидался source=203n, а там {rule['source']!r}"
    rule.update(fields)
    rule["verified_at"] = "2026-08-19"

p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("patched:", ", ".join(patch))
PY
```

- [ ] **Step 2: Проверить, что на 203н остался только `treatment_dosage_clarity`**

Run:
```bash
python3 -c "
import json
d=json.load(open('src/audit/formal_structure/rules.json'))
print([r['rule_id'] for r in d['rules'] if r.get('source')=='203n'])"
```
Expected: `['treatment_dosage_clarity']` (его переносит Задача 8).

- [ ] **Step 3: Прогнать схема-тесты**

Run: `pytest tests/test_formal_rules_schema.py -v`
Expected: `test_no_rule_references_retired_203n` по-прежнему FAIL (остался
`treatment_dosage_clarity`), `test_every_rule_has_source_ref_and_verified_at`
FAIL (у большинства правил полей ещё нет). Остальные PASS — важно, что
`test_enums_are_valid` не сломался после смены severity.

- [ ] **Step 4: Коммит**

```bash
git add src/audit/formal_structure/rules.json
git commit -m "feat(formal): 6 правил сняты с утратившего силу 203н-2017 на 274н/323-ФЗ"
```

---

### Task 6: Новые правила по 1094н (назначение лекарственных препаратов)

Спека §4.1. Два новых правила. Правка `treatment_dosage_clarity` — Задача 8.

**Files:**
- Modify: `src/audit/formal_structure/rules.json`
- Modify: `tests/test_formal_rules_schema.py` (`EXPECTED_RULE_COUNT`)

**Interfaces:**
- Consumes: формат файла (Задача 1).
- Produces: правила `prescription_by_trade_name`,
  `off_standard_prescription_needs_vk` с флагами
  `НАЗНАЧЕНИЕ_ПО_ТОРГОВОМУ_БЕЗ_МНН`, `НАЗНАЧЕНИЕ_ВНЕ_СТАНДАРТА_БЕЗ_ВК`.

- [ ] **Step 1: Добавить правила**

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("src/audit/formal_structure/rules.json")
doc = json.loads(p.read_text(encoding="utf-8"))

new = [
    {
        "rule_id": "prescription_by_trade_name",
        "source": "1094n",
        "source_ref": "приказ МЗ РФ от 24.11.2021 № 1094н, прил. 1, п. 5",
        "verified_at": "2026-08-19",
        "rule_type": "content_analysis",
        "applies_to": {
            "visit_types": ["primary", "repeat"],
            "specialties": ["all"],
            "age_group": "all",
        },
        "targets": ["treatment_plan"],
        "expectation": (
            "Лекарственный препарат назначен по международному непатентованному "
            "наименованию (или группировочному/химическому наименованию). Назначение "
            "только по торговому наименованию допустимо, если у препарата нет МНН, "
            "либо при наличии в записи решения врачебной комиссии."
        ),
        "flag_code": "НАЗНАЧЕНИЕ_ПО_ТОРГОВОМУ_БЕЗ_МНН",
        "severity": "незначительный",
    },
    {
        "rule_id": "off_standard_prescription_needs_vk",
        "source": "1094n",
        "source_ref": "приказ МЗ РФ от 24.11.2021 № 1094н, прил. 1, п. 5, п. 32",
        "verified_at": "2026-08-19",
        "rule_type": "conditional_required",
        "applies_to": {
            "visit_types": ["primary", "repeat"],
            "specialties": ["all"],
            "age_group": "all",
        },
        "targets": ["treatment_plan"],
        "condition": (
            "Если в записи назначен препарат вне инструкции по применению, вне "
            "клинических рекомендаций или стандарта медицинской помощи, либо "
            "одновременно назначено 5 и более препаратов, либо назначены "
            "наркотические/психотропные средства впервые."
        ),
        "expectation": (
            "В записи должна быть отметка о решении врачебной комиссии "
            "(номер и/или дата протокола либо прямое указание на врачебную комиссию)."
        ),
        "flag_code": "НАЗНАЧЕНИЕ_ВНЕ_СТАНДАРТА_БЕЗ_ВК",
        "severity": "значительный",
    },
]

have = {r["rule_id"] for r in doc["rules"]}
for r in new:
    assert r["rule_id"] not in have, f"{r['rule_id']} уже есть"
doc["rules"].extend(new)

p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("rules:", len(doc["rules"]))
PY
```

Expected вывод: `rules: 33`

- [ ] **Step 2: Обновить снимок в тесте**

В `tests/test_formal_rules_schema.py` заменить `EXPECTED_RULE_COUNT = 31` на:

```python
EXPECTED_RULE_COUNT = 33
```

- [ ] **Step 3: Прогнать схема-тесты**

Run: `pytest tests/test_formal_rules_schema.py -v`
Expected: `test_rule_count_snapshot` PASS, `test_enums_are_valid` PASS,
`test_flag_codes_are_unique_except_shared_pairs` PASS. По-прежнему красные —
только два теста из Задачи 4.

- [ ] **Step 4: Коммит**

```bash
git add src/audit/formal_structure/rules.json tests/test_formal_rules_schema.py
git commit -m "feat(formal): правила 1094н — МНН и врачебная комиссия при назначении вне стандарта"
```

---

### Task 7: Правила по 404н (профилактический осмотр взрослых)

Спека §4.2. Четыре новых правила, все `visit_types: [prophylactic]`,
`age_group: adult`. Существующие `prophylactic_*`-исключения (211н) остаются
детскими и не трогаются.

**Files:**
- Modify: `src/audit/formal_structure/rules.json`
- Modify: `tests/test_formal_rules_schema.py` (`EXPECTED_RULE_COUNT`)
- Modify: `tests/test_formal_get_rules.py` (тест разделения adult/child)

**Interfaces:**
- Consumes: `get_rules` с фильтром по возрасту (существующий).
- Produces: правила `adult_prophylactic_scope`,
  `adult_prophylactic_health_group`, `adult_prophylactic_counselling`,
  `adult_prophylactic_marker`.

- [ ] **Step 1: Добавить правила**

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("src/audit/formal_structure/rules.json")
doc = json.loads(p.read_text(encoding="utf-8"))

_APPLIES = {"visit_types": ["prophylactic"], "specialties": ["all"], "age_group": "adult"}

new = [
    {
        "rule_id": "adult_prophylactic_scope",
        "source": "404n",
        "source_ref": "приказ МЗ РФ от 27.04.2021 № 404н, п. 16 (пп. 1–12), п. 20",
        "verified_at": "2026-08-19",
        "rule_type": "required_field",
        "applies_to": dict(_APPLIES),
        "targets": ["objective_exam", "state", "recommendations"],
        "expectation": (
            "В записи профилактического осмотра взрослого отражены: анкетирование "
            "(сбор анамнеза и факторов риска), антропометрия (рост, вес, индекс массы "
            "тела, окружность талии), артериальное давление, оценка сердечно-сосудистого "
            "риска (относительного до 40 лет, абсолютного с 40 лет) и осмотр на "
            "визуальные локализации онкологических заболеваний. Отсутствие результата "
            "исследования допустимо, если указано, что оно выполнено в течение года."
        ),
        "flag_code": "ПРОФ_ВЗРОСЛЫЙ_НЕПОЛНЫЙ_ОБЪЁМ",
        "severity": "значительный",
    },
    {
        "rule_id": "adult_prophylactic_health_group",
        "source": "404n",
        "source_ref": "приказ МЗ РФ от 27.04.2021 № 404н, п. 14 пп. 2, п. 17, п. 23",
        "verified_at": "2026-08-19",
        "rule_type": "required_field",
        "applies_to": dict(_APPLIES),
        "targets": ["recommendations", "follow_up"],
        "expectation": (
            "По итогам осмотра установлена группа здоровья (I, II, IIIа или IIIб); при "
            "II группе с высоким или очень высоким сердечно-сосудистым риском и при "
            "III группах указано диспансерное наблюдение и кем оно проводится."
        ),
        "flag_code": "ПРОФ_ВЗРОСЛЫЙ_НЕТ_ГРУППЫ_ЗДОРОВЬЯ",
        "severity": "значительный",
    },
    {
        "rule_id": "adult_prophylactic_counselling",
        "source": "404n",
        "source_ref": "приказ МЗ РФ от 27.04.2021 № 404н, п. 14 пп. 3, п. 16 пп. 1, п. 17, п. 18 пп. 14",
        "verified_at": "2026-08-19",
        "rule_type": "conditional_required",
        "applies_to": dict(_APPLIES),
        "targets": ["recommendations"],
        "condition": (
            "Если выявлены факторы риска хронических неинфекционных заболеваний, "
            "высокий или очень высокий сердечно-сосудистый риск, ишемическая болезнь "
            "сердца, цереброваскулярное заболевание или артериальная гипертензия."
        ),
        "expectation": (
            "В записи отражено краткое профилактическое консультирование, включая "
            "разъяснение мер по снижению факторов риска, а при высоком риске — "
            "симптомов инфаркта миокарда и инсульта и необходимости своевременного "
            "вызова скорой медицинской помощи."
        ),
        "flag_code": "ПРОФ_ВЗРОСЛЫЙ_НЕТ_КОНСУЛЬТИРОВАНИЯ",
        "severity": "незначительный",
    },
    {
        "rule_id": "adult_prophylactic_marker",
        "source": "404n",
        "source_ref": "приказ МЗ РФ от 27.04.2021 № 404н, п. 22",
        "verified_at": "2026-08-19",
        "rule_type": "required_field",
        "applies_to": dict(_APPLIES),
        "targets": ["visit_meta"],
        "expectation": (
            "Запись помечена как «Профилактический медицинский осмотр» или "
            "«Диспансеризация» — в наименовании услуги или в тексте записи."
        ),
        "flag_code": "ПРОФ_ВЗРОСЛЫЙ_НЕТ_ПОМЕТКИ",
        "severity": "незначительный",
    },
]

have = {r["rule_id"] for r in doc["rules"]}
for r in new:
    assert r["rule_id"] not in have, f"{r['rule_id']} уже есть"
doc["rules"].extend(new)

p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("rules:", len(doc["rules"]))
PY
```

Expected вывод: `rules: 37`

- [ ] **Step 2: Обновить снимок**

В `tests/test_formal_rules_schema.py`: `EXPECTED_RULE_COUNT = 37`

- [ ] **Step 3: Написать тест разделения взрослых и детских профправил**

Дописать в `tests/test_formal_get_rules.py` (тест против **настоящего**
`rules.json`, без monkeypatch):

```python
def test_adult_prophylactic_rules_do_not_leak_to_children():
    """404н-правила — только взрослым, 211н-исключения — только детям."""
    v = FormalValidator()

    adult = _flags(v.get_rules({VisitType.PROPHYLACTIC}, 45, ["Z00.0"]))
    child = _flags(v.get_rules({VisitType.PROPHYLACTIC}, 10, ["Z00.1"]))

    assert "ПРОФ_ВЗРОСЛЫЙ_НЕПОЛНЫЙ_ОБЪЁМ" in adult
    assert "ПРОФ_ВЗРОСЛЫЙ_НЕТ_ГРУППЫ_ЗДОРОВЬЯ" in adult
    assert "ПРОФ_ВЗРОСЛЫЙ_НЕПОЛНЫЙ_ОБЪЁМ" not in child

    assert "ДОПУСТИМО_БЕЗ_ЖАЛОБ_ПРОФИЛАКТИКА" in child
    assert "ДОПУСТИМО_БЕЗ_ЖАЛОБ_ПРОФИЛАКТИКА" not in adult
```

- [ ] **Step 4: Прогнать тесты**

Run: `pytest tests/test_formal_get_rules.py tests/test_formal_rules_schema.py -v`
Expected: новый тест PASS, `test_rule_count_snapshot` PASS.

- [ ] **Step 5: Коммит**

```bash
git add src/audit/formal_structure/rules.json tests/test_formal_rules_schema.py tests/test_formal_get_rules.py
git commit -m "feat(formal): 4 правила 404н — объём ПМО взрослых, группа здоровья, консультирование, пометка"
```

---

### Task 8: Диспансерное наблюдение (168н/192н) + перенос `treatment_dosage_clarity`

Спека §4.3 и §4.1. Два правила с общим `flag_code`
`ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО` (взрослое — с `icd_prefixes` по
Приложению А спеки, детское — без). Здесь же `treatment_dosage_clarity`
уезжает с 203н на 1094н с новым `expectation` и `age_group: all` — последнее
правило, снимающее ярлык `203n`.

**Files:**
- Modify: `src/audit/formal_structure/rules.json`
- Modify: `tests/test_formal_rules_schema.py` (`EXPECTED_RULE_COUNT`)
- Modify: `tests/test_formal_get_rules.py` (тесты `icd_prefixes` на реальных данных)

**Interfaces:**
- Consumes: фильтр `icd_prefixes` из Задачи 3.
- Produces: правила `dispensary_followup_adult`, `dispensary_followup_child`;
  `treatment_dosage_clarity` с `source: 1094n` и `age_group: all`.

- [ ] **Step 1: Добавить правила и перенести `treatment_dosage_clarity`**

Префиксы — дословно из Приложения А спеки, диапазоны раскрыты:

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("src/audit/formal_structure/rules.json")
doc = json.loads(p.read_text(encoding="utf-8"))
by_id = {r["rule_id"]: r for r in doc["rules"]}

# ── Приложение А спеки: префиксы МКБ-10 для ДН взрослых (168н, прил. 1–3) ──
PREFIXES = [
    # прил. 1 — терапевт
    "I10", "I11", "I12", "I13", "I14", "I15",
    "I20", "I21", "I22", "I23", "I24", "I25",
    "Z95.0", "Z95.1", "Z95.5",
    "I44", "I45", "I46", "I47", "I48", "I49", "I50",
    "I65.2", "E78", "R73.0", "R73.9", "E11",
    "I69.0", "I69.1", "I69.2", "I69.3", "I69.4", "I67.8",
    "K20", "K21.0", "K25", "K26", "K31.7", "K86",
    "J41.0", "J41.1", "J41.8", "J44.0", "J44.8", "J44.9", "J47.0",
    "J45.0", "J45.1", "J45.8", "J45.9", "J12", "J13", "J14", "J84.1",
    "N18.1", "N18.9", "M81.5", "K29.4", "K29.5",
    "D12.6", "D12.8", "K62.1", "K50", "K51",
    "K22.0", "K22.2", "K22.7", "K70.3",
    "K74.3", "K74.4", "K74.5", "K74.6", "D13.4", "D37.6",
    # прил. 2 — кардиолог
    "I05", "I06", "I07", "I08", "I09",
    "I34", "I35", "I36", "I37",
    "I51.0", "I51.1", "I51.2", "I71",
    "Z95.2", "Z95.3", "Z95.4", "Z95.8", "Z95.9",
    "I26", "I27.0", "I27.2", "I27.8", "I28",
    "I33", "I38", "I39", "I40", "I41", "I51.4", "I42",
    "Q20", "Q21", "Q22", "Q23", "Q24", "Q25", "Q26", "Q27", "Q28",
    # прил. 3 — предраковые состояния
    "B18.0", "B18.1", "B18.2", "B20", "B21", "B22", "B23", "B24",
    "E34.8", "D13.7", "D35.0", "D35.1", "D35.2", "D35.8", "D44.8",
    "E34.5", "E22.0", "E04.1", "E04.2", "E05.1", "E05.2", "E21.0",
    "Q85.1", "D11", "Q78.1", "D30.0", "D30.3", "D30.4", "D41.0",
    "D29.1", "M96", "M88", "D16", "M85", "Q78.4", "D31", "D23.1",
    "J38.1", "D14.0", "D14.1", "D14.2", "J33", "D14",
    "D10.0", "D10.1", "D10.2", "D10.3", "D10.4", "D10.5", "D10.6", "D10.7", "D10.9",
    "J37", "J31", "K13.0", "K13.2", "K13.7", "L43", "D22", "Q82.5",
    "D23", "L57.1", "L82", "Q82.1", "E28.2", "D39.1", "D24",
]
assert len(PREFIXES) == len(set(PREFIXES)), "дубли в списке префиксов"
assert all(x == x.upper().strip() for x in PREFIXES)

new = [
    {
        "rule_id": "dispensary_followup_adult",
        "source": "168n",
        "source_ref": (
            "приказ МЗ РФ от 15.03.2022 № 168н, п. 4, п. 13, п. 14, прил. 1–3; "
            "приказ МЗ РФ от 13.05.2025 № 274н, прил. 1 (форма 025/у), раздел "
            "«Диспансерное наблюдение» — «Рекомендации и дата следующего "
            "диспансерного осмотра, консультации»"
        ),
        "verified_at": "2026-08-19",
        "rule_type": "followup_check",
        "applies_to": {
            "visit_types": ["primary", "repeat"],
            "specialties": ["all"],
            "age_group": "adult",
            "icd_prefixes": PREFIXES,
        },
        "targets": ["follow_up", "diagnosis", "recommendations", "treatment_plan"],
        "condition": (
            "Если у пациента диагноз из перечня заболеваний, подлежащих диспансерному "
            "наблюдению."
        ),
        "expectation": (
            "При впервые установленном диагнозе в записи есть решение о постановке на "
            "диспансерное наблюдение или направление к специалисту для него; при "
            "повторном приёме по такому диагнозу отражены оценка эффективности лечения "
            "и контролируемых показателей, коррекция терапии при необходимости и "
            "рекомендации с датой или сроком следующего осмотра."
        ),
        "flag_code": "ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО",
        "severity": "значительный",
    },
    {
        "rule_id": "dispensary_followup_child",
        "source": "192n",
        "source_ref": "приказ МЗ РФ от 11.04.2025 № 192н, п. 3, п. 9, п. 12, п. 14",
        "verified_at": "2026-08-19",
        "rule_type": "followup_check",
        "applies_to": {
            "visit_types": ["primary", "repeat"],
            "specialties": ["all"],
            "age_group": "child",
        },
        "targets": ["follow_up", "diagnosis", "recommendations"],
        "condition": (
            "Если у несовершеннолетнего хроническое заболевание или состояние, "
            "требующее диспансерного наблюдения по клиническим рекомендациям."
        ),
        "expectation": (
            "При впервые установленном диагнозе в записи есть решение о постановке на "
            "диспансерное наблюдение или направление к специалисту для него; при "
            "повторном приёме отражены оценка состояния и эффективности назначенного "
            "лечения, коррекция терапии при необходимости и рекомендации с датой или "
            "сроком следующего осмотра."
        ),
        "flag_code": "ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО",
        "severity": "незначительный",
    },
]

have = {r["rule_id"] for r in doc["rules"]}
for r in new:
    assert r["rule_id"] not in have, f"{r['rule_id']} уже есть"
doc["rules"].extend(new)

# ── treatment_dosage_clarity: 203н → 1094н (спека §4.1) ──
tdc = by_id["treatment_dosage_clarity"]
assert tdc["source"] == "203n"
tdc["source"] = "1094n"
tdc["source_ref"] = "приказ МЗ РФ от 24.11.2021 № 1094н, прил. 1, п. 2, п. 17"
tdc["verified_at"] = "2026-08-19"
tdc["applies_to"]["age_group"] = "all"
tdc["expectation"] = (
    "Каждое назначение лекарственного препарата содержит наименование, дозировку, "
    "способ введения и применения, режим дозирования (кратность), продолжительность "
    "лечения и обоснование назначения."
)

p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("rules:", len(doc["rules"]), "| префиксов:", len(PREFIXES))
PY
```

Expected вывод: `rules: 39 | префиксов: 180`

- [ ] **Step 2: Обновить снимок**

В `tests/test_formal_rules_schema.py`: `EXPECTED_RULE_COUNT = 39`

- [ ] **Step 3: Проверить, что ярлык 203n исчез**

Run: `pytest tests/test_formal_rules_schema.py::test_no_rule_references_retired_203n -v`
Expected: PASS (впервые с Задачи 4).

- [ ] **Step 4: Написать тесты `icd_prefixes` на настоящем наборе правил**

Дописать в `tests/test_formal_get_rules.py`:

```python
def test_dispensary_adult_requires_matching_icd():
    """Взрослое ДН-правило включается только на кодах из перечня 168н."""
    v = FormalValidator()

    assert "ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО" in _flags(
        v.get_rules({VisitType.PRIMARY}, 55, ["I11.9"])
    )
    assert "ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО" not in _flags(
        v.get_rules({VisitType.PRIMARY}, 55, ["J06.9"])
    )
    assert "ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО" not in _flags(
        v.get_rules({VisitType.PRIMARY}, 55, [])
    )


def test_dispensary_child_rule_needs_no_icd():
    """Детское ДН-правило — без перечня кодов, но только детям."""
    v = FormalValidator()

    child = v.get_rules({VisitType.PRIMARY}, 10, ["J06.9"])
    assert "ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО" in _flags(child)
    assert [r["rule_id"] for r in child if r["flag_code"] == "ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО"] == [
        "dispensary_followup_child"
    ]

    adult = v.get_rules({VisitType.PRIMARY}, 55, ["I11.9"])
    assert [r["rule_id"] for r in adult if r["flag_code"] == "ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО"] == [
        "dispensary_followup_adult"
    ]


def test_format_rules_renders_new_rules():
    """_format_rules не падает на правилах с condition и без него."""
    v = FormalValidator()
    rules = v.get_rules({VisitType.PRIMARY}, 55, ["I11.9"])
    text = v._format_rules(rules)

    assert "(ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО)" in text
    assert len(text.splitlines()) == len(rules)


def test_flag_source_lookup_covers_new_flags():
    """_FLAG_SOURCE строится по всем правилам, включая новые."""
    import audit.formal_structure.validator as v

    assert v._FLAG_SOURCE["ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО"] in {"168n", "192n"}
    assert v._FLAG_SOURCE["ПРОФ_ВЗРОСЛЫЙ_НЕПОЛНЫЙ_ОБЪЁМ"] == "404n"
    assert v._FLAG_SOURCE["НАЗНАЧЕНИЕ_ПО_ТОРГОВОМУ_БЕЗ_МНН"] == "1094n"
```

Примечание про `_FLAG_SOURCE`: у двух ДН-правил один `flag_code`, поэтому в
словаре останется ярлык последнего из них — отсюда проверка на множество.

- [ ] **Step 5: Прогнать тесты**

Run: `pytest tests/test_formal_get_rules.py tests/test_formal_rules_schema.py -v`
Expected: PASS, кроме `test_every_rule_has_source_ref_and_verified_at`
(остальные правила получают поля в Задаче 10).

- [ ] **Step 6: Коммит**

```bash
git add src/audit/formal_structure/rules.json tests/test_formal_rules_schema.py tests/test_formal_get_rules.py
git commit -m "feat(formal): диспансерное наблюдение 168н/192н с фильтром по МКБ; назначения — на 1094н"
```

---

### Task 9: ИДС вне перечня 390н и листки нетрудоспособности

Спека §4.4 и §4.5. Три новых правила.

**Files:**
- Modify: `src/audit/formal_structure/rules.json`
- Modify: `tests/test_formal_rules_schema.py` (`EXPECTED_RULE_COUNT`)

**Interfaces:**
- Consumes: формат файла (Задача 1).
- Produces: правила `consent_for_intervention_outside_list`,
  `sick_leave_needs_justification`, `sick_leave_over_15_days_needs_vk`.

- [ ] **Step 1: Добавить правила**

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("src/audit/formal_structure/rules.json")
doc = json.loads(p.read_text(encoding="utf-8"))

new = [
    {
        "rule_id": "consent_for_intervention_outside_list",
        "source": "323-FZ",
        "source_ref": (
            "Федеральный закон от 21.11.2011 № 323-ФЗ, ст. 20 ч. 1, ч. 6, ч. 7; "
            "приказ МЗ РФ от 12.11.2021 № 1051н, п. 1, п. 3, п. 6, п. 9; приказ "
            "Минздравсоцразвития РФ от 23.04.2012 № 390н (перечень); приказ МЗ РФ от "
            "13.05.2025 № 274н, прил. 1 (форма 025/у), строка «Информированное "
            "добровольное согласие на медицинское вмешательство, отказ от медицинского "
            "вмешательства»"
        ),
        "verified_at": "2026-08-19",
        "rule_type": "conditional_required",
        "applies_to": {
            "visit_types": ["lab_research_intervention", "primary", "repeat"],
            "specialties": ["all"],
            "age_group": "all",
        },
        "targets": ["visit_meta", "treatment_plan"],
        "condition": (
            "Если в записи выполнено вмешательство, не входящее в перечень 390н: "
            "вакцинация, инвазивная манипуляция, малое хирургическое вмешательство, "
            "эндоскопия, биопсия, введение препарата иным путём, чем внутримышечно, "
            "внутривенно, подкожно или внутрикожно."
        ),
        "expectation": (
            "В записи отражено информированное добровольное согласие пациента или его "
            "законного представителя на это вмешательство либо отказ от него."
        ),
        "flag_code": "ИДС_НЕ_ОТРАЖЕНО_ПРИ_ВМЕШАТЕЛЬСТВЕ",
        "severity": "незначительный",
    },
    {
        "rule_id": "sick_leave_needs_justification",
        "source": "1089n",
        "source_ref": (
            "приказ МЗ РФ от 23.11.2021 № 1089н, п. 9, п. 11; приказ МЗ РФ от "
            "11.04.2025 № 195н, п. 6 пп. 4; приказ МЗ РФ от 13.05.2025 № 274н, прил. 1 "
            "(форма 025/у), строка «Листок нетрудоспособности, справка»"
        ),
        "verified_at": "2026-08-19",
        "rule_type": "conditional_required",
        "applies_to": {
            "visit_types": ["primary", "repeat"],
            "specialties": ["all"],
            "age_group": "all",
        },
        "targets": ["visit_meta", "state", "recommendations"],
        "condition": (
            "Если в записи упомянут листок нетрудоспособности (ЛН, ЭЛН), справка о "
            "временной нетрудоспособности, или указано, что пациент либо ухаживающий "
            "за ребёнком нетрудоспособен."
        ),
        "expectation": (
            "В записи есть обоснование временной нетрудоспособности состоянием "
            "пациента и отметка о сформированном или продлённом листке "
            "нетрудоспособности (номер и/или даты) либо о его закрытии с обоснованием."
        ),
        "flag_code": "ЛН_БЕЗ_ОБОСНОВАНИЯ",
        "severity": "значительный",
    },
    {
        "rule_id": "sick_leave_over_15_days_needs_vk",
        "source": "1089n",
        "source_ref": (
            "приказ МЗ РФ от 23.11.2021 № 1089н, п. 20, п. 21, п. 22, п. 44, п. 46; "
            "приказ МЗ РФ от 11.04.2025 № 195н, п. 7"
        ),
        "verified_at": "2026-08-19",
        "rule_type": "conditional_required",
        "applies_to": {
            "visit_types": ["primary", "repeat"],
            "specialties": ["all"],
            "age_group": "all",
        },
        "targets": ["visit_meta", "recommendations"],
        "condition": (
            "Если из записи следует, что суммарный срок временной нетрудоспособности "
            "по данному случаю превышает 15 календарных дней, либо листок "
            "нетрудоспособности по уходу за ребёнком до 15 лет продлён с 16-го дня."
        ),
        "expectation": "В записи указано решение врачебной комиссии о продлении.",
        "flag_code": "ЛН_СВЫШЕ_15_ДНЕЙ_БЕЗ_ВК",
        "severity": "значительный",
    },
]

have = {r["rule_id"] for r in doc["rules"]}
for r in new:
    assert r["rule_id"] not in have, f"{r['rule_id']} уже есть"
doc["rules"].extend(new)

p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("rules:", len(doc["rules"]))
PY
```

Expected вывод: `rules: 42`

- [ ] **Step 2: Обновить снимок**

В `tests/test_formal_rules_schema.py`: `EXPECTED_RULE_COUNT = 42`

- [ ] **Step 3: Прогнать схема-тесты**

Run: `pytest tests/test_formal_rules_schema.py -v`
Expected: PASS, кроме `test_every_rule_has_source_ref_and_verified_at`.

- [ ] **Step 4: Коммит**

```bash
git add src/audit/formal_structure/rules.json tests/test_formal_rules_schema.py
git commit -m "feat(formal): ИДС вне перечня 390н и правила по листкам нетрудоспособности (1089н/195н)"
```

---

### Task 10: Разъярлычивание 464н и `source_ref`/`verified_at` остальным правилам

Спека §4.8. Ярлык `464n` сейчас висит на шести правилах, из которых пять — не
лабораторные: УЗИ уезжает на `557n`, рентген на `560n`, функциональные на
`205n`, манипуляции и инъекции — на `274n`. Здесь же все оставшиеся правила
получают `source_ref`/`verified_at`, и оба красных теста Задачи 4 становятся
зелёными.

**Files:**
- Modify: `src/audit/formal_structure/rules.json`
- Test: `tests/test_formal_rules_schema.py` (уже написан)

**Interfaces:**
- Consumes: схема-тесты (Задача 4).
- Produces: `rules.json`, в котором у **всех** правил есть `source_ref` и
  `verified_at`; новые ярлыки `557n`, `560n`, `205n` в `_FLAG_SOURCE`.

- [ ] **Step 1: Применить правки**

`ecg_functional_description_and_conclusion` — единственное правило этой задачи,
у которого меняется `expectation` (спека §4.8: дополнить расчётными
показателями).

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("src/audit/formal_structure/rules.json")
doc = json.loads(p.read_text(encoding="utf-8"))
by_id = {r["rule_id"]: r for r in doc["rules"]}

F025 = "приказ МЗ РФ от 13.05.2025 № 274н, прил. 1 (форма 025/у)"

patch = {
    # ── 464н: остаётся только лаборатория ──
    "lab_biomaterial_and_result": {
        "source": "464n",
        "source_ref": "приказ МЗ РФ от 18.05.2021 № 464н, прил. 1 к Правилам, п. 16",
    },
    "ultrasound_protocol_and_conclusion": {
        "source": "557n",
        "source_ref": "приказ МЗ РФ от 08.06.2020 № 557н, п. 19, п. 20, п. 24",
    },
    "xray_protocol_and_conclusion": {
        "source": "560n",
        "source_ref": (
            "приказ МЗ РФ от 09.06.2020 № 560н, п. 16, п. 17, п. 21 (действует до "
            "01.09.2026; с 01.09.2026 — приказ МЗ РФ от 08.05.2026 № 359н, прил. 1, "
            "п. 20, п. 21, п. 25)"
        ),
    },
    "ecg_functional_description_and_conclusion": {
        "source": "205n",
        "source_ref": "приказ МЗ РФ от 14.04.2025 № 205н, п. 18, п. 19, п. 24",
        "expectation": (
            "Протокол функционального исследования содержит описание параметров и "
            "результатов, расчётные показатели функциональных нарушений (при наличии) "
            "и заключение по результатам исследования."
        ),
    },
    "manipulation_technique_and_outcome": {
        "source": "274n",
        "source_ref": (
            f"{F025}, запись о выполненной процедуре (манипуляции) в записи "
            "врача-специалиста и в разделе «Медицинское наблюдение в динамике»; "
            "отдельного приказа о содержании такой записи нет"
        ),
    },
    "injection_procedure_completeness": {
        "source": "274n",
        "source_ref": (
            f"{F025}, запись о выполненной процедуре (манипуляции); приказ МЗ РФ от "
            "24.11.2021 № 1094н, прил. 1, п. 2 (наименование, доза, способ введения)"
        ),
    },
    # ── 274н: старые правила ──
    "visit_meta_required": {
        "source_ref": (
            f"{F025}, стр. 1–2 (дата рождения, пол; дата осмотра, врач — должность, "
            "специальность); прил. 4, п. 9.11, п. 9.13, п. 9.16"
        ),
    },
    "primary_core_sections_required": {
        "source_ref": f"{F025}, раздел «Записи врачей-специалистов»; прил. 2, п. 13",
    },
    "repeat_core_sections_required": {
        "source_ref": f"{F025}, раздел «Медицинское наблюдение в динамике»; прил. 2, п. 14",
    },
    "diagnosis_required": {
        "source_ref": (
            f"{F025}, строка «Диагноз основного заболевания / код по МКБ»; прил. 4, п. 9.20"
        ),
    },
    "plan_vs_result_separation": {
        "source_ref": (
            f"{F025}: разделение строк «Назначения (исследования, консультации)» и "
            "разделов «Результаты функциональных методов исследования» / «Результаты "
            "лабораторных методов исследования»; прил. 2, п. 22, п. 23"
        ),
    },
    "service_specialty_visit_alignment": {
        "source_ref": (
            f"{F025}, строка «Врач (должность, специальность)»; прил. 4, п. 9.11, п. 9.16"
        ),
    },
    "placeholder_values_are_defect": {
        "source_ref": "внутренний стандарт качества (привязка к 274н формальная)",
    },
    "duplicate_semantic_blocks_are_defect": {
        "source_ref": "внутренний стандарт качества (привязка к 274н формальная)",
    },
    # ── 211н: детские профилактические исключения (косвенные) ──
    "prophylactic_absence_of_complaints_allowed": {
        "source_ref": (
            "приказ МЗ РФ от 14.04.2025 № 211н, п. 1, п. 17, п. 19, п. 20 — предмет "
            "профосмотра составляют осмотры специалистов и исследования по Перечню и "
            "группа здоровья; жалобы, динамика и лечение его содержанием не являются"
        ),
    },
    "prophylactic_absence_of_dynamics_allowed": {
        "source_ref": (
            "приказ МЗ РФ от 14.04.2025 № 211н, п. 1, п. 17, п. 19, п. 20 — предмет "
            "профосмотра составляют осмотры специалистов и исследования по Перечню и "
            "группа здоровья; жалобы, динамика и лечение его содержанием не являются"
        ),
    },
    "prophylactic_no_treatment_allowed_if_decision_present": {
        "source_ref": (
            "приказ МЗ РФ от 14.04.2025 № 211н, п. 1, п. 17, п. 19, п. 20 — предмет "
            "профосмотра составляют осмотры специалистов и исследования по Перечню и "
            "группа здоровья; жалобы, динамика и лечение его содержанием не являются"
        ),
    },
    # ── 190н: туберкулинодиагностика ──
    "tuberculin_objective_data_required": {
        "source_ref": "приказ МЗ РФ от 11.04.2025 № 190н, п. 18 пп. «а»",
    },
    "tuberculin_specialist_examination_if_pathology": {
        "source_ref": "приказ МЗ РФ от 11.04.2025 № 190н, п. 18 пп. «б», п. 19",
    },
    "tuberculin_conclusion_required": {
        "source_ref": "приказ МЗ РФ от 11.04.2025 № 190н, п. 18 пп. «в»",
    },
    # ── прочие ──
    "icd_text_alignment": {
        "source_ref": "МКБ-10 — справочник, не нормативный правовой акт",
    },
    "too_general_icd_for_rich_detail": {
        "source_ref": "МКБ-10 — справочник, не нормативный правовой акт",
    },
    "has_typos": {
        "source_ref": "внутренний стандарт качества",
    },
    "legal_representative_info": {
        "source_ref": (
            "Федеральный закон от 21.11.2011 № 323-ФЗ, ст. 20 ч. 1 (согласие законного "
            "представителя); ст. 54"
        ),
    },
}

for rid, fields in patch.items():
    rule = by_id[rid]
    rule.update(fields)
    rule["verified_at"] = "2026-08-19"

missing = [r["rule_id"] for r in doc["rules"] if not r.get("source_ref") or not r.get("verified_at")]
assert not missing, f"без source_ref/verified_at остались: {missing}"

p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("правил всего:", len(doc["rules"]))
print("ярлыки:", sorted({r.get("source", "") for r in doc["rules"]}))
PY
```

Expected вывод: `правил всего: 42`, ярлыки —
`['', '1089n', '1094n', '168n', '190n', '192n', '205n', '211n', '274n', '323-FZ', '404n', '464n', '557n', '560n', 'icd10']`
(пустой — у `has_typos`, у него `source` не задан по исходному файлу).

- [ ] **Step 2: Прогнать все схема-тесты — должны быть полностью зелёными**

Run: `pytest tests/test_formal_rules_schema.py -v`
Expected: PASS всех тестов, включая
`test_every_rule_has_source_ref_and_verified_at` и
`test_no_rule_references_retired_203n`.

- [ ] **Step 3: Прогнать все тесты формальной проверки**

Run: `pytest tests/test_formal_rules_schema.py tests/test_formal_get_rules.py tests/test_formal_validator_meta.py -v`
Expected: PASS. В логе загрузки — `oldest verified_at=2026-08-19`.

- [ ] **Step 4: Коммит**

```bash
git add src/audit/formal_structure/rules.json
git commit -m "feat(formal): 464н разъярлычен на 557н/560н/205н/274н; source_ref и verified_at у всех правил"
```

---

### Task 11: Фикс мёртвого определения туберкулинового визита

Не из спеки — дефект, найденный при разборе кода и подтверждённый
пользователем. `get_visit_types` читает `visit["Диагноз"]["Код"]`
(единственное число), тогда как все остальные читатели карты (`json_parser.py`,
`pipeline.py`, `icd_check/validator.py`, `filters.py`, `api/routes/visits.py`)
и документированный контракт `docs/clinic-data-requirements.md` используют
`visit["Диагнозы"][].КодМКБ`. Ключа `Диагноз` в данных нет — значит,
`PROPHYLACTIC_TUBERCULIN` недостижим и три правила `190n` никогда не
подключаются.

**Внимание при приёмке:** это единственная задача плана, меняющая поведение
аудита на существующих картах — после неё на визитах с Z11.1 начнут срабатывать
три критичных туберкулиновых правила. Отдельный коммит именно для того, чтобы
эффект можно было оценить (и при желании откатить) независимо от ревизии
нормативки.

**Files:**
- Modify: `src/audit/formal_structure/validator.py:134` (метод `get_visit_types`)
- Test: `tests/test_formal_visit_types.py` (создать)

**Interfaces:**
- Consumes: `VisitType`, `FormalValidator` (существующие).
- Produces: поведение — `get_visit_types` возвращает
  `PROPHYLACTIC_TUBERCULIN`, если любой элемент `visit["Диагнозы"]` имеет
  `КодМКБ` `Z11.1` (без учёта регистра и пробелов).

- [ ] **Step 1: Написать падающий тест**

Создать `tests/test_formal_visit_types.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.formal_structure.validator import FormalValidator, VisitType


def _visit(diagnoses=None, services=None):
    return {
        "Прием": {"GUID": "test-guid"},
        "Диагнозы": diagnoses or [],
        "Услуги": services or [{"Наименование": "Приём первичный"}],
    }


async def test_z11_1_in_diagnoses_gives_tuberculin_type():
    """Z11.1 читается из Диагнозы[].КодМКБ — контракт clinic-data-requirements.md."""
    got = await FormalValidator().get_visit_types(_visit(diagnoses=[{"КодМКБ": "Z11.1"}]))
    assert VisitType.PROPHYLACTIC_TUBERCULIN in got


async def test_z11_1_is_case_and_space_insensitive():
    got = await FormalValidator().get_visit_types(_visit(diagnoses=[{"КодМКБ": " z11.1 "}]))
    assert VisitType.PROPHYLACTIC_TUBERCULIN in got


async def test_z11_1_found_among_several_diagnoses():
    visit = _visit(diagnoses=[{"КодМКБ": "J06.9"}, {"КодМКБ": "Z11.1"}])
    got = await FormalValidator().get_visit_types(visit)
    assert VisitType.PROPHYLACTIC_TUBERCULIN in got


async def test_other_diagnosis_does_not_give_tuberculin_type():
    got = await FormalValidator().get_visit_types(_visit(diagnoses=[{"КодМКБ": "J06.9"}]))
    assert VisitType.PROPHYLACTIC_TUBERCULIN not in got


async def test_no_diagnoses_does_not_crash():
    got = await FormalValidator().get_visit_types(_visit())
    assert got  # тип определился по услуге, исключения нет
```

- [ ] **Step 2: Прогнать тесты и убедиться, что они падают**

Run: `pytest tests/test_formal_visit_types.py -v`
Expected: FAIL — три первых теста (`PROPHYLACTIC_TUBERCULIN` не попадает в
результат, потому что ключа `Диагноз` в карте нет).

- [ ] **Step 3: Починить чтение**

В `get_visit_types` заменить блок

```python
        diag_code: str = (visit.get("Диагноз") or {}).get("Код", "") or ""
        if diag_code.strip().upper() == "Z11.1":
            result.add(VisitType.PROPHYLACTIC_TUBERCULIN)
```

на

```python
        diagnoses: list = visit.get("Диагнозы") or []
        if any(
            str(d.get("КодМКБ") or "").strip().upper() == "Z11.1"
            for d in diagnoses
            if isinstance(d, dict)
        ):
            result.add(VisitType.PROPHYLACTIC_TUBERCULIN)
```

В докстринге метода поправить пункт 1: `Z11.1 ICD code` →
`Z11.1 among visit["Диагнозы"][].КодМКБ`.

- [ ] **Step 4: Прогнать тесты — должны пройти**

Run: `pytest tests/test_formal_visit_types.py -v`
Expected: PASS (5 тестов)

- [ ] **Step 5: Убедиться, что ключа `Диагноз` в коде больше нет**

Run: `grep -rn '"Диагноз"' src/`
Expected: пустой вывод.

- [ ] **Step 6: Убедиться, что разбор кодов услуг не задет**

Важно: `Код` есть и у услуг (`Услуги[].Код`, см. `tests/test_validations.py:76`),
и он разбирается в том же методе — циклом `for raw in svc.values()` по значениям
**услуги**. Правка касается только верхнеуровневого ключа визита `Диагноз`,
до услуг не достаёт, но проверяем это прогоном, а не рассуждением:

Run: `pytest tests/test_formal_visit_types.py tests/test_filters.py tests/test_validations.py -v`
Expected: PASS. Если `test_validations.py` требует LLM и не идёт локально —
прогнать первые два и отметить это при сдаче задачи.

Дополнительно — услуга с NMU-кодом в поле `Код` по-прежнему даёт свой тип:

```bash
python3 -c "
import asyncio, sys; sys.path.insert(0,'src')
from audit.formal_structure.validator import FormalValidator
v = {'Услуги': [{'Наименование': 'Прием врача-терапевта первичный', 'Код': 'B01.070.001'}], 'Диагнозы': [{'КодМКБ': 'Z11.1'}]}
print(sorted(t.name for t in asyncio.run(FormalValidator().get_visit_types(v))))"
```
Expected: `['PRIMARY', 'PROPHYLACTIC_TUBERCULIN']` — тип по коду услуги
сохранился, туберкулиновый добавился.

Проверено на состоянии до правки: та же карта без строки `Диагнозы` даёт
`['PRIMARY']`, то есть разбор `Услуги[].Код` работает и правкой не затрагивается.

**Осторожно с выбором кода в ручных проверках.** PRIMARY/REPEAT дают только
коды `B01.070.*` (терапевт); `B01.058.001` и подобные по действующей ветке
`if middle != "070"` намеренно попадают в `OTHER`. Это не дефект и в этом плане
не меняется.

- [ ] **Step 7: Коммит**

```bash
git add src/audit/formal_structure/validator.py tests/test_formal_visit_types.py
git commit -m "fix(formal): Z11.1 читается из Диагнозы[].КодМКБ — туберкулиновый тип визита был недостижим"
```

---

### Task 12: Фикс `NMU_RE` — A-коды услуг не распознаются

Не из спеки — дефект, найденный при разборе кода и подтверждённый на реальных
выгрузках кодов услуг (`~/projects/mdsgrep`, `~/projects/alenkagrep`).

`NMU_RE` требует **три** цифры в среднем сегменте кода
(`^[ABАВ]\d{2}\.\d{3}\.\d{3}(?:\.\d{3})?$`), а у A-кодов номенклатуры их
**две**: `A04.16.001`, `A09.05.023`, `A11.02.002`. Регулярка их не матчит, до
ветки `if prefix.startswith("A")` управление не доходит, `LAB_RESEARCH_INTERVENTION`
недостижим — и шесть правил (`lab_biomaterial_and_result`,
`ultrasound_protocol_and_conclusion`, `xray_protocol_and_conclusion`,
`ecg_functional_description_and_conclusion`, `manipulation_technique_and_outcome`,
`injection_procedure_completeness`) не срабатывают никогда. Именно тем шести,
которым Задача 10 переставляет ярлыки.

**Замер на реальных данных (по 999 строк на клинику, срез):**

| | МДС (`Артикул`) | Алёнка (`КодЕГИСЗ`) |
|---|---|---|
| Номенклатурных кодов | 420 строк / 32 уник | 911 строк / 10 уник |
| Матчит `NMU_RE` сейчас | 262 (только B-коды) | 911 |
| Матчилось бы после фикса | 420 | 911 |
| **Теряется сейчас** | **158 строк / 20 A-кодов (38%)** | 0 |

У Алёнки A-кодов в выборке нет вовсе, единственное нематчащееся значение —
`-` (88 строк, пустышка, и после фикса оно тоже не матчится — так и надо).
Эффект фикса сосредоточен на МДС.

**Внимание при приёмке:** вместе с Задачей 11 это вторая задача плана,
меняющая поведение аудита на существующих картах, и более крупная из двух.
Отдельный коммит — чтобы на стендовом гейте (Задача 15) её эффект можно было
отделить от ревизии нормативки и при необходимости откатить отдельно.

**Files:**
- Modify: `src/audit/formal_structure/validator.py:29` (`NMU_RE`)
- Test: `tests/test_formal_visit_types.py` (дописать к файлу из Задачи 11)

**Interfaces:**
- Consumes: `VisitType`, `FormalValidator`, `NMU_RE` (существующие).
- Produces: поведение — услуга с A-кодом даёт `LAB_RESEARCH_INTERVENTION`;
  B-коды классифицируются как прежде.

- [ ] **Step 1: Написать падающие тесты на реальных кодах**

Коды взяты из выгрузки МДС — все они встречаются в боевых данных.
Дописать в `tests/test_formal_visit_types.py`:

```python
import pytest

from audit.formal_structure.validator import NMU_RE


# Реальные коды из выгрузки МДС (~/projects/mdsgrep).
_REAL_A_CODES = [
    "A01.01.002",
    "A04.01.001",
    "A04.10.002",
    "A04.16.001",
    "A04.12.018",
    "A04.22.001",
    "A04.28.002",
    "A11.01.009",
    "A16.01.017",
    "A04.12.005.005",      # четыре сегмента
    "A04.20.001.001",
    "A11.22.002.001",
]

_REAL_B_CODES = [
    "B01.023.001",
    "B01.004.001",
    "B01.058.001",
    "B01.070.001",
    "B04.031.002",
    "B02.031.001",
]

# Внутренние артикулы МДС — номенклатурными кодами не являются и матчиться
# не должны ни до, ни после фикса.
_INTERNAL_ARTICLES = ["4.1.A2.201", "50.0.H95.201", "1.0.D2.202", "6.1.D1.401", "-"]


@pytest.mark.parametrize("code", _REAL_A_CODES)
def test_real_a_codes_match_nmu_re(code):
    """A-коды номенклатуры имеют 2 цифры в среднем сегменте."""
    assert NMU_RE.match(code), f"{code} не распознан как код номенклатуры"


@pytest.mark.parametrize("code", _REAL_B_CODES)
def test_real_b_codes_still_match(code):
    assert NMU_RE.match(code), f"{code} перестал распознаваться"


@pytest.mark.parametrize("code", _INTERNAL_ARTICLES)
def test_internal_articles_do_not_match(code):
    """Внутренние артикулы клиники не должны считаться кодами номенклатуры."""
    assert not NMU_RE.match(code), f"{code} ошибочно распознан как код номенклатуры"


@pytest.mark.parametrize("code", ["A04.16.001", "A09.05.023", "A11.22.002.001"])
async def test_a_code_service_gives_lab_research_type(code):
    visit = _visit(services=[{"Наименование": "Исследование", "Артикул": code}])
    got = await FormalValidator().get_visit_types(visit)
    assert VisitType.LAB_RESEARCH_INTERVENTION in got


async def test_a_code_with_trailing_space_is_handled():
    """В боевых данных встречаются коды с хвостовым пробелом."""
    visit = _visit(services=[{"Наименование": "УЗИ", "Артикул": "A04.12.018 "}])
    got = await FormalValidator().get_visit_types(visit)
    assert VisitType.LAB_RESEARCH_INTERVENTION in got


async def test_b_code_classification_unchanged():
    """Фикс среднего сегмента не задевает разбор B-кодов."""
    v = FormalValidator()

    primary = await v.get_visit_types(_visit(services=[{"Наименование": "x", "Код": "B01.070.001"}]))
    assert VisitType.PRIMARY in primary

    repeat = await v.get_visit_types(_visit(services=[{"Наименование": "x", "Код": "B01.070.011"}]))
    assert VisitType.REPEAT in repeat

    prof = await v.get_visit_types(_visit(services=[{"Наименование": "x", "Код": "B04.031.002"}]))
    assert VisitType.PROPHYLACTIC in prof

    # B01 с middle != 070 по действующей ветке — намеренно OTHER
    other = await v.get_visit_types(_visit(services=[{"Наименование": "x", "Код": "B01.058.001"}]))
    assert VisitType.OTHER in other
```

- [ ] **Step 2: Прогнать и убедиться, что падают только A-коды**

Run: `pytest tests/test_formal_visit_types.py -v`
Expected: FAIL — 12 параметров `test_real_a_codes_match_nmu_re` и оба теста
`test_a_code_*` про тип визита. Тесты B-кодов и внутренних артикулов — PASS
(это доказывает, что до фикса ломается ровно A-ветка).

- [ ] **Step 3: Починить регулярку**

В `validator.py:29` заменить

```python
NMU_RE = re.compile(r"^[ABАВ]\d{2}\.\d{3}\.\d{3}(?:\.\d{3})?$", re.I)
```

на

```python
# Средний сегмент: 2 цифры у A-кодов (A04.16.001), 3 у B-кодов (B01.070.001).
NMU_RE = re.compile(r"^[ABАВ]\d{2}\.\d{2,3}\.\d{3}(?:\.\d{3})?$", re.I)
```

Меняется **только** средний сегмент. Четвёртый сегмент уже был предусмотрен, и
он нужен: `A04.12.005.005` и `A11.22.002.001` встречаются в боевых данных.
Копировать регулярку из `filters.py` (`[AА]\d{2,}\.\d{2,}\.\d{3,}`) целиком не
надо — она без якорей и пропустила бы внутренние артикулы.

- [ ] **Step 4: Прогнать тесты — должны пройти**

Run: `pytest tests/test_formal_visit_types.py -v`
Expected: PASS всё.

- [ ] **Step 5: Проверить замер на реальной выгрузке**

Если файлы `~/projects/mdsgrep` и `~/projects/alenkagrep` на месте:

```bash
python3 - <<'PY'
import sys
sys.path.insert(0, "src")
from audit.formal_structure.validator import NMU_RE

for name in ("/home/savoy/projects/mdsgrep", "/home/savoy/projects/alenkagrep"):
    try:
        vals = [l.split('": "')[1].rstrip('",\n') for l in open(name) if '": "' in l]
    except FileNotFoundError:
        print(f"{name}: файла нет, пропуск")
        continue
    ok = sum(1 for v in vals if NMU_RE.match(v.strip()))
    print(f"{name}: матчится {ok}/{len(vals)}")
PY
```
Expected: `mdsgrep: матчится 420/999` (было 262), `alenkagrep: матчится 911/999`
(без изменений — там A-кодов нет).

- [ ] **Step 6: Убедиться, что смежные тесты не сломались**

Run: `pytest tests/test_filters.py tests/test_formal_get_rules.py tests/test_formal_rules_schema.py -v`
Expected: PASS. `filters.py` имеет свою регулярку и этой правкой не задет —
прогон подтверждает, что связи нет.

- [ ] **Step 7: Коммит**

```bash
git add src/audit/formal_structure/validator.py tests/test_formal_visit_types.py
git commit -m "fix(formal): NMU_RE — средний сегмент 2 цифры у A-кодов; шесть правил по исследованиям были мертвы"
```

---

### Task 13: Реестр НПА `docs/formal-rules-sources.md`

Спека §3 и Приложение Б. Новый справочник, на который ссылается
`sources_doc` в `rules.json`.

**Files:**
- Create: `docs/formal-rules-sources.md`
- Test: `tests/test_formal_rules_schema.py` (дописать проверку связности)

**Interfaces:**
- Consumes: `source`-ярлыки из готового `rules.json` (Задачи 5–10).
- Produces: документ, у которого на каждый ярлык из `rules.json` есть строка.

- [ ] **Step 1: Написать падающий тест связности**

Дописать в `tests/test_formal_rules_schema.py`:

```python
def test_sources_doc_covers_every_source_label():
    """У каждого ярлыка source из rules.json есть строка в реестре НПА."""
    doc = _doc()
    registry = (ROOT / doc["sources_doc"]).read_text(encoding="utf-8")

    labels = {r.get("source", "") for r in doc["rules"]} - {""}
    missing = sorted(lbl for lbl in labels if f"`{lbl}`" not in registry)
    assert not missing, f"ярлыки без строки в {doc['sources_doc']}: {missing}"
```

- [ ] **Step 2: Прогнать и убедиться, что падает**

Run: `pytest tests/test_formal_rules_schema.py::test_sources_doc_covers_every_source_label -v`
Expected: FAIL — `FileNotFoundError` (документа ещё нет).

- [ ] **Step 3: Создать реестр**

Создать `docs/formal-rules-sources.md` со следующим содержимым:

```markdown
# Реестр НПА правил формальной проверки

Справочник к `src/audit/formal_structure/rules.json`: на какие нормативные
правовые акты опираются правила, в какой редакции они действуют и когда
сверялись. Обновляется при каждой ревизии правил; запись о ревизии — в
`docs/formal-rules-revision-log.md`, раздел «Нормативка».

Дата последней ревизии: **2026-08-19** (поле `revised_at` в `rules.json`).

Колонка «Прочитано» — по чему сверена формулировка: «PDF» — официальная
публикация с pravo.gov.ru (**первоначальная редакция, без поправок**), «HTML» —
консолидированный текст (КонсультантПлюс/Гарант/Контур), «карточка» — только
статус документа. Дословные выжимки по пунктам —
`docs/superpowers/research/2026-08-18-npa-pdf/`.

## Действующие источники

| Ярлык | Документ | Действует | Прочитано | Правила |
|---|---|---|---|---|
| `323-FZ` | Федеральный закон от 21.11.2011 № 323-ФЗ «Об основах охраны здоровья граждан в Российской Федерации» — ст. 2 п. 21, ст. 20 ч. 1, 6, 7, ст. 37 ч. 1 п. 3–4, ст. 54, ст. 70 ч. 2, 5, 6 | бессрочно, редакция на дату сверки | RTF КонсультантПлюс | `diagnosis_should_be_supported`, `management_should_follow_diagnosis`, `followup_needed_for_nontrivial_case`, `diagnosis_justification_presence`, `consent_for_intervention_outside_list`, `legal_representative_info` |
| `274n` | приказ МЗ РФ от 13.05.2025 № 274н «Об утверждении унифицированных форм медицинской документации … в амбулаторных условиях, и порядков по их заполнению» (форма 025/у) | 01.09.2025–01.09.2031 | PDF целиком: прил. 1 (форма 025/у), прил. 2 (п. 10–24), прил. 3–4 (талон 025-1/у) | `visit_meta_required`, `primary_core_sections_required`, `repeat_core_sections_required`, `repeat_needs_context_or_dynamics`, `diagnosis_required`, `objective_exam_required`, `plan_vs_result_separation`, `service_specialty_visit_alignment`, `placeholder_values_are_defect`, `duplicate_semantic_blocks_are_defect`, `manipulation_technique_and_outcome`, `injection_procedure_completeness` |
| `1094n` | приказ МЗ РФ от 24.11.2021 № 1094н «Об утверждении Порядка назначения лекарственных препаратов…» | 01.03.2022–01.03.2028 | PDF: прил. 1 п. 2, 5, 7, 17, 29, 32 | `treatment_dosage_clarity`, `prescription_by_trade_name`, `off_standard_prescription_needs_vk` |
| `404n` | приказ МЗ РФ от 27.04.2021 № 404н «Об утверждении Порядка проведения профилактического медицинского осмотра и диспансеризации … взрослого населения» (ред. 15.06.2026 № 620н) | до 01.07.2027 | PDF (первоначальная ред.): п. 14, 16, 17, 18, 20, 22, 23; HTML (ред. 620н): анти-HCV | `adult_prophylactic_scope`, `adult_prophylactic_health_group`, `adult_prophylactic_counselling`, `adult_prophylactic_marker` |
| `168n` | приказ МЗ РФ от 15.03.2022 № 168н «Об утверждении порядка проведения диспансерного наблюдения за взрослыми» (ред. 28.02.2024 № 91н) | до 01.09.2028 | PDF (первоначальная ред.): п. 4, 13, 14, 15, прил. 1–3 | `dispensary_followup_adult` |
| `192n` | приказ МЗ РФ от 11.04.2025 № 192н «Об утверждении порядка прохождения несовершеннолетними диспансерного наблюдения…» | 01.09.2025–01.09.2031 | PDF: п. 3, 9, 11, 12, 13, 14 | `dispensary_followup_child` |
| `1089n` | приказ МЗ РФ от 23.11.2021 № 1089н «Об утверждении Условий и порядка формирования листков нетрудоспособности…» (ред. 31.03.2026 № 222н) | до 01.03.2029 | PDF (первоначальная ред.): п. 9, 11, 20–22, 44, 46 | `sick_leave_needs_justification`, `sick_leave_over_15_days_needs_vk` |
| `464n` | приказ МЗ РФ от 18.05.2021 № 464н «Об утверждении Правил проведения лабораторных исследований» (ред. 23.11.2021 № 1088н) | 01.09.2021–01.09.2027 | PDF + HTML: прил. 1 к Правилам, п. 16 (совпали) | `lab_biomaterial_and_result` |
| `557n` | приказ МЗ РФ от 08.06.2020 № 557н «Об утверждении Правил проведения ультразвуковых исследований» | с 01.01.2021, срок не установлен | PDF + HTML: п. 19, 20, 24 (совпали) | `ultrasound_protocol_and_conclusion` |
| `560n` | приказ МЗ РФ от 09.06.2020 № 560н «Об утверждении Правил проведения рентгенологических исследований» (ред. 18.02.2021 № 110н) | 01.01.2021–31.08.2026 | PDF + HTML: п. 16, 17, 21 (совпали) | `xray_protocol_and_conclusion` |
| `205n` | приказ МЗ РФ от 14.04.2025 № 205н «Об утверждении Правил проведения функциональных исследований» | 01.09.2025–01.09.2031 | PDF + HTML: п. 18, 19, 24 (совпали) | `ecg_functional_description_and_conclusion` |
| `211n` | приказ МЗ РФ от 14.04.2025 № 211н «Об утверждении Порядка проведения профилактических медицинских осмотров несовершеннолетних» | 01.09.2025–01.09.2031 | PDF: п. 15, 17, 19, 20, прил. 1 | `prophylactic_absence_of_complaints_allowed`, `prophylactic_absence_of_dynamics_allowed`, `prophylactic_no_treatment_allowed_if_decision_present` |
| `190n` | приказ МЗ РФ от 11.04.2025 № 190н «Об утверждении порядка и сроков проведения профилактических медицинских осмотров граждан в целях выявления туберкулёза» | 01.09.2025–01.09.2031 | PDF: п. 7, 18, 19 | `tuberculin_objective_data_required`, `tuberculin_specialist_examination_if_pathology`, `tuberculin_conclusion_required` |
| `icd10` | МКБ-10 — справочник, не нормативный правовой акт (переход на МКБ-11 в РФ не завершён) | — | — | `icd_text_alignment`, `too_general_icd_for_rich_detail` |

Правило `has_typos` источника не имеет — это внутренний стандарт качества.

## Документы, на которые ссылаются `source_ref`, но не ярлыки

| Документ | Действует | Где используется |
|---|---|---|
| приказ МЗ РФ от 12.11.2021 № 1051н «Об утверждении Порядка дачи информированного добровольного согласия…» | 01.03.2022–01.03.2028 | `consent_for_intervention_outside_list` |
| приказ Минздравсоцразвития РФ от 23.04.2012 № 390н «Об утверждении Перечня определённых видов медицинских вмешательств…» | действует | `consent_for_intervention_outside_list` |
| приказ МЗ РФ от 11.04.2025 № 195н «Об утверждении Порядка проведения экспертизы временной нетрудоспособности» | 01.09.2025–01.09.2031 | `sick_leave_needs_justification`, `sick_leave_over_15_days_needs_vk` |
| приказ МЗ РФ от 13.10.2017 № 804н «Об утверждении номенклатуры медицинских услуг» (ред. 24.09.2020) | действует | коды услуг в `expectation` «исследовательских» правил и в детекторе типа визита |

## Плановые смены редакций

| Когда | Что | Действие |
|---|---|---|
| 01.09.2026 | приказ 560н утрачивает силу, вступает приказ МЗ РФ от 08.05.2026 № 359н «Об утверждении Правил проведения рентгенологических исследований и унифицированной формы протокола…» (до 01.09.2032) | у `xray_protocol_and_conclusion` сменить `source` на `359n`, `source_ref` — прил. 1, п. 20, 21, 25 |

## Утратившие силу (для истории `source_ref`)

| Документ | Заменён |
|---|---|
| приказ МЗ РФ от 10.05.2017 № 203н (критерии качества, п. 2.1) | приказ от 14.04.2025 № 203н — универсальных критериев по условиям оказания помощи в нём нет, ярлык `203n` из правил убран |
| приказ МЗ РФ от 26.12.2016 № 997н (функциональные исследования) | приказ от 14.04.2025 № 205н |
| приказ МЗ РФ от 15.12.2014 № 834н (формы 025/у, 030/у) | приказ от 13.05.2025 № 274н; ссылка на форму 030/у в 168н п. 14 после отмены 834н «висит» |
| приказ МЗ РФ от 20.12.2012 № 1177н (ИДС) | приказ от 12.11.2021 № 1051н |
| приказ МЗ РФ от 10.08.2017 № 514н (профосмотры несовершеннолетних) | приказ от 14.04.2025 № 211н |
| приказ МЗ РФ от 21.03.2017 № 124н (туберкулёз) | приказ от 11.04.2025 № 190н |
| приказ МЗ РФ от 23.08.2016 № 625н (экспертиза временной нетрудоспособности) | приказ от 11.04.2025 № 195н |
| приказ МЗ РФ от 20.12.2012 № 1175н (назначение лекарственных препаратов) | приказ от 24.11.2021 № 1094н |

## Открытые хвосты вычитки

Решение 2026-08-19 (спека, §Б.4): вычитку останавливаем, хвосты закрываем при
следующей ревизии — они не влияют на формулировки правил, только на точность
номеров пунктов.

- PDF с pravo.gov.ru — первоначальные редакции. Для **404н** (ред. 620н),
  **168н** (ред. 91н), **1089н** (ред. 222н) и **560н** (ред. 110н) номера
  изменённых пунктов подтверждать по консолидированному тексту
  (КонсультантПлюс/Гарант) перед обновлением `verified_at`.
- PDF приказа **390н** и верного приказа **от 31.03.2026 № 222н** (поправки к
  1089н) не найдены — сверено по HTML-карточкам.
- Форма протокола рентгенологического исследования (прил. 2 к приказу 359н) —
  поля сверх п. 21 не проверялись.
- Соответствие классов кодов номенклатуры услуг (804н) тем, что перечислены в
  `expectation` «исследовательских» правил, не проверялось построчно.
- Правила для типа визита `OTHER` отсутствуют — пробел зафиксирован в
  `docs/audit-2026-07-07.md`, ревизией не закрывался.
- Вне скоупа ревизии (низкий приоритет по аудиту): 947н, 286н/972н, 29н, 785н.
```

- [ ] **Step 4: Прогнать тест связности**

Run: `pytest tests/test_formal_rules_schema.py -v`
Expected: PASS всех тестов файла.

- [ ] **Step 5: Коммит**

```bash
git add docs/formal-rules-sources.md tests/test_formal_rules_schema.py
git commit -m "docs(formal): реестр НПА — редакции, сверки, плановые смены, открытые хвосты"
```

---

### Task 14: Документация и журнал ревизий

Спека §6. Последняя задача перед стендовым гейтом.

**Files:**
- Modify: `docs/formal_validator.md`
- Modify: `docs/formal-rules-revision-log.md`
- Modify: `docs/README.md` (индекс — добавить новый документ)

**Interfaces:**
- Consumes: итоговое состояние `rules.json` и `validator.py`.
- Produces: документацию, соответствующую коду на HEAD ветки.

- [ ] **Step 1: Свериться с фактическим состоянием**

Run:
```bash
python3 -c "
import json
d=json.load(open('src/audit/formal_structure/rules.json'))
print('revised_at:', d['revised_at'], '| правил:', len(d['rules']))
print('с icd_prefixes:', [r['rule_id'] for r in d['rules'] if r['applies_to'].get('icd_prefixes')])"
```
Expected: `revised_at: 2026-08-19 | правил: 42`, с `icd_prefixes` — только
`dispensary_followup_adult`.

- [ ] **Step 2: Обновить `docs/formal_validator.md`**

В разделе «Rule filtering» заменить заголовок
`## Rule filtering — get_rules(visit_types, patient_age) -> list[dict]` на
`## Rule filtering — get_rules(visit_types, patient_age, icd_codes=None) -> list[dict]`
и дописать в конец раздела:

```markdown
Third filter — ICD codes.  A rule carrying `applies_to.icd_prefixes` applies
only when one of the visit's diagnosis codes (`Диагнозы[].КодМКБ`, passed in as
`icd_codes`) starts with one of those prefixes; comparison is upper-case and
whitespace-trimmed.  Without codes such a rule never applies.  Today only
`dispensary_followup_adult` uses it (the 168н list of conditions subject to
dispensary follow-up).
```

Дописать в конец файла новый раздел:

```markdown
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
```

- [ ] **Step 3: Дописать запись в `docs/formal-rules-revision-log.md`**

В разделе «Нормативка» заменить последнюю строку таблицы (`| — | Дата
применения ревизии — дописать при выкатке …`) на:

```markdown
| 2026-08-19 | Ревизия применена: формат `rules.json` с `revised_at`/`source_ref`/`verified_at`; 6 правил сняты с утратившего силу 203н-2017; 464н разъярлычен на 557н/560н/205н/274н; добавлены правила 1094н, 404н, 168н/192н, 1051н+390н, 1089н+195н (31 → 42) | редакции НПА на 2026-08, сверка по PDF первоисточников | ручная ревизия по спеке `docs/superpowers/specs/2026-08-17-formal-rules-npa-revision-design.md` | ветка ревизии нормативки |
```

- [ ] **Step 4: Добавить реестр в индекс `docs/README.md`**

Найти строку с `formal_validator.md` и добавить рядом строку в том же формате,
что принят в файле, со ссылкой на `formal-rules-sources.md` и описанием
«реестр НПА, на которые опираются правила формальной проверки».

- [ ] **Step 5: Прогнать все тесты формальной проверки**

Run: `pytest tests/test_formal_rules_schema.py tests/test_formal_get_rules.py tests/test_formal_validator_meta.py tests/test_formal_visit_types.py -v`
Expected: PASS всё.

- [ ] **Step 6: Коммит**

```bash
git add docs/formal_validator.md docs/formal-rules-revision-log.md docs/README.md
git commit -m "docs(formal): формат rules.json, фильтр по МКБ, запись в журнал ревизий"
```

---

### Task 15: Стендовый гейт (выполняет пользователь)

Спека §5 «Стендовый гейт» и §7 DoD п. 4. Требует БД и LLM — в этой сессии не
выполняется. Задача существует, чтобы гейт не потерялся: **ветка не считается
готовой, пока он не пройден**.

**Files:** изменений нет; при обнаружении дефектов — правки в `rules.json`
отдельными коммитами.

- [ ] **Step 1: Зафиксировать состояние «до»**

На стенде, на коммите-предке ветки:

```bash
python scripts/audit-file.py --file <10-кешированных-карт>.json --excel before.xlsx
```

Выборка карт по спеке: взрослые профилактические визиты, приёмы с
назначениями лекарственных препаратов, ребёнок с листком нетрудоспособности по
уходу.

- [ ] **Step 2: Прогнать «после»**

На HEAD ветки, на тех же картах:

```bash
python scripts/audit-file.py --file <тот-же-файл>.json --excel after.xlsx
```

- [ ] **Step 3: Сравнить**

Критерии приёмки:
- новые флаги (`ПРОФ_ВЗРОСЛЫЙ_*`, `ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО`,
  `НАЗНАЧЕНИЕ_*`, `ЛН_*`, `ИДС_НЕ_ОТРАЖЕНО_ПРИ_ВМЕШАТЕЛЬСТВЕ`) **не срабатывают
  массово** на здоровых картах;
- колонка «Проверка по приказам МЗ» показывает новые ярлыки (`1094n`, `404n`,
  `168n`, `557n`, `560n`, `205n`) и нигде не показывает `203n`;
- отдельно оценить эффект Задачи 11: на визитах с Z11.1 появились
  туберкулиновые флаги (раньше не появлялись никогда). Если срабатывания
  выглядят ложными — откатывать надо именно коммит Задачи 11, а не ревизию.

- [ ] **Step 4: Ручная проверка 5 срабатываний**

Прочитать пять новых находок на осмысленность: соответствует ли текст находки
`expectation` правила и действительно ли в карте нет того, что правило требует.

- [ ] **Step 5: Дописать факт прогона в журнал**

После зелёного гейта — в `docs/formal-rules-revision-log.md` к записи 2026-08-19 добавить
ссылку на коммит ветки, и закоммитить.

---

## Самопроверка плана

**Покрытие спеки:**

| Раздел спеки | Задача |
|---|---|
| §2.1 формат `rules.json` + лог | 1, 2 |
| §2.2 новые поля правила | 5–10 (данные), 4 (тест-инвариант) |
| §2.3 `get_rules` и `icd_prefixes` | 3 |
| §2.4 возрастные группы (`adult`) | 7 (404н), 8 (168н) |
| §3 реестр НПА | 12 |
| §4.1 1094н | 6, 8 (`treatment_dosage_clarity`) |
| §4.2 404н | 7 |
| §4.3 168н/192н + Приложение А | 8 |
| §4.4 1051н/390н/ст. 20 | 9 |
| §4.5 1089н/195н | 9 |
| §4.6 сверка 203н (7 правил) | 5 (шесть), 8 (`treatment_dosage_clarity`) |
| §4.8 сверка старых ярлыков | 10 |
| §5 тесты | 1–4, 7, 8, 11 (юнит), 14 (стенд) |
| §6 документация и журнал | 12, 13 |
| §7 DoD | 1–14 |
| §8 влияние на интеграцию с «Искрой» | правок не требует (спека: «ни кода, ни промптов») |

Сознательно вне плана: §4.7 (что не добавляем), Приложение Б.3 (низкий
приоритет) — это разделы-решения, кода не порождают.

Сверх спеки — два дефекта детектора типа визита, найденные при разборе кода и
включённые в план по решению пользователя, каждый отдельным коммитом:

- **Задача 11** — `visit["Диагноз"]["Код"]` вместо `Диагнозы[].КодМКБ`:
  туберкулиновый тип визита был недостижим, 3 правила `190n` не срабатывали.
- **Задача 12** — `NMU_RE` требовала 3 цифры в среднем сегменте: A-коды
  номенклатуры не распознавались, 6 «исследовательских» правил не срабатывали.
  Замер на боевых выгрузках: МДС теряет 158 из 420 номенклатурных кодов (38%),
  Алёнка — ни одного (A-кодов в её данных нет).

Обе задачи меняют поведение аудита на существующих картах — единственные такие
в плане. На стендовом гейте (Задача 15) их эффект оценивается отдельно от
ревизии нормативки, для чего они и вынесены в самостоятельные коммиты.

## Открытые хвосты, в план не вошли

- **Последний сегмент `NMU_RE`.** Задача 12 меняет только средний сегмент. В
  `filters.py` последний сегмент — `\d{3,}`, в `NMU_RE` — `\d{3}`. В выгрузках
  МДС и Алёнки кодов с четырёхзначным последним сегментом нет, поэтому правка
  не нужна; если такие появятся — вернуться к этому.
- **Две регулярки на одну номенклатуру.** `NMU_RE` в валидаторе и `_CODE_RE` в
  `filters.py:95` описывают одни и те же коды и разошлись. Объединять в этом
  плане не стал: у них разные задачи (якорный матч токена против поиска в
  строке), слияние — отдельное решение.
- **Классы кодов в `expectation` шести «исследовательских» правил** (`A09.*`,
  `A04.*`, `A06.*`, `A05.*`, `A12.*`, `A11.*`, `A16.*`, `A02.*`, `A03.*`) не
  сверялись с разделами номенклатуры 804н построчно — отмечено и в реестре НПА.

**Согласованность имён:** `_RULES_DOC`/`_RULES`/`_REVISED_AT` (Задача 1)
используются в Задачах 2 и 3; `get_rules(visit_types, patient_age, icd_codes)`
(Задача 3) вызывается в тестах Задач 7 и 8 в том же порядке аргументов;
`EXPECTED_RULE_COUNT` растёт 31 → 33 → 37 → 39 → 42 по задачам 4, 6, 7, 8, 9;
`_SHARED_FLAGS` (Задача 4) содержит ровно тот флаг, который Задача 8 даёт двум
правилам.
