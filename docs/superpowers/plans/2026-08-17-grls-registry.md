# Реестр ГРЛС со статусами РУ — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Заменить ЕСКЛП-таблицу `drugs` на выгрузку ГРЛС со статусами РУ (7 статусов), отдавать статус относительно даты визита в `search_medicine`, зафиксировать контракт синка в engine.

**Architecture:** Новый пакет `src/grls/` — чистые функции (нормализация, статус/даты, парсер xlsx, дамп, форматирование справки) без I/O; `src/storage/grls_storage.py` — psycopg3-хранилище с `grls_norm()`-поиском; `scripts/import-grls.py` — тонкий CLI (архив → записи → полная замена в одной транзакции); `SearchMedicineTool` становится обёрткой над `grls.lookup.lookup_medicine` + `format_medicine_lookup`. Всё, что не требует БД, покрыто тестами без БД; storage-тесты — стендовые.

**Tech Stack:** Python 3.11+, psycopg3 (`psycopg[binary]`, `psycopg-pool`), openpyxl, PostgreSQL + `pg_trgm`, pytest (`asyncio_mode=auto`, `pythonpath=src`).

**Spec:** `docs/superpowers/specs/2026-08-17-grls-registry-design.md` (читать целиком перед началом; план ссылается на её параграфы).

## Global Constraints

- Ветка реализации форкается **от `specs-2026-08-17`** (не от release): `git worktree add ../medkard-grls -b grls-registry specs-2026-08-17` (worktree'и лежат в `/home/savoy/projects/worktrees-medkard/`).
- Push — только по явной команде пользователя. Коммитить — после каждой задачи.
- Даты наружу (тул, логи, дамп, доки) — только ISO `YYYY-MM-DD`.
- Значения статусов хранятся **дословно** как маркеры выгрузки (спека §2); статус — из файла, из дат не выводится (спека §5.1).
- Комментарии в коде — кратко, по-английски; тексты для LLM/пользователя — по-русски.
- Тесты гонять точечно (`pytest tests/test_grls_*.py -q`); полный прогон на dev-машине падает без БД — это норма. Storage-тесты (`tests/test_grls_storage.py`) — только на стенде.
- Архив ГРЛС в git не кладём. Локально он лежит в `/home/savoy/projects/grls2026-08-17-1.zip`; распаковывать в scratchpad **своими именами** (`sheet_N.xlsx`) — оригинальные имена в архиве битые (cp437) и слишком длинные для ФС.
- Номер миграции — `027`; если к моменту мёржа появится чужая 027 — перенумеровать (миграции применяются лексикографически).
- `docs/storage.md` в этой ветке отсутствует (лежит untracked в основном checkout) — документацию ГРЛС кладём в новый `docs/grls.md`, а не в `storage.md`.

## Структура файлов

| Файл | Ответственность |
|---|---|
| `migrations/027_grls_registry.sql` | таблицы `grls_registry`/`grls_imports`, `grls_norm()`, индексы, `COMMENT ON COLUMN`, `DROP TABLE drugs` |
| `src/grls/__init__.py` | пустой |
| `src/grls/status.py` | константы статусов, `STATUS_RANK`, `LIVE_STATUSES`, `StatusAtVisit`, `status_at()` |
| `src/grls/normalize.py` | `clean_cell`, `parse_date`, `split_forms`, `derive_dosage_forms`, `derive_dispensing`, `is_substance`, `parse_yes_no`, `parse_narcotic`, `row_hash`, `normalize_query` |
| `src/storage/models/grls_record.py` | dataclass'ы `GrlsRecord`, `GrlsImport` |
| `src/grls/parser.py` | xlsx → `SheetResult(status, registry_date, records)`; `read_sheet`, `read_archive`, `build_record`, `GrlsFormatError` |
| `src/grls/dump.py` | JSONL(.gz)-дамп: `write_dump`, `read_dump` |
| `src/storage/grls_storage.py` | `GrlsStorage`: `replace_all`, `search_by_trade_name`, `search_by_inn`, `inn_status_counts`, `latest_import`, `count` |
| `src/grls/format.py` | `MedicineLookup`, `status_line`, `format_medicine_lookup` (чистое) |
| `src/grls/lookup.py` | `lookup_medicine(query, on)` — открывает storage'ы, собирает `MedicineLookup` |
| `scripts/import-grls.py` | CLI: `<archive|dir> [--dry-run] [--make-dump FILE]` |
| `src/LLM/tools.py` | `SearchMedicineTool` → `lookup_medicine` + `format_medicine_lookup`; удаление `_format_drug`, импортов `Drug` |
| `src/LLM/prompts/treatment_checker.txt` | ЕСКЛП → ГРЛС; правила трактовки статусов |
| `tests/grls_fixtures.py` | билдер мини-xlsx по структуре выгрузки для тестов |
| `tests/test_migration_027.py`, `test_grls_normalize.py`, `test_grls_status.py`, `test_grls_parser.py`, `test_grls_dump.py`, `test_grls_storage.py` (стенд), `test_import_grls_script.py`, `test_grls_format.py`, `test_search_medicine_tool.py` | тесты |
| `docs/grls.md`, `docs/revision-log.md`, `CLAUDE.md`, `.gitignore`, `scripts/seed-reference-lists.sh` | доки/хозяйство |

Удаляются: `src/storage/drugs_storage.py`, `src/storage/models/drug.py`, `resources/Drugs list.csv`, drugs-часть `scripts/seed-reference-lists.sh`.

---

### Task 0: Ветка и worktree

**Files:** —

- [ ] **Step 1: Создать worktree от ветки спек**

```bash
cd /home/savoy/projects/worktrees-medkard/medkard-specs
git fetch origin
git worktree add ../medkard-grls -b grls-registry specs-2026-08-17
cd ../medkard-grls
git log --oneline -1   # ожидаем HEAD ветки спек (994d4d5 или новее)
```

- [ ] **Step 2: Убедиться, что зависимости есть**

Run: `python -c "import openpyxl, psycopg, psycopg_pool; print('ok')"`
Expected: `ok` (иначе `pip install -r requirements.txt`).

---

### Task 1: Миграция 027

**Files:**
- Create: `migrations/027_grls_registry.sql`
- Test: `tests/test_migration_027.py`

**Interfaces:**
- Produces: таблицы `grls_registry`, `grls_imports`; SQL-функция `grls_norm(text)`; статусы CHECK ровно 7 строк из `src/grls/status.py` (Task 2) — строки должны совпасть **посимвольно**.

- [ ] **Step 1: Написать статический тест**

```python
# tests/test_migration_027.py
"""Static assertions on the GRLS registry migration SQL (no DB required)."""
import re
from pathlib import Path

SQL = (Path(__file__).resolve().parent.parent
       / "migrations" / "027_grls_registry.sql").read_text(encoding="utf-8")

STATUSES = (
    "Действующий",
    "Выдано по правилам ЕАЭС",
    "Действует, на подтверждении государственной регистрации",
    "Действует, в иностранных упаковках",
    "Приостановлено применение",
    "Истёкший",
    "Исключённый",
)

REGISTRY_COLUMNS = (
    "id", "status", "reg_number", "registered_at", "expires_at", "annulled_at",
    "holder", "holder_country", "trade_name", "inn_name", "forms", "forms_raw",
    "dosage_forms", "dispensing", "is_substance", "production_stages",
    "normative_docs", "pharm_group", "is_vital", "narcotic_list", "is_orphan",
    "row_hash", "imported_at",
)


def test_creates_both_tables():
    assert "CREATE TABLE IF NOT EXISTS grls_registry" in SQL
    assert "CREATE TABLE IF NOT EXISTS grls_imports" in SQL


def test_status_check_lists_all_seven_verbatim():
    for s in STATUSES:
        assert f"'{s}'" in SQL, s
    assert "CHECK (status IN (" in SQL


def test_row_hash_unique():
    assert re.search(r"row_hash\s+TEXT\s+NOT NULL\s+UNIQUE", SQL)


def test_grls_norm_function_is_immutable():
    assert "CREATE OR REPLACE FUNCTION grls_norm(" in SQL
    assert "IMMUTABLE" in SQL


def test_functional_trgm_indexes():
    assert "USING GIN (grls_norm(trade_name) gin_trgm_ops)" in SQL
    assert "USING GIN (grls_norm(inn_name) gin_trgm_ops)" in SQL


def test_every_registry_column_has_comment():
    for col in REGISTRY_COLUMNS:
        assert f"COMMENT ON COLUMN grls_registry.{col} IS" in SQL, col


def test_drops_drugs_table():
    assert "DROP TABLE IF EXISTS drugs" in SQL
```

- [ ] **Step 2: Запустить — должен упасть**

Run: `pytest tests/test_migration_027.py -q`
Expected: FAIL (`FileNotFoundError`).

- [ ] **Step 3: Написать миграцию**

```sql
-- 027_grls_registry.sql
-- GRLS (State Register of Medicines) with registration-certificate statuses.
-- Replaces the ЕСКЛП `drugs` table. Idempotent, forward-only.
-- Design: docs/superpowers/specs/2026-08-17-grls-registry-design.md

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Mirror of grls.normalize.normalize_query (keep in sync):
-- lower, drop quotes/®™©/~, ё→е, collapse whitespace, trim; empty → NULL.
CREATE OR REPLACE FUNCTION grls_norm(t TEXT) RETURNS TEXT
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT NULLIF(
        btrim(regexp_replace(
            translate(lower(coalesce(t, '')), 'ё"«»„“”‘’''®™©~', 'е'),
            '\s+', ' ', 'g')),
        '')
$$;

CREATE TABLE IF NOT EXISTS grls_registry (
    id                BIGSERIAL PRIMARY KEY,
    status            TEXT NOT NULL CHECK (status IN (
        'Действующий',
        'Выдано по правилам ЕАЭС',
        'Действует, на подтверждении государственной регистрации',
        'Действует, в иностранных упаковках',
        'Приостановлено применение',
        'Истёкший',
        'Исключённый'
    )),
    reg_number        TEXT NOT NULL,
    registered_at     DATE,
    expires_at        DATE,
    annulled_at       DATE,
    holder            TEXT,
    holder_country    TEXT,
    trade_name        TEXT NOT NULL,
    inn_name          TEXT,
    forms             TEXT[] NOT NULL DEFAULT '{}',
    forms_raw         TEXT,
    dosage_forms      TEXT[] NOT NULL DEFAULT '{}',
    dispensing        TEXT[] NOT NULL DEFAULT '{}',
    is_substance      BOOLEAN NOT NULL DEFAULT false,
    production_stages TEXT,
    normative_docs    TEXT,
    pharm_group       TEXT,
    is_vital          BOOLEAN,
    narcotic_list     TEXT,
    is_orphan         BOOLEAN,
    row_hash          TEXT NOT NULL UNIQUE,
    imported_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  grls_registry IS 'ГРЛС: одна строка = одна строка выгрузки xlsx (статус-файл). Номер РУ не уникален.';
COMMENT ON COLUMN grls_registry.id IS 'Суррогатный ключ.';
COMMENT ON COLUMN grls_registry.status IS 'Состояние записи реестра — маркер из строки 6 листа выгрузки, дословно (7 значений). Истина о статусе; из дат не выводится.';
COMMENT ON COLUMN grls_registry.reg_number IS 'Номер регистрационного удостоверения (xlsx: «Номер регистрационного удостоверения»). Не уникален: перерегистрация сохраняет номер, советские номера общие для нескольких фабрик.';
COMMENT ON COLUMN grls_registry.registered_at IS 'Дата выдачи РУ (xlsx: «Дата регистрации»).';
COMMENT ON COLUMN grls_registry.expires_at IS 'Срок действия РУ (xlsx: «Дата окончания действия регистрационного удостоверения»); NULL = бессрочно только для живых статусов.';
COMMENT ON COLUMN grls_registry.annulled_at IS 'Дата аннулирования (xlsx: «Дата аннулирования регистрационного удостоверения»); заполнена в основном у «Исключённый».';
COMMENT ON COLUMN grls_registry.holder IS 'Держатель РУ (xlsx: «Юридическое лицо, на имя которого выдано регистрационное удостоверение»).';
COMMENT ON COLUMN grls_registry.holder_country IS 'Страна держателя (xlsx: безымянная колонка H сразу после ЮЛ).';
COMMENT ON COLUMN grls_registry.trade_name IS 'Торговое наименование (xlsx: «Торговое наименование лекарственного препарата»). Хранится как есть; поиск через grls_norm().';
COMMENT ON COLUMN grls_registry.inn_name IS 'МНН / группировочное / химическое наименование (xlsx: «Международное непатентованное или химическое наименование»); «~» → NULL.';
COMMENT ON COLUMN grls_registry.forms IS 'Формы выпуска, split по «;» (xlsx: «Формы выпуска»): «форма, дозировка, фасовка - упаковка - … - условия отпуска».';
COMMENT ON COLUMN grls_registry.forms_raw IS 'xlsx: «Формы выпуска» как есть — для отладки парсера и пересчёта row_hash.';
COMMENT ON COLUMN grls_registry.dosage_forms IS 'Производное от forms: уникальные лекарственные формы (первый сегмент элемента до запятой).';
COMMENT ON COLUMN grls_registry.dispensing IS 'Производное от forms: уникальные условия отпуска (последний сегмент после « - »): По рецепту / Без рецепта / для стационаров / Не указано / In-Bulk / …';
COMMENT ON COLUMN grls_registry.is_substance IS 'Производное: фармацевтическая субстанция (номер ФС-… или форма «субстанция…»), не препарат; поиск по умолчанию исключает.';
COMMENT ON COLUMN grls_registry.production_stages IS 'Производители по стадиям (xlsx: «Сведения о стадиях производства»).';
COMMENT ON COLUMN grls_registry.normative_docs IS 'Реквизиты НД/ФС и редакций РУ (xlsx: «Нормативная документация»).';
COMMENT ON COLUMN grls_registry.pharm_group IS 'Фармако-терапевтическая группа (xlsx: «Фармако-терапевтическая группа»).';
COMMENT ON COLUMN grls_registry.is_vital IS 'Входит в перечень ЖНВЛП (xlsx: «Наличие лекарственного препарата в перечне ЖНВЛП», Да/Нет).';
COMMENT ON COLUMN grls_registry.narcotic_list IS 'Список ПКУ по ПП РФ № 681 (xlsx: «Наличие в лекарственном препарате наркотических средств, психотропных веществ…»): НII/ПII/ПIII/ПК; «~»/«Нет» → NULL.';
COMMENT ON COLUMN grls_registry.is_orphan IS 'Орфанный препарат (xlsx: «Орфанный»); в выгрузке 2026-08 пусто → NULL.';
COMMENT ON COLUMN grls_registry.row_hash IS 'sha256 нормализованного кортежа исходных полей (см. спеку §4.3) — ключ дедупликации и синка в engine.';
COMMENT ON COLUMN grls_registry.imported_at IS 'Момент загрузки строки.';

CREATE INDEX IF NOT EXISTS grls_registry_trade_name_trgm_idx
    ON grls_registry USING GIN (grls_norm(trade_name) gin_trgm_ops);
CREATE INDEX IF NOT EXISTS grls_registry_inn_trgm_idx
    ON grls_registry USING GIN (grls_norm(inn_name) gin_trgm_ops);
CREATE INDEX IF NOT EXISTS grls_registry_status_idx ON grls_registry (status);
CREATE INDEX IF NOT EXISTS grls_registry_reg_number_idx ON grls_registry (reg_number);
CREATE INDEX IF NOT EXISTS grls_registry_is_substance_idx ON grls_registry (is_substance);

CREATE TABLE IF NOT EXISTS grls_imports (
    id             BIGSERIAL PRIMARY KEY,
    archive_name   TEXT NOT NULL,
    registry_date  DATE NOT NULL,
    status_counts  JSONB NOT NULL,
    skipped_files  JSONB NOT NULL DEFAULT '[]',
    imported_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  grls_imports IS 'Журнал загрузок ГРЛС; последняя строка = текущая версия реестра.';
COMMENT ON COLUMN grls_imports.id IS 'Суррогатный ключ.';
COMMENT ON COLUMN grls_imports.archive_name IS 'Имя архива/каталога выгрузки.';
COMMENT ON COLUMN grls_imports.registry_date IS 'Дата выгрузки ГРЛС из строки 3 листа («по состоянию на DD.MM.YYYY») — версия реестра.';
COMMENT ON COLUMN grls_imports.status_counts IS 'Строк по статусам после дедупликации: {"Действующий": N, …}.';
COMMENT ON COLUMN grls_imports.skipped_files IS 'Пропущенные файлы (напр. «Изменённый»).';
COMMENT ON COLUMN grls_imports.imported_at IS 'Момент загрузки.';

-- ЕСКЛП table replaced by GRLS.
DROP TABLE IF EXISTS drugs;
```

- [ ] **Step 4: Запустить тест**

Run: `pytest tests/test_migration_027.py -q`
Expected: 7 passed.

- [ ] **Step 5: Коммит**

```bash
git add migrations/027_grls_registry.sql tests/test_migration_027.py
git commit -m "feat(grls): migration 027 — grls_registry/grls_imports, grls_norm(), drop drugs"
```

---

### Task 2: Статусы и семантика дат (`src/grls/status.py`)

**Files:**
- Create: `src/grls/__init__.py`, `src/grls/status.py`, `src/storage/models/grls_record.py`
- Test: `tests/test_grls_status.py`

**Interfaces:**
- Produces:
  ```python
  # src/grls/status.py
  STATUS_ACTIVE, STATUS_EAEU, STATUS_CONFIRMING, STATUS_FOREIGN_PACK, STATUS_SUSPENDED, STATUS_EXPIRED, STATUS_ANNULLED: str
  STATUS_CHANGED: str = "Изменённый"          # not loaded
  ALL_STATUSES: tuple[str, ...]              # 7, in rank order
  LIVE_STATUSES: frozenset[str]              # 4 живых (без Приостановлено)
  STATUS_RANK: dict[str, int]                # живые 0, приостановлено 1, истёкший 2, исключённый 3
  class StatusAtVisit(str, Enum): ACTIVE, ACTIVE_WITH_NOTE, VALID_AT_VISIT, EXPIRED, ANNULLED, UNKNOWN_END
  def status_at(record: GrlsRecord, on: date | None) -> StatusAtVisit
  ```
  ```python
  # src/storage/models/grls_record.py
  @dataclass class GrlsRecord: status, reg_number, trade_name (str); registered_at, expires_at, annulled_at (date|None); holder, holder_country, inn_name, forms_raw, production_stages, normative_docs, pharm_group, narcotic_list (str|None); forms, dosage_forms, dispensing (list[str]); is_substance (bool); is_vital, is_orphan (bool|None); row_hash (str); id (int|None=None); imported_at (datetime|None=None)
  @dataclass class GrlsImport: archive_name (str); registry_date (date); status_counts (dict[str,int]); skipped_files (list[str]); id (int|None=None); imported_at (datetime|None=None)
  ```

- [ ] **Step 1: Модели**

```python
# src/storage/models/grls_record.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class GrlsRecord:
    """One row of a GRLS status sheet (see migration 027)."""

    status: str
    reg_number: str
    trade_name: str
    row_hash: str
    registered_at: date | None = None
    expires_at: date | None = None
    annulled_at: date | None = None
    holder: str | None = None
    holder_country: str | None = None
    inn_name: str | None = None
    forms: list[str] = field(default_factory=list)
    forms_raw: str | None = None
    dosage_forms: list[str] = field(default_factory=list)
    dispensing: list[str] = field(default_factory=list)
    is_substance: bool = False
    production_stages: str | None = None
    normative_docs: str | None = None
    pharm_group: str | None = None
    is_vital: bool | None = None
    narcotic_list: str | None = None
    is_orphan: bool | None = None
    id: int | None = None
    imported_at: datetime | None = None


@dataclass
class GrlsImport:
    """One row of grls_imports — a registry version."""

    archive_name: str
    registry_date: date
    status_counts: dict[str, int]
    skipped_files: list[str] = field(default_factory=list)
    id: int | None = None
    imported_at: datetime | None = None
```

Добавить в `src/storage/models/__init__.py`: `from .grls_record import GrlsRecord, GrlsImport` и в `__all__` — `"GrlsRecord", "GrlsImport"` (строку `Drug` пока не трогать — удаляется в Task 9).

- [ ] **Step 2: Тест `status_at`**

```python
# tests/test_grls_status.py
from datetime import date

import pytest

from grls import status as st
from grls.status import StatusAtVisit, status_at
from storage.models.grls_record import GrlsRecord


def _rec(status: str, expires_at=None, annulled_at=None) -> GrlsRecord:
    return GrlsRecord(status=status, reg_number="ЛП-000001", trade_name="Тест",
                      row_hash="h", expires_at=expires_at, annulled_at=annulled_at)


VISIT = date(2025, 3, 10)


def test_constants_are_consistent():
    assert len(st.ALL_STATUSES) == 7
    assert set(st.STATUS_RANK) == set(st.ALL_STATUSES)
    assert st.LIVE_STATUSES == {st.STATUS_ACTIVE, st.STATUS_EAEU,
                                st.STATUS_CONFIRMING, st.STATUS_FOREIGN_PACK}
    assert st.STATUS_CHANGED not in st.ALL_STATUSES


@pytest.mark.parametrize("status", [st.STATUS_ACTIVE, st.STATUS_EAEU])
def test_live_is_active_even_if_expires_in_past(status):
    # status wins over dates: registry has 87 such rows
    assert status_at(_rec(status, expires_at=date(2020, 1, 1)), VISIT) is StatusAtVisit.ACTIVE
    assert status_at(_rec(status), None) is StatusAtVisit.ACTIVE


@pytest.mark.parametrize("status", [st.STATUS_CONFIRMING, st.STATUS_FOREIGN_PACK, st.STATUS_SUSPENDED])
def test_note_statuses(status):
    assert status_at(_rec(status), VISIT) is StatusAtVisit.ACTIVE_WITH_NOTE


def test_expired_before_visit():
    assert status_at(_rec(st.STATUS_EXPIRED, expires_at=date(2025, 1, 1)), VISIT) is StatusAtVisit.EXPIRED


def test_expired_after_visit_is_valid_at_visit():
    assert status_at(_rec(st.STATUS_EXPIRED, expires_at=date(2025, 12, 31)), VISIT) is StatusAtVisit.VALID_AT_VISIT


def test_expired_on_visit_day_is_valid():
    assert status_at(_rec(st.STATUS_EXPIRED, expires_at=VISIT), VISIT) is StatusAtVisit.VALID_AT_VISIT


def test_expired_without_visit_date_is_expired():
    assert status_at(_rec(st.STATUS_EXPIRED, expires_at=date(2025, 12, 31)), None) is StatusAtVisit.EXPIRED


def test_expired_without_boundary_is_unknown():
    assert status_at(_rec(st.STATUS_EXPIRED), VISIT) is StatusAtVisit.UNKNOWN_END


def test_annulled_uses_annulled_at_first():
    r = _rec(st.STATUS_ANNULLED, expires_at=date(2030, 1, 1), annulled_at=date(2024, 2, 14))
    assert status_at(r, VISIT) is StatusAtVisit.ANNULLED


def test_annulled_falls_back_to_expires_at():
    r = _rec(st.STATUS_ANNULLED, expires_at=date(2025, 6, 1))
    assert status_at(r, VISIT) is StatusAtVisit.VALID_AT_VISIT


def test_annulled_without_any_date_is_unknown():
    assert status_at(_rec(st.STATUS_ANNULLED), VISIT) is StatusAtVisit.UNKNOWN_END


def test_unknown_status_raises():
    with pytest.raises(ValueError):
        status_at(_rec("Изменённый"), VISIT)
```

- [ ] **Step 3: Запустить — должен упасть**

Run: `pytest tests/test_grls_status.py -q`
Expected: FAIL (`ModuleNotFoundError: grls`).

- [ ] **Step 4: Реализация**

```python
# src/grls/__init__.py
"""GRLS (State Register of Medicines) import, lookup and formatting."""
```

```python
# src/grls/status.py
"""Registration-certificate statuses and their meaning relative to a visit date.

Status is the truth (taken from the sheet the row came from); dates never
override it — they only soften a dead status when the certificate was still
valid on the visit date. See spec §5.1.
"""
from __future__ import annotations

from datetime import date
from enum import Enum

from storage.models.grls_record import GrlsRecord

STATUS_ACTIVE = "Действующий"
STATUS_EAEU = "Выдано по правилам ЕАЭС"
STATUS_CONFIRMING = "Действует, на подтверждении государственной регистрации"
STATUS_FOREIGN_PACK = "Действует, в иностранных упаковках"
STATUS_SUSPENDED = "Приостановлено применение"
STATUS_EXPIRED = "Истёкший"
STATUS_ANNULLED = "Исключённый"
STATUS_CHANGED = "Изменённый"  # revision journal, not loaded

ALL_STATUSES: tuple[str, ...] = (
    STATUS_ACTIVE, STATUS_EAEU, STATUS_CONFIRMING, STATUS_FOREIGN_PACK,
    STATUS_SUSPENDED, STATUS_EXPIRED, STATUS_ANNULLED,
)
LIVE_STATUSES: frozenset[str] = frozenset(
    {STATUS_ACTIVE, STATUS_EAEU, STATUS_CONFIRMING, STATUS_FOREIGN_PACK})
STATUS_RANK: dict[str, int] = {
    STATUS_ACTIVE: 0, STATUS_EAEU: 0, STATUS_CONFIRMING: 0, STATUS_FOREIGN_PACK: 0,
    STATUS_SUSPENDED: 1, STATUS_EXPIRED: 2, STATUS_ANNULLED: 3,
}


class StatusAtVisit(str, Enum):
    ACTIVE = "active"
    ACTIVE_WITH_NOTE = "active_note"      # confirming / foreign pack / suspended
    VALID_AT_VISIT = "valid_at_visit"     # expired/annulled now, but valid on the visit date
    EXPIRED = "expired"
    ANNULLED = "annulled"
    UNKNOWN_END = "unknown_end"           # dead status without a usable boundary date


def status_at(record: GrlsRecord, on: date | None) -> StatusAtVisit:
    """Interpret record.status relative to visit date `on` (None = no softening)."""
    status = record.status
    if status in (STATUS_ACTIVE, STATUS_EAEU):
        return StatusAtVisit.ACTIVE
    if status in (STATUS_CONFIRMING, STATUS_FOREIGN_PACK, STATUS_SUSPENDED):
        return StatusAtVisit.ACTIVE_WITH_NOTE
    if status == STATUS_EXPIRED:
        boundary = record.expires_at
        dead = StatusAtVisit.EXPIRED
    elif status == STATUS_ANNULLED:
        boundary = record.annulled_at or record.expires_at
        dead = StatusAtVisit.ANNULLED
    else:
        raise ValueError(f"unknown GRLS status: {status!r}")
    if boundary is None:
        return StatusAtVisit.UNKNOWN_END
    if on is not None and boundary >= on:
        return StatusAtVisit.VALID_AT_VISIT
    return dead
```

- [ ] **Step 5: Запустить тесты**

Run: `pytest tests/test_grls_status.py -q`
Expected: все passed (15).

- [ ] **Step 6: Коммит**

```bash
git add src/grls/__init__.py src/grls/status.py src/storage/models/grls_record.py src/storage/models/__init__.py tests/test_grls_status.py
git commit -m "feat(grls): GrlsRecord/GrlsImport models, statuses and status_at()"
```

---

### Task 3: Нормализация (`src/grls/normalize.py`)

**Files:**
- Create: `src/grls/normalize.py`
- Test: `tests/test_grls_normalize.py`

**Interfaces:**
- Produces:
  ```python
  def clean_cell(value: object) -> str | None            # strip, '' и '~' → None, '_x000D_' удалён
  def parse_date(value: object) -> date | None           # 'DD.MM.YYYY[ HH:MM[:SS]]' | datetime | date; мусор → warning + None
  def split_forms(forms_raw: str | None) -> list[str]
  def derive_dosage_forms(forms: list[str]) -> list[str]
  def derive_dispensing(forms: list[str]) -> list[str]
  def is_substance(reg_number: str, dosage_forms: list[str]) -> bool
  def parse_yes_no(value: object) -> bool | None
  def parse_narcotic(value: object) -> str | None
  def row_hash(*, status, reg_number, registered_at, expires_at, annulled_at, holder, holder_country, trade_name, inn_name, forms_raw, production_stages, normative_docs, pharm_group, is_vital, narcotic_list, is_orphan) -> str
  def normalize_query(text: str) -> str                  # зеркало SQL grls_norm()
  ```

- [ ] **Step 1: Тесты**

```python
# tests/test_grls_normalize.py
from datetime import date, datetime

from grls import normalize as n


def test_clean_cell_null_markers_and_xlsx_cr():
    assert n.clean_cell(None) is None
    assert n.clean_cell("") is None
    assert n.clean_cell("  ~ ") is None
    assert n.clean_cell("  Ампициллин ") == "Ампициллин"
    assert n.clean_cell("ЛП-000001-280722_x000D_\nИзм. №1") == "ЛП-000001-280722\nИзм. №1"


def test_parse_date_variants():
    assert n.parse_date("05.06.2000") == date(2000, 6, 5)
    assert n.parse_date("17.08.2026 05:00:00") == date(2026, 8, 17)
    assert n.parse_date(datetime(2024, 2, 14, 10, 0)) == date(2024, 2, 14)
    assert n.parse_date(date(2024, 2, 14)) == date(2024, 2, 14)
    assert n.parse_date("") is None
    assert n.parse_date("~") is None
    assert n.parse_date("2024-02-14") is None      # unexpected format → None + warning
    assert n.parse_date("31.02.2024") is None      # invalid calendar date


FORMS_RAW = ("таблетки, покрытые пленочной оболочкой, 5 мг, 10 шт. - блистеры (2 шт.)  - пачки картонные (20 шт.)  - Без рецепта; "
             "таблетки, покрытые пленочной оболочкой, 5 мг, 7 шт. - блистеры (4 шт.)  - пачки картонные (28 шт.)  - Без рецепта; "
             "мазь для местного и наружного применения, 0.2%, 5 кг - ведра - для стационаров; "
             " - Без рецепта; "
             "капсулы")


def test_split_forms_trims_and_drops_empty():
    forms = n.split_forms(FORMS_RAW)
    assert len(forms) == 5
    assert forms[3] == "- Без рецепта"
    assert n.split_forms(None) == []
    assert n.split_forms("") == []


def test_derive_dosage_forms_unique_in_order_skips_fragments():
    forms = n.split_forms(FORMS_RAW)
    assert n.derive_dosage_forms(forms) == [
        "таблетки", "мазь для местного и наружного применения", "капсулы"]


def test_derive_dispensing_unique_skips_elements_without_separator():
    forms = n.split_forms(FORMS_RAW)
    assert n.derive_dispensing(forms) == ["Без рецепта", "для стационаров"]


def test_is_substance_by_number_or_form():
    assert n.is_substance("ФС-000001", ["субстанция-порошок"]) is True
    assert n.is_substance("ЛП-000001", ["Субстанция-жидкость"]) is True
    assert n.is_substance("ФС-000002", []) is True
    assert n.is_substance("ЛП-000001", ["таблетки"]) is False


def test_parse_yes_no_and_narcotic():
    assert n.parse_yes_no("Да") is True
    assert n.parse_yes_no("нет") is False
    assert n.parse_yes_no("") is None
    assert n.parse_yes_no("~") is None
    assert n.parse_narcotic("~") is None
    assert n.parse_narcotic("Нет") is None
    assert n.parse_narcotic("ПIII") == "ПIII"


def _hash_kwargs(**over):
    base = dict(status="Действующий", reg_number="ЛП-000001", registered_at=date(2020, 1, 1),
                expires_at=None, annulled_at=None, holder="ООО Тест", holder_country="Россия",
                trade_name="Тестин", inn_name="тестамол", forms_raw="таблетки, 5 мг - По рецепту;",
                production_stages=None, normative_docs=None, pharm_group=None,
                is_vital=True, narcotic_list=None, is_orphan=None)
    base.update(over)
    return base


def test_row_hash_deterministic_and_sensitive():
    h1 = n.row_hash(**_hash_kwargs())
    assert h1 == n.row_hash(**_hash_kwargs())
    assert len(h1) == 64
    assert h1 != n.row_hash(**_hash_kwargs(status="Истёкший"))
    assert h1 != n.row_hash(**_hash_kwargs(is_vital=False))
    assert h1 != n.row_hash(**_hash_kwargs(is_vital=None))
    assert h1 != n.row_hash(**_hash_kwargs(registered_at=date(2020, 1, 2)))


def test_row_hash_is_stable_across_versions():
    # Pin the algorithm: engine recomputes this from the dump (spec §4.3/§7).
    expected = n.row_hash(**_hash_kwargs())
    import hashlib
    parts = ["Действующий", "ЛП-000001", "2020-01-01", "", "", "ООО Тест", "Россия",
             "Тестин", "тестамол", "таблетки, 5 мг - По рецепту;", "", "", "", "1", "", ""]
    assert expected == hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def test_normalize_query():
    assert n.normalize_query('  "ЭФКУРИЯ®"  ') == "эфкурия"
    assert n.normalize_query("«Кей Джи Пи»") == "кей джи пи"
    assert n.normalize_query("Ёлкин\tчай") == "елкин чай"
    assert n.normalize_query("Аспирин™ 500") == "аспирин 500"
    assert n.normalize_query("~") == ""
```

- [ ] **Step 2: Запустить — должен упасть**

Run: `pytest tests/test_grls_normalize.py -q`
Expected: FAIL (`ModuleNotFoundError: grls.normalize`).

- [ ] **Step 3: Реализация**

```python
# src/grls/normalize.py
"""Pure normalization helpers for the GRLS import (no I/O)."""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import date, datetime

logger = logging.getLogger(__name__)

_NULL_MARKERS = {"", "~"}
_XLSX_CR = "_x000D_"
_DATE_RE = re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?$")
_HASH_SEP = "\x1f"
_SUBSTANCE_PREFIX = "ФС-"
_SUBSTANCE_FORM = "субстанция"

# Keep in sync with SQL grls_norm() in migrations/027_grls_registry.sql.
_DROP_CHARS = "\"«»„“”‘’'®™©~"
_QUERY_TABLE = str.maketrans({"ё": "е", **{c: None for c in _DROP_CHARS}})


def clean_cell(value: object) -> str | None:
    """Cell → stripped text; '' and '~' → None; xlsx CR artefact removed."""
    if value is None:
        return None
    text = str(value).replace(_XLSX_CR, "").strip()
    return None if text in _NULL_MARKERS else text


def parse_date(value: object) -> date | None:
    """'DD.MM.YYYY[ HH:MM[:SS]]' | datetime | date → date; junk → warning + None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if text in _NULL_MARKERS:
        return None
    m = _DATE_RE.match(text)
    if not m:
        logger.warning("GRLS: unparsable date %r", text)
        return None
    d, mo, y = (int(x) for x in m.groups())
    try:
        return date(y, mo, d)
    except ValueError:
        logger.warning("GRLS: invalid calendar date %r", text)
        return None


def split_forms(forms_raw: str | None) -> list[str]:
    if not forms_raw:
        return []
    return [p.strip() for p in forms_raw.split(";") if p.strip()]


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


def derive_dosage_forms(forms: list[str]) -> list[str]:
    """First comma-segment of each element; fragments starting with '-' skipped."""
    return _unique([el.split(",", 1)[0].strip() for el in forms if not el.startswith("-")])


def derive_dispensing(forms: list[str]) -> list[str]:
    """Last ' - '-segment of each element; elements without the separator skipped."""
    return _unique([el.rsplit(" - ", 1)[1].strip() for el in forms if " - " in el])


def is_substance(reg_number: str, dosage_forms: list[str]) -> bool:
    return reg_number.startswith(_SUBSTANCE_PREFIX) or any(
        f.lower().startswith(_SUBSTANCE_FORM) for f in dosage_forms)


def parse_yes_no(value: object) -> bool | None:
    text = clean_cell(value)
    if text is None:
        return None
    low = text.lower()
    return True if low == "да" else False if low == "нет" else None


def parse_narcotic(value: object) -> str | None:
    text = clean_cell(value)
    if text is None or text.lower() == "нет":
        return None
    return text


def _hash_part(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, date):
        return v.isoformat()
    return str(v)


def row_hash(*, status, reg_number, registered_at, expires_at, annulled_at, holder,
             holder_country, trade_name, inn_name, forms_raw, production_stages,
             normative_docs, pharm_group, is_vital, narcotic_list, is_orphan) -> str:
    """sha256 over the fixed-order source tuple (spec §4.3). Sync contract with engine."""
    parts = (status, reg_number, registered_at, expires_at, annulled_at, holder,
             holder_country, trade_name, inn_name, forms_raw, production_stages,
             normative_docs, pharm_group, is_vital, narcotic_list, is_orphan)
    payload = _HASH_SEP.join(_hash_part(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_query(text: str) -> str:
    """Python mirror of SQL grls_norm(): lower, drop quotes/®™©/~, ё→е, collapse spaces."""
    return " ".join(text.lower().translate(_QUERY_TABLE).split())
```

- [ ] **Step 4: Запустить тесты**

Run: `pytest tests/test_grls_normalize.py -q`
Expected: 10 passed.

- [ ] **Step 5: Коммит**

```bash
git add src/grls/normalize.py tests/test_grls_normalize.py
git commit -m "feat(grls): normalization helpers — cells, dates, forms, row_hash, normalize_query"
```

---

### Task 4: Парсер xlsx (`src/grls/parser.py`) + фикстура-билдер

**Files:**
- Create: `src/grls/parser.py`, `tests/grls_fixtures.py`
- Test: `tests/test_grls_parser.py`

**Interfaces:**
- Consumes: `grls.normalize.*`, `grls.status.ALL_STATUSES/STATUS_CHANGED`, `GrlsRecord`.
- Produces:
  ```python
  class GrlsFormatError(ValueError)
  @dataclass class SheetResult: path: Path; source_name: str; status: str; registry_date: date; records: list[GrlsRecord]; skipped: bool = False
  def build_record(status: str, cells: Sequence[object]) -> GrlsRecord | None   # 15 ячеек C..Q
  def read_sheet(path: Path, source_name: str | None = None) -> SheetResult
  def read_archive(path: Path) -> list[SheetResult]     # zip или каталог с *.xlsx
  ```
  Тестовый билдер: `tests/grls_fixtures.py: make_sheet(path, status, rows, registry_date="17.08.2026", trailer=True)`; `rows` — список из 15-элементных кортежей (C..Q) строк; `sample_row(**over) -> tuple`.

- [ ] **Step 1: Билдер мини-xlsx**

```python
# tests/grls_fixtures.py
"""Build minimal GRLS-like xlsx sheets for tests (layout of the 2026-08-17 export)."""
from __future__ import annotations

from pathlib import Path

import openpyxl

HEADERS = (
    "Номер регистрационного удостоверения",
    "Дата регистрации",
    "Дата окончания действия регистрационного удостоверения",
    "Дата аннулирования регистрацион- ного удостоверения",
    "Юридическое лицо, на имя которого выдано регистрационное удостоверение",
    None,
    "Торговое наименование\nлекарственного препарата",
    "Международное непатентованное или химическое наименование",
    "Формы выпуска",
    "Сведения о стадиях производства",
    "Нормативная документация",
    "Фармако-терапевтическая группа",
    "Наличие лекарственного препарата в перечне ЖНВЛП",
    "Наличие в лекарственном препарате наркотических средств, психотропных веществ",
    "Орфанный",
)


def sample_row(**over) -> tuple:
    base = {
        "reg_number": "ЛП-000001", "registered_at": "01.02.2020", "expires_at": "",
        "annulled_at": "", "holder": 'ООО "Тест"', "holder_country": "Россия",
        "trade_name": "Тестин®", "inn_name": "тестамол",
        "forms_raw": "таблетки, 5 мг, 10 шт. - блистеры - пачки картонные - По рецепту; ",
        "production_stages": "Все стадии, ООО Тест, Россия",
        "normative_docs": "ЛП-000001-010220_x000D_\nИзм. №1",
        "pharm_group": "анальгетик", "is_vital": "Да", "narcotic": "~", "is_orphan": "",
    }
    base.update(over)
    return (base["reg_number"], base["registered_at"], base["expires_at"], base["annulled_at"],
            base["holder"], base["holder_country"], base["trade_name"], base["inn_name"],
            base["forms_raw"], base["production_stages"], base["normative_docs"],
            base["pharm_group"], base["is_vital"], base["narcotic"], base["is_orphan"])


def make_sheet(path: Path, status: str, rows: list[tuple], *,
               registry_date: str = "17.08.2026", trailer: bool = True,
               headers: tuple = HEADERS, title: str | None = None) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    # rows 1-2 empty; row 3 title in column D; row 4 empty; row 5 headers from C; row 6 marker in C
    ws.cell(row=3, column=4, value=title if title is not None
            else f"Государственный реестр лекарственных средств\nпо состоянию на {registry_date}")
    for i, h in enumerate(headers):
        if h is not None:
            ws.cell(row=5, column=3 + i, value=h)
    ws.cell(row=6, column=3, value=status)
    r = 7
    for row in rows:
        for i, v in enumerate(row):
            if v is not None:
                ws.cell(row=r, column=3 + i, value=v)
        r += 1
    if trailer:
        ws.cell(row=r, column=3, value=f"{registry_date} 05:00:00")
    wb.save(path)
    return path
```

- [ ] **Step 2: Тесты парсера**

```python
# tests/test_grls_parser.py
import zipfile
from datetime import date
from pathlib import Path

import pytest

from grls import status as st
from grls.parser import GrlsFormatError, build_record, read_archive, read_sheet
from grls_fixtures import HEADERS, make_sheet, sample_row


def test_read_sheet_marker_date_and_rows(tmp_path: Path):
    p = make_sheet(tmp_path / "a.xlsx", st.STATUS_ACTIVE,
                   [sample_row(), sample_row(reg_number="ЛП-000002", trade_name="Другой")])
    res = read_sheet(p)
    assert res.status == st.STATUS_ACTIVE
    assert res.registry_date == date(2026, 8, 17)
    assert res.skipped is False
    assert [r.reg_number for r in res.records] == ["ЛП-000001", "ЛП-000002"]


def test_trailer_row_is_not_a_record(tmp_path: Path):
    p = make_sheet(tmp_path / "a.xlsx", st.STATUS_ACTIVE, [sample_row()], trailer=True)
    assert len(read_sheet(p).records) == 1


def test_record_fields_are_normalized(tmp_path: Path):
    p = make_sheet(tmp_path / "a.xlsx", st.STATUS_EXPIRED,
                   [sample_row(expires_at="31.12.2025", inn_name="~", is_vital="Нет")])
    rec = read_sheet(p).records[0]
    assert rec.status == st.STATUS_EXPIRED
    assert rec.registered_at == date(2020, 2, 1)
    assert rec.expires_at == date(2025, 12, 31)
    assert rec.annulled_at is None
    assert rec.inn_name is None
    assert rec.is_vital is False
    assert rec.narcotic_list is None
    assert rec.is_orphan is None
    assert rec.forms == ["таблетки, 5 мг, 10 шт. - блистеры - пачки картонные - По рецепту"]
    assert rec.dosage_forms == ["таблетки"]
    assert rec.dispensing == ["По рецепту"]
    assert rec.is_substance is False
    assert rec.normative_docs == "ЛП-000001-010220\nИзм. №1"
    assert len(rec.row_hash) == 64


def test_substance_row_flagged(tmp_path: Path):
    p = make_sheet(tmp_path / "a.xlsx", st.STATUS_ACTIVE, [sample_row(
        reg_number="ФС-000001", trade_name="Норфлоксацин",
        forms_raw="субстанция-порошок, ~, 25 кг - пакеты - барабаны - Не указано;")])
    rec = read_sheet(p).records[0]
    assert rec.is_substance is True
    assert rec.dosage_forms == ["субстанция-порошок"]
    assert rec.dispensing == ["Не указано"]


def test_changed_sheet_is_skipped(tmp_path: Path):
    p = make_sheet(tmp_path / "ch.xlsx", st.STATUS_CHANGED, [sample_row()])
    res = read_sheet(p)
    assert res.skipped is True
    assert res.records == []
    assert res.status == st.STATUS_CHANGED


def test_unknown_marker_raises(tmp_path: Path):
    p = make_sheet(tmp_path / "x.xlsx", "Неведомый", [sample_row()])
    with pytest.raises(GrlsFormatError):
        read_sheet(p)


def test_wrong_headers_raise(tmp_path: Path):
    bad = list(HEADERS)
    bad[0] = "Номер чего-то другого"
    p = make_sheet(tmp_path / "x.xlsx", st.STATUS_ACTIVE, [sample_row()], headers=tuple(bad))
    with pytest.raises(GrlsFormatError):
        read_sheet(p)


def test_missing_registry_date_raises(tmp_path: Path):
    p = make_sheet(tmp_path / "x.xlsx", st.STATUS_ACTIVE, [sample_row()],
                   title="Государственный реестр лекарственных средств")
    with pytest.raises(GrlsFormatError):
        read_sheet(p)


def test_build_record_skips_empty_and_nameless():
    assert build_record(st.STATUS_ACTIVE, ("",) * 15) is None
    assert build_record(st.STATUS_ACTIVE, ("17.08.2026 05:00:00",) + (None,) * 14) is None
    assert build_record(st.STATUS_ACTIVE, sample_row(trade_name="")) is None


def test_read_archive_zip_and_dir(tmp_path: Path):
    d = tmp_path / "xlsx"
    d.mkdir()
    make_sheet(d / "1.xlsx", st.STATUS_ACTIVE, [sample_row()])
    make_sheet(d / "2.xlsx", st.STATUS_CHANGED, [sample_row()])
    zpath = tmp_path / "grls.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for f in sorted(d.iterdir()):
            # long/garbled names like the real export must not matter
            z.write(f, arcname="grls2026-08-17-1-" + "Действующий" * 20 + f.name)
    from_dir = read_archive(d)
    from_zip = read_archive(zpath)
    for results in (from_dir, from_zip):
        assert sorted(r.status for r in results) == sorted([st.STATUS_ACTIVE, st.STATUS_CHANGED])
        assert sum(len(r.records) for r in results) == 1
```

- [ ] **Step 3: Запустить — должен упасть**

Run: `pytest tests/test_grls_parser.py -q`
Expected: FAIL (`ModuleNotFoundError: grls.parser`). Если падает раньше на `from grls_fixtures import …` — `tests/` является пакетом (`__init__.py`), тогда во всех тестах ГРЛС писать `from tests.grls_fixtures import …` (и так же в Tasks 5–9).

- [ ] **Step 4: Реализация парсера**

```python
# src/grls/parser.py
"""Read GRLS xlsx exports (one status sheet per file) into GrlsRecord objects."""
from __future__ import annotations

import logging
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import openpyxl

from grls.normalize import (clean_cell, derive_dispensing, derive_dosage_forms,
                            is_substance, parse_date, parse_narcotic, parse_yes_no,
                            row_hash, split_forms)
from grls.status import ALL_STATUSES, STATUS_CHANGED
from storage.models.grls_record import GrlsRecord

logger = logging.getLogger(__name__)

TITLE_ROW, HEADER_ROW, MARKER_ROW = 3, 5, 6
FIRST_COL = 2          # zero-based index of column C
N_COLS = 15            # C..Q
_TITLE_COL = 3         # column D
_REGISTRY_DATE_RE = re.compile(r"по состоянию на\s+(\d{2}\.\d{2}\.\d{4})")
# Header prefixes (whitespace-collapsed); None = column must have no header (H = country).
EXPECTED_HEADER_PREFIXES: tuple[str | None, ...] = (
    "Номер регистрационного удостоверения", "Дата регистрации", "Дата окончания действия",
    "Дата аннулирования", "Юридическое лицо", None, "Торговое наименование",
    "Международное непатентованное", "Формы выпуска", "Сведения о стадиях производства",
    "Нормативная документация", "Фармако-терапевтическая группа",
    "Наличие лекарственного препарата в перечне ЖНВЛП",
    "Наличие в лекарственном препарате наркотических", "Орфанный",
)


class GrlsFormatError(ValueError):
    """The xlsx does not look like a GRLS export (layout changed?)."""


@dataclass
class SheetResult:
    path: Path
    source_name: str
    status: str
    registry_date: date
    records: list[GrlsRecord]
    skipped: bool = False


def _slice(row: Sequence[object] | None) -> tuple:
    row = tuple(row or ()) + (None,) * (FIRST_COL + N_COLS)
    return row[FIRST_COL:FIRST_COL + N_COLS]


def _norm_header(value: object) -> str | None:
    text = " ".join(str(value).split()) if value is not None else ""
    return text or None


def _check_headers(cells: Sequence[object], name: str) -> None:
    for i, (expected, got) in enumerate(zip(EXPECTED_HEADER_PREFIXES, cells)):
        actual = _norm_header(got)
        if expected is None:
            if actual is not None:
                raise GrlsFormatError(f"{name}: column {i} expected empty header, got {actual!r}")
        elif actual is None or not actual.startswith(expected):
            raise GrlsFormatError(f"{name}: column {i} header {actual!r} does not start with {expected!r}")


def build_record(status: str, cells: Sequence[object]) -> GrlsRecord | None:
    """15 cells (C..Q) → GrlsRecord; None for blank/trailer/nameless rows."""
    (reg_number, registered_at, expires_at, annulled_at, holder, holder_country,
     trade_name, inn_name, forms_raw, production_stages, normative_docs, pharm_group,
     vital, narcotic, orphan) = (clean_cell(c) for c in cells)
    if reg_number is None:
        return None
    others = (registered_at, expires_at, annulled_at, holder, holder_country, trade_name,
              inn_name, forms_raw, production_stages, normative_docs, pharm_group)
    if all(v is None for v in others):
        return None  # trailer row (export date) or junk
    if trade_name is None:
        logger.warning("GRLS: row %s without trade name skipped", reg_number)
        return None
    reg_d, exp_d, ann_d = parse_date(registered_at), parse_date(expires_at), parse_date(annulled_at)
    is_vital, is_orphan, narcotic_list = parse_yes_no(vital), parse_yes_no(orphan), parse_narcotic(narcotic)
    forms = split_forms(forms_raw)
    dosage_forms = derive_dosage_forms(forms)
    return GrlsRecord(
        status=status, reg_number=reg_number, trade_name=trade_name,
        registered_at=reg_d, expires_at=exp_d, annulled_at=ann_d,
        holder=holder, holder_country=holder_country, inn_name=inn_name,
        forms=forms, forms_raw=forms_raw, dosage_forms=dosage_forms,
        dispensing=derive_dispensing(forms),
        is_substance=is_substance(reg_number, dosage_forms),
        production_stages=production_stages, normative_docs=normative_docs,
        pharm_group=pharm_group, is_vital=is_vital, narcotic_list=narcotic_list,
        is_orphan=is_orphan,
        row_hash=row_hash(
            status=status, reg_number=reg_number, registered_at=reg_d, expires_at=exp_d,
            annulled_at=ann_d, holder=holder, holder_country=holder_country,
            trade_name=trade_name, inn_name=inn_name, forms_raw=forms_raw,
            production_stages=production_stages, normative_docs=normative_docs,
            pharm_group=pharm_group, is_vital=is_vital, narcotic_list=narcotic_list,
            is_orphan=is_orphan),
    )


def read_sheet(path: Path, source_name: str | None = None) -> SheetResult:
    name = source_name or path.name
    wb = openpyxl.load_workbook(path, read_only=True)
    try:
        ws = wb.worksheets[0]
        rows = ws.iter_rows(min_row=1, values_only=True)
        head = [tuple(next(rows, ())) for _ in range(MARKER_ROW)]
        title_cells = head[TITLE_ROW - 1] + (None,) * (_TITLE_COL + 1)
        m = _REGISTRY_DATE_RE.search(str(title_cells[_TITLE_COL] or ""))
        if not m:
            raise GrlsFormatError(f"{name}: registry date not found in row {TITLE_ROW}")
        registry_date = parse_date(m.group(1))
        assert registry_date is not None
        _check_headers(_slice(head[HEADER_ROW - 1]), name)
        status = clean_cell(_slice(head[MARKER_ROW - 1])[0])
        if status == STATUS_CHANGED:
            logger.info("GRLS: %s is the revision journal (%s) — skipped", name, status)
            return SheetResult(path, name, status, registry_date, [], skipped=True)
        if status not in ALL_STATUSES:
            raise GrlsFormatError(f"{name}: unknown status marker {status!r}")
        records = [r for r in (build_record(status, _slice(row)) for row in rows) if r is not None]
        return SheetResult(path, name, status, registry_date, records)
    finally:
        wb.close()


def _decode_zip_name(raw: str) -> str:
    try:
        return raw.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return raw


def read_archive(path: Path) -> list[SheetResult]:
    """zip or directory with *.xlsx → one SheetResult per sheet (sorted by source name)."""
    path = Path(path)
    if path.is_dir():
        return [read_sheet(p) for p in sorted(path.glob("*.xlsx"))]
    results: list[SheetResult] = []
    with zipfile.ZipFile(path) as z, tempfile.TemporaryDirectory() as tmp:
        members = [i for i in z.infolist() if i.filename.lower().endswith(".xlsx")]
        for idx, info in enumerate(sorted(members, key=lambda i: i.filename)):
            # Names in the export are cp437-garbled and too long for the FS — use our own.
            target = Path(tmp) / f"sheet_{idx}.xlsx"
            target.write_bytes(z.read(info))
            results.append(read_sheet(target, source_name=_decode_zip_name(info.filename)))
    return results
```

- [ ] **Step 5: Запустить тесты**

Run: `pytest tests/test_grls_parser.py -q`
Expected: 10 passed.

- [ ] **Step 6: Прогнать на реальном архиве (без БД, смоук)**

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "src")
from pathlib import Path
from grls.parser import read_archive
res = read_archive(Path("/home/savoy/projects/grls2026-08-17-1.zip"))
for r in res:
    print(r.status, "skipped" if r.skipped else len(r.records), r.registry_date, r.source_name[:40])
EOF
```
Expected: 8 строк; `Изменённый skipped`; счётчики ≈ таблице спеки §2 (Действующий 19 665, ЕАЭС 9 259, подтверждении 77, иностр. 36, приостановлено 33, истёкший 5 474, исключённый 4 342 — допустимы отклонения ±несколько строк из-за пропуска строк без торгового наименования; логи предупреждений посмотреть); `registry_date` = `2026-08-17` у всех.

- [ ] **Step 7: Коммит**

```bash
git add src/grls/parser.py tests/grls_fixtures.py tests/test_grls_parser.py
git commit -m "feat(grls): xlsx parser — status marker, registry date, header check, record build"
```

---

### Task 5: Дамп JSONL (`src/grls/dump.py`)

**Files:**
- Create: `src/grls/dump.py`
- Test: `tests/test_grls_dump.py`

**Interfaces:**
- Consumes: `GrlsRecord`, `grls.normalize.row_hash`.
- Produces:
  ```python
  def record_to_dict(rec: GrlsRecord) -> dict            # без id/imported_at, даты ISO
  def record_from_dict(d: dict) -> GrlsRecord            # пересчитывает row_hash и сверяет с полем
  def write_dump(path: Path, records: Iterable[GrlsRecord], *, registry_date: date, archive_name: str) -> int
  def read_dump(path: Path) -> tuple[dict, list[GrlsRecord]]   # (meta, records)
  ```
  Формат: первая строка `{"_meta": {"registry_date": "2026-08-17", "archive_name": "...", "row_count": N}}`; далее по записи в строке; `.gz` — gzip.

- [ ] **Step 1: Тесты**

```python
# tests/test_grls_dump.py
import gzip
import json
from datetime import date
from pathlib import Path

import pytest

from grls import status as st
from grls.dump import read_dump, record_from_dict, record_to_dict, write_dump
from grls.parser import build_record
from grls_fixtures import sample_row


def _records():
    return [build_record(st.STATUS_ACTIVE, sample_row()),
            build_record(st.STATUS_EXPIRED, sample_row(reg_number="ЛП-000002", expires_at="31.12.2025"))]


def test_record_dict_roundtrip_iso_dates_no_id():
    rec = _records()[1]
    d = record_to_dict(rec)
    assert "id" not in d and "imported_at" not in d
    assert d["expires_at"] == "2025-12-31"
    assert d["forms"] == rec.forms
    back = record_from_dict(d)
    assert back == rec


def test_record_from_dict_rejects_hash_mismatch():
    d = record_to_dict(_records()[0])
    d["trade_name"] = "Подмена"
    with pytest.raises(ValueError):
        record_from_dict(d)


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz"])
def test_write_and_read_dump(tmp_path: Path, suffix: str):
    p = tmp_path / f"grls{suffix}"
    n = write_dump(p, _records(), registry_date=date(2026, 8, 17), archive_name="grls2026-08-17-1.zip")
    assert n == 2
    opener = gzip.open if suffix.endswith(".gz") else open
    with opener(p, "rt", encoding="utf-8") as fh:
        first = json.loads(fh.readline())
    assert first == {"_meta": {"registry_date": "2026-08-17", "archive_name": "grls2026-08-17-1.zip", "row_count": 2}}
    meta, records = read_dump(p)
    assert meta["registry_date"] == "2026-08-17"
    assert [r.reg_number for r in records] == ["ЛП-000001", "ЛП-000002"]
```

- [ ] **Step 2: Запустить — должен упасть**

Run: `pytest tests/test_grls_dump.py -q`
Expected: FAIL (`ModuleNotFoundError: grls.dump`).

- [ ] **Step 3: Реализация**

```python
# src/grls/dump.py
"""JSONL(.gz) dump of grls_registry — the sync format for engine (spec §7)."""
from __future__ import annotations

import gzip
import json
from dataclasses import fields
from datetime import date
from pathlib import Path
from typing import IO, Iterable

from grls.normalize import row_hash
from storage.models.grls_record import GrlsRecord

_EXCLUDED = {"id", "imported_at"}
_DATE_FIELDS = ("registered_at", "expires_at", "annulled_at")
_HASH_FIELDS = ("status", "reg_number", "registered_at", "expires_at", "annulled_at", "holder",
                "holder_country", "trade_name", "inn_name", "forms_raw", "production_stages",
                "normative_docs", "pharm_group", "is_vital", "narcotic_list", "is_orphan")


def record_to_dict(rec: GrlsRecord) -> dict:
    out = {}
    for f in fields(GrlsRecord):
        if f.name in _EXCLUDED:
            continue
        v = getattr(rec, f.name)
        out[f.name] = v.isoformat() if isinstance(v, date) else v
    return out


def record_from_dict(d: dict) -> GrlsRecord:
    data = {k: v for k, v in d.items() if k not in _EXCLUDED}
    for k in _DATE_FIELDS:
        if data.get(k):
            data[k] = date.fromisoformat(data[k])
        else:
            data[k] = None
    rec = GrlsRecord(**data)
    expected = row_hash(**{k: getattr(rec, k) for k in _HASH_FIELDS})
    if rec.row_hash != expected:
        raise ValueError(f"row_hash mismatch for {rec.reg_number} / {rec.trade_name}")
    return rec


def _open(path: Path, mode: str) -> IO[str]:
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def write_dump(path: Path, records: Iterable[GrlsRecord], *, registry_date: date, archive_name: str) -> int:
    records = list(records)
    with _open(Path(path), "w") as fh:
        fh.write(json.dumps({"_meta": {"registry_date": registry_date.isoformat(),
                                        "archive_name": archive_name,
                                        "row_count": len(records)}}, ensure_ascii=False) + "\n")
        for rec in records:
            fh.write(json.dumps(record_to_dict(rec), ensure_ascii=False) + "\n")
    return len(records)


def read_dump(path: Path) -> tuple[dict, list[GrlsRecord]]:
    with _open(Path(path), "r") as fh:
        first = json.loads(fh.readline())
        meta = first["_meta"]
        records = [record_from_dict(json.loads(line)) for line in fh if line.strip()]
    return meta, records
```

- [ ] **Step 4: Запустить тесты**

Run: `pytest tests/test_grls_dump.py -q`
Expected: 4 passed.

- [ ] **Step 5: Коммит**

```bash
git add src/grls/dump.py tests/test_grls_dump.py
git commit -m "feat(grls): JSONL dump read/write with row_hash verification"
```

---

### Task 6: `GrlsStorage` (`src/storage/grls_storage.py`) + стендовые тесты

**Files:**
- Create: `src/storage/grls_storage.py`
- Test: `tests/test_grls_storage.py` (стенд; на dev-машине не запускать)

**Interfaces:**
- Consumes: `BaseStorage`, `GrlsRecord`, `GrlsImport`, `grls.status.STATUS_RANK`, `grls.normalize.normalize_query`.
- Produces:
  ```python
  class GrlsStorage(BaseStorage):
      async def replace_all(self, records: list[GrlsRecord], imp: GrlsImport) -> int   # returns inserted count
      async def search_by_trade_name(self, query: str, *, threshold: float = 0.85, limit: int = 6, include_substances: bool = False) -> list[GrlsRecord]
      async def search_by_inn(self, query: str, *, limit: int = 20, include_substances: bool = False) -> list[GrlsRecord]
      async def inn_status_counts(self, query: str, *, include_substances: bool = False) -> dict[str, int]
      async def latest_import(self) -> GrlsImport | None
      async def count(self) -> int
  ```

- [ ] **Step 1: Стендовый тест (написать сейчас, запустить на стенде в Task 10)**

```python
# tests/test_grls_storage.py
"""Интеграционные тесты storage.grls_storage.GrlsStorage.

Требует Postgres (.env) с применённой миграцией 027. Запускается на стенде —
на dev-машине нет доступа к БД. Тесты подменяют содержимое grls_registry
целиком (replace_all) — не гонять на БД с боевыми данными без последующего
повторного импорта.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from grls import status as st
from grls.normalize import normalize_query
from grls.parser import build_record
from grls_fixtures import sample_row
from storage.grls_storage import GrlsStorage
from storage.models.grls_record import GrlsImport


def _fixture_records():
    return [
        build_record(st.STATUS_EXPIRED, sample_row(reg_number="ЛП-000001", trade_name="Амоксиклав®",
                                                   inn_name="амоксициллин+клавулановая кислота", expires_at="31.12.2025")),
        build_record(st.STATUS_ACTIVE, sample_row(reg_number="ЛП-000002", trade_name="АМОКСИКЛАВ",
                                                  inn_name="амоксициллин+клавулановая кислота")),
        build_record(st.STATUS_ANNULLED, sample_row(reg_number="ЛП-000003", trade_name="амоксиклав",
                                                    inn_name="амоксициллин+клавулановая кислота", annulled_at="14.02.2024")),
        build_record(st.STATUS_ACTIVE, sample_row(reg_number="ФС-000001", trade_name="Амоксициллин",
                                                  inn_name="амоксициллин",
                                                  forms_raw="субстанция-порошок, ~, 25 кг - мешки - Не указано;")),
    ]


def _import():
    return GrlsImport(archive_name="test", registry_date=date(2026, 8, 17),
                      status_counts={st.STATUS_ACTIVE: 2, st.STATUS_EXPIRED: 1, st.STATUS_ANNULLED: 1},
                      skipped_files=["Изменённый"])


async def test_replace_all_and_latest_import():
    async with GrlsStorage() as s:
        n = await s.replace_all(_fixture_records(), _import())
        assert n == 4
        assert await s.count() == 4
        imp = await s.latest_import()
        assert imp is not None and imp.registry_date == date(2026, 8, 17)
        # idempotent: second run replaces, not appends
        assert await s.replace_all(_fixture_records(), _import()) == 4
        assert await s.count() == 4


async def test_search_by_trade_name_orders_by_status_and_ignores_case_and_marks():
    async with GrlsStorage() as s:
        await s.replace_all(_fixture_records(), _import())
        got = await s.search_by_trade_name('"амоксиклав®"')
        assert [r.status for r in got][:2] == [st.STATUS_ACTIVE, st.STATUS_EXPIRED]
        assert got[-1].status == st.STATUS_ANNULLED


async def test_search_by_inn_composite_and_substance_filter():
    async with GrlsStorage() as s:
        await s.replace_all(_fixture_records(), _import())
        got = await s.search_by_inn("амоксициллин + клавулановая кислота")
        assert {r.reg_number for r in got} == {"ЛП-000001", "ЛП-000002", "ЛП-000003"}
        assert await s.search_by_inn("амоксициллин") == []          # substance hidden
        assert len(await s.search_by_inn("амоксициллин", include_substances=True)) >= 1
        counts = await s.inn_status_counts("амоксициллин+клавулановая кислота")
        assert counts == {st.STATUS_ACTIVE: 1, st.STATUS_EXPIRED: 1, st.STATUS_ANNULLED: 1}


async def test_grls_norm_parity_with_python():
    samples = ['  "ЭФКУРИЯ®"  ', "«Кей Джи Пи»", "Ёлкин\tчай", "Аспирин™ 500", "~", "Bayer's"]
    async with GrlsStorage() as s:
        async with s._pool.connection() as conn:
            for text in samples:
                cur = await conn.execute("SELECT grls_norm(%(t)s) AS v", {"t": text})
                row = await cur.fetchone()
                assert (row["v"] or "") == normalize_query(text), text
```

- [ ] **Step 2: Реализация**

```python
# src/storage/grls_storage.py
"""GrlsStorage — async psycopg3 interface for grls_registry / grls_imports (migration 027)."""
from __future__ import annotations

from psycopg import sql
from psycopg.types.json import Jsonb

from grls.normalize import normalize_query
from grls.status import STATUS_RANK
from storage.base import BaseStorage
from storage.models.grls_record import GrlsImport, GrlsRecord

_COLS = ("status", "reg_number", "registered_at", "expires_at", "annulled_at", "holder",
         "holder_country", "trade_name", "inn_name", "forms", "forms_raw", "dosage_forms",
         "dispensing", "is_substance", "production_stages", "normative_docs", "pharm_group",
         "is_vital", "narcotic_list", "is_orphan", "row_hash")
_SELECT_COLS = "id, imported_at, " + ", ".join(_COLS)
_INSERT_SQL = (
    f"INSERT INTO grls_registry ({', '.join(_COLS)}) VALUES ("
    + ", ".join(f"%({c})s" for c in _COLS)
    + ") ON CONFLICT (row_hash) DO NOTHING"
)
_BATCH = 1000
_INN_FUZZY_THRESHOLD = 0.6

_RANK_CASE = sql.SQL("CASE status {} ELSE 9 END").format(
    sql.SQL(" ").join(sql.SQL("WHEN {} THEN {}").format(sql.Literal(s), sql.Literal(r))
                      for s, r in STATUS_RANK.items()))
_ORDER = sql.SQL("ORDER BY {} ASC, sim DESC, expires_at DESC NULLS FIRST").format(_RANK_CASE)


def _row_to_record(row: dict) -> GrlsRecord:
    return GrlsRecord(**{k: row[k] for k in ("id", "imported_at", *_COLS)})


def _record_params(rec: GrlsRecord) -> dict:
    return {c: getattr(rec, c) for c in _COLS}


class GrlsStorage(BaseStorage):
    """Usage::
        async with GrlsStorage() as storage:
            hits = await storage.search_by_trade_name("амоксиклав")
    """

    async def replace_all(self, records: list[GrlsRecord], imp: GrlsImport) -> int:
        """Full replacement in one transaction (DELETE, not TRUNCATE — readers are not blocked)."""
        inserted = 0
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM grls_registry")
                async with conn.cursor() as cur:
                    for i in range(0, len(records), _BATCH):
                        batch = records[i:i + _BATCH]
                        await cur.executemany(_INSERT_SQL, [_record_params(r) for r in batch])
                cur2 = await conn.execute("SELECT count(*) AS n FROM grls_registry")
                inserted = (await cur2.fetchone())["n"]
                await conn.execute(
                    """
                    INSERT INTO grls_imports (archive_name, registry_date, status_counts, skipped_files)
                    VALUES (%(archive_name)s, %(registry_date)s, %(status_counts)s, %(skipped_files)s)
                    """,
                    {"archive_name": imp.archive_name, "registry_date": imp.registry_date,
                     "status_counts": Jsonb(imp.status_counts), "skipped_files": Jsonb(imp.skipped_files)},
                )
        return inserted

    async def search_by_trade_name(self, query: str, *, threshold: float = 0.85, limit: int = 6,
                                   include_substances: bool = False) -> list[GrlsRecord]:
        q = normalize_query(query)
        if not q:
            return []
        stmt = sql.SQL(
            "SELECT " + _SELECT_COLS + ", similarity(grls_norm(trade_name), %(q)s) AS sim "
            "FROM grls_registry "
            "WHERE grls_norm(trade_name) %% %(q)s "
            "  AND similarity(grls_norm(trade_name), %(q)s) >= %(threshold)s "
            "  AND (%(inc)s OR NOT is_substance) {} LIMIT %(limit)s"
        ).format(_ORDER)
        async with self._pool.connection() as conn:
            cur = await conn.execute(stmt, {"q": q, "threshold": threshold, "inc": include_substances, "limit": limit})
            rows = await cur.fetchall()
        return [_row_to_record(r) for r in rows]

    async def search_by_inn(self, query: str, *, limit: int = 20,
                            include_substances: bool = False) -> list[GrlsRecord]:
        q = normalize_query(query)
        if not q:
            return []
        stmt = sql.SQL(
            "SELECT " + _SELECT_COLS + ", similarity(grls_norm(inn_name), %(q)s) AS sim "
            "FROM grls_registry "
            "WHERE (grls_norm(inn_name) = %(q)s "
            "       OR (grls_norm(inn_name) %% %(q)s AND similarity(grls_norm(inn_name), %(q)s) >= %(fuzzy)s)) "
            "  AND (%(inc)s OR NOT is_substance) {} LIMIT %(limit)s"
        ).format(_ORDER)
        async with self._pool.connection() as conn:
            cur = await conn.execute(stmt, {"q": q, "fuzzy": _INN_FUZZY_THRESHOLD, "inc": include_substances, "limit": limit})
            rows = await cur.fetchall()
        return [_row_to_record(r) for r in rows]

    async def inn_status_counts(self, query: str, *, include_substances: bool = False) -> dict[str, int]:
        q = normalize_query(query)
        if not q:
            return {}
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT status, count(*) AS n FROM grls_registry
                WHERE (grls_norm(inn_name) = %(q)s
                       OR (grls_norm(inn_name) %% %(q)s AND similarity(grls_norm(inn_name), %(q)s) >= %(fuzzy)s))
                  AND (%(inc)s OR NOT is_substance)
                GROUP BY status
                """,
                {"q": q, "fuzzy": _INN_FUZZY_THRESHOLD, "inc": include_substances},
            )
            rows = await cur.fetchall()
        return {r["status"]: r["n"] for r in rows}

    async def latest_import(self) -> GrlsImport | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id, archive_name, registry_date, status_counts, skipped_files, imported_at "
                "FROM grls_imports ORDER BY id DESC LIMIT 1")
            row = await cur.fetchone()
        if not row:
            return None
        return GrlsImport(id=row["id"], archive_name=row["archive_name"], registry_date=row["registry_date"],
                          status_counts=row["status_counts"], skipped_files=row["skipped_files"],
                          imported_at=row["imported_at"])

    async def count(self) -> int:
        async with self._pool.connection() as conn:
            cur = await conn.execute("SELECT count(*) AS n FROM grls_registry")
            return (await cur.fetchone())["n"]
```

Примечание: в `_row_to_record` psycopg отдаёт `TEXT[]` как `list[str]`, `DATE` как `date`, `JSONB` как `dict/list` — конвертации не нужны. `%%` — экранирование `%` для psycopg-параметров; в `sql.SQL(...)` тоже нужно `%%`, т.к. итоговая строка идёт через клиентскую подстановку.

- [ ] **Step 3: Проверить импорт модуля без БД**

Run: `python -c "import sys; sys.path.insert(0,'src'); import storage.grls_storage as m; print(m._INSERT_SQL[:60]); print(m._ORDER.as_string(None)[:80])"`
Expected: печатает начало INSERT и `ORDER BY CASE status WHEN 'Действующий' THEN 0 …` без исключений. (`as_string(None)` у `psycopg.sql` работает без соединения для литералов; если версия psycopg требует контекст — пропустить вторую печать.)

- [ ] **Step 4: Коммит**

```bash
git add src/storage/grls_storage.py tests/test_grls_storage.py
git commit -m "feat(grls): GrlsStorage — replace_all, normalized trgm searches, latest_import"
```

---

### Task 7: CLI `scripts/import-grls.py`

**Files:**
- Create: `scripts/import-grls.py`
- Test: `tests/test_import_grls_script.py`

**Interfaces:**
- Consumes: `grls.parser.read_archive`, `grls.dump.write_dump`, `storage.grls_storage.GrlsStorage`, `GrlsImport`.
- Produces: `main(argv: list[str] | None = None) -> int`; чистая `plan_import(results: list[SheetResult]) -> ImportPlan` (dataclass: `records`, `status_counts`, `skipped_files`, `registry_date`, `duplicates_dropped`).

- [ ] **Step 1: Тест (без БД — dry-run и дамп)**

```python
# tests/test_import_grls_script.py
import importlib.util
import zipfile
from datetime import date
from pathlib import Path

from grls import status as st
from grls.dump import read_dump
from grls_fixtures import make_sheet, sample_row

_spec = importlib.util.spec_from_file_location(
    "import_grls", Path(__file__).resolve().parent.parent / "scripts" / "import-grls.py")
imp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(imp)


def _archive(tmp_path: Path) -> Path:
    d = tmp_path / "x"
    d.mkdir()
    dup = sample_row()
    make_sheet(d / "1.xlsx", st.STATUS_ACTIVE, [dup, dup, sample_row(reg_number="ЛП-000002")])
    make_sheet(d / "2.xlsx", st.STATUS_CHANGED, [sample_row()])
    make_sheet(d / "3.xlsx", st.STATUS_EXPIRED, [sample_row(reg_number="ЛП-000003", expires_at="31.12.2025")])
    z = tmp_path / "grls2026-08-17-1.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in sorted(d.iterdir()):
            zf.write(f, arcname=f.name)
    return z


def test_plan_import_dedups_and_counts(tmp_path: Path):
    from grls.parser import read_archive
    plan = imp.plan_import(read_archive(_archive(tmp_path)))
    assert plan.registry_date == date(2026, 8, 17)
    assert plan.status_counts == {st.STATUS_ACTIVE: 2, st.STATUS_EXPIRED: 1}
    assert plan.duplicates_dropped == 1
    assert plan.skipped_files and "2.xlsx" in plan.skipped_files[0]
    assert len(plan.records) == 3


def test_dry_run_prints_summary_and_writes_nothing(tmp_path: Path, capsys):
    rc = imp.main([str(_archive(tmp_path)), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "registry_date: 2026-08-17" in out
    assert f"{st.STATUS_ACTIVE}: 2" in out
    assert "dry-run" in out


def test_make_dump_without_db(tmp_path: Path):
    dump = tmp_path / "grls.jsonl.gz"
    rc = imp.main([str(_archive(tmp_path)), "--dry-run", "--make-dump", str(dump)])
    assert rc == 0
    meta, records = read_dump(dump)
    assert meta["row_count"] == 3
    assert meta["archive_name"] == "grls2026-08-17-1.zip"
    assert len(records) == 3
```

- [ ] **Step 2: Запустить — должен упасть**

Run: `pytest tests/test_import_grls_script.py -q`
Expected: FAIL (`FileNotFoundError` на скрипте).

- [ ] **Step 3: Реализация**

```python
#!/usr/bin/env python3
"""Import a GRLS xlsx export (zip or directory) into grls_registry.

Usage:
    python scripts/import-grls.py <archive.zip | dir-with-xlsx> [--dry-run] [--make-dump FILE]

--dry-run        parse, dedup, print counts; do not touch the DB
--make-dump FILE write JSONL(.gz) dump for engine sync (spec §7); works with --dry-run
Full replacement of grls_registry in one transaction; idempotent.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from grls.dump import write_dump  # noqa: E402
from grls.parser import SheetResult, read_archive  # noqa: E402
from storage.models.grls_record import GrlsImport, GrlsRecord  # noqa: E402

logger = logging.getLogger("import-grls")


@dataclass
class ImportPlan:
    registry_date: date
    records: list[GrlsRecord]
    status_counts: dict[str, int]
    skipped_files: list[str] = field(default_factory=list)
    duplicates_dropped: int = 0


def plan_import(results: list[SheetResult]) -> ImportPlan:
    """Merge sheets, drop exact duplicates by row_hash, count per status."""
    if not results:
        raise SystemExit("no xlsx sheets found")
    dates = {r.registry_date for r in results}
    if len(dates) > 1:
        raise SystemExit(f"sheets carry different registry dates: {sorted(d.isoformat() for d in dates)}")
    seen: set[str] = set()
    records: list[GrlsRecord] = []
    dropped = 0
    skipped: list[str] = []
    for res in results:
        if res.skipped:
            skipped.append(res.source_name)
            continue
        for rec in res.records:
            if rec.row_hash in seen:
                dropped += 1
                continue
            seen.add(rec.row_hash)
            records.append(rec)
    counts = Counter(r.status for r in records)
    return ImportPlan(registry_date=dates.pop(), records=records, status_counts=dict(counts),
                      skipped_files=skipped, duplicates_dropped=dropped)


def _print_summary(plan: ImportPlan, archive_name: str, dry_run: bool) -> None:
    print(f"archive: {archive_name}")
    print(f"registry_date: {plan.registry_date.isoformat()}")
    for status, n in sorted(plan.status_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {status}: {n}")
    print(f"rows: {len(plan.records)} (exact duplicates dropped: {plan.duplicates_dropped})")
    print(f"skipped files: {plan.skipped_files}")
    if dry_run:
        print("dry-run: database not touched")


async def _write(plan: ImportPlan, archive_name: str) -> int:
    from storage.grls_storage import GrlsStorage  # DB deps only when writing

    imp = GrlsImport(archive_name=archive_name, registry_date=plan.registry_date,
                     status_counts=plan.status_counts, skipped_files=plan.skipped_files)
    async with GrlsStorage() as storage:
        return await storage.replace_all(plan.records, imp)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="zip archive or directory with xlsx")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--make-dump", type=Path, metavar="FILE")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    plan = plan_import(read_archive(args.source))
    archive_name = args.source.name
    _print_summary(plan, archive_name, args.dry_run)
    if args.make_dump:
        n = write_dump(args.make_dump, plan.records, registry_date=plan.registry_date, archive_name=archive_name)
        print(f"dump: {args.make_dump} ({n} rows)")
    if args.dry_run:
        return 0
    inserted = asyncio.run(_write(plan, archive_name))
    print(f"inserted: {inserted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`chmod +x scripts/import-grls.py`.

- [ ] **Step 4: Запустить тесты**

Run: `pytest tests/test_import_grls_script.py -q`
Expected: 3 passed.

- [ ] **Step 5: Dry-run на реальном архиве**

Run: `python scripts/import-grls.py /home/savoy/projects/grls2026-08-17-1.zip --dry-run`
Expected: `registry_date: 2026-08-17`, 7 статусов со счётчиками ≈ спека §2, `skipped files: ['grls2026-08-17-1-Изменённый.xlsx']` (имя декодировано из cp437), `dry-run: database not touched`. Записать реальные счётчики — они пойдут в `docs/revision-log.md` (Task 10).

- [ ] **Step 6: Коммит**

```bash
git add scripts/import-grls.py tests/test_import_grls_script.py
git commit -m "feat(grls): import-grls.py — parse, dedup, dry-run, dump, full replace"
```

---

### Task 8: Форматирование справки (`src/grls/format.py`) и `lookup_medicine` (`src/grls/lookup.py`)

**Files:**
- Create: `src/grls/format.py`, `src/grls/lookup.py`
- Test: `tests/test_grls_format.py`

**Interfaces:**
- Consumes: `GrlsRecord`, `DietarySupplement` (`storage/models/dietary_supplement.py`), `status_at`, `LIVE_STATUSES`, `GrlsStorage`, `DietarySupplementsStorage`.
- Produces:
  ```python
  # src/grls/format.py
  @dataclass class MedicineLookup: query: str; on: date | None; registry_date: date | None; inn_records: list[GrlsRecord]; inn_counts: dict[str,int]; trade_records: list[GrlsRecord]; supplements: list[DietarySupplement]
  MAX_TRADE_RECORDS = 6; MAX_LIST_ITEMS = 5; TRADE_THRESHOLD = 0.85
  def status_line(record: GrlsRecord, on: date | None) -> str
  def format_record(record: GrlsRecord, on: date | None) -> str
  def format_medicine_lookup(lookup: MedicineLookup) -> str
  NOT_FOUND = "Препарат или БАД не найден в реестрах."
  # src/grls/lookup.py
  async def lookup_medicine(query: str, *, on: date | None = None) -> MedicineLookup
  ```
  Граф (другая ветка) использует `lookup_medicine` + `status_line` для однострочной справки.

- [ ] **Step 1: Тесты форматирования**

```python
# tests/test_grls_format.py
from datetime import date

from grls import status as st
from grls.format import (NOT_FOUND, MedicineLookup, format_medicine_lookup, format_record,
                         status_line)
from grls.parser import build_record
from grls_fixtures import sample_row
from storage.models.dietary_supplement import DietarySupplement

VISIT = date(2025, 3, 10)


def _rec(status, **over):
    return build_record(status, sample_row(**over))


def _lookup(**over):
    base = dict(query="амоксиклав", on=None, registry_date=date(2026, 8, 17),
                inn_records=[], inn_counts={}, trade_records=[], supplements=[])
    base.update(over)
    return MedicineLookup(**base)


def test_status_line_active_termless_and_termed():
    assert status_line(_rec(st.STATUS_ACTIVE), None) == "Действующий (РУ ЛП-000001, бессрочно)"
    assert status_line(_rec(st.STATUS_EAEU, expires_at="01.03.2027"), None) == \
        "Выдано по правилам ЕАЭС (РУ ЛП-000001, действует до 2027-03-01)"


def test_status_line_notes():
    assert status_line(_rec(st.STATUS_SUSPENDED), None) == \
        "Действующий, приостановлено применение (предупреждение, не запрет назначения) (РУ ЛП-000001)"
    assert status_line(_rec(st.STATUS_CONFIRMING, expires_at="17.10.2021"), None) == \
        "Действующий, на подтверждении регистрации (РУ ЛП-000001, срок до 2021-10-17)"
    assert status_line(_rec(st.STATUS_FOREIGN_PACK), None) == \
        "Действующий, в иностранной упаковке (РУ ЛП-000001)"


def test_status_line_dead_and_softened():
    r = _rec(st.STATUS_EXPIRED, expires_at="31.12.2025")
    assert status_line(r, None) == "Истёкший (истекло 2025-12-31; РУ ЛП-000001)"
    assert status_line(r, VISIT) == "Истёкший (истекло 2025-12-31; на дату визита 2025-03-10 действовало; РУ ЛП-000001)"
    a = _rec(st.STATUS_ANNULLED, annulled_at="14.02.2024")
    assert status_line(a, VISIT) == "Исключённый (аннулировано 2024-02-14; РУ ЛП-000001)"
    assert status_line(_rec(st.STATUS_ANNULLED), VISIT) == "Исключённый (дата неизвестна; РУ ЛП-000001)"


def test_format_record_uses_derived_forms_and_caps():
    r = _rec(st.STATUS_ACTIVE, forms_raw="; ".join(
        f"форма{i}, 5 мг - блистеры - По рецепту" for i in range(7)) + "; мазь, 1% - тубы - для стационаров;")
    text = format_record(r, None)
    assert "Торговое наименование: Тестин®" in text
    assert "МНН: тестамол" in text
    assert "Лекарственные формы: форма0; форма1; форма2; форма3; форма4 (+ ещё 3)" in text
    assert "Отпуск: По рецепту; для стационаров" in text
    assert "ЖНВЛП: да" in text
    assert "Формы выпуска:" not in text


def test_inn_branch_counts_and_examples():
    recs = [_rec(st.STATUS_ACTIVE, trade_name="Амоксиклав"), _rec(st.STATUS_EXPIRED, trade_name="Аугментин")]
    text = format_medicine_lookup(_lookup(query="амоксициллин+клавулановая кислота", inn_records=recs,
                                          inn_counts={st.STATUS_ACTIVE: 12, st.STATUS_EXPIRED: 3}))
    assert text.startswith("В ГРЛС «амоксициллин+клавулановая кислота» — это МНН.")
    assert "Регистраций: 15, из них действующих: 12" in text
    assert "Амоксиклав" in text and "Аугментин" in text
    assert "реестр от 2026-08-17" in text
    assert "внимание" not in text.lower()


def test_inn_branch_warns_when_nothing_live():
    text = format_medicine_lookup(_lookup(inn_records=[_rec(st.STATUS_EXPIRED)],
                                          inn_counts={st.STATUS_EXPIRED: 2}))
    assert "Внимание: все РУ по этому МНН истекли или аннулированы." in text


def test_trade_branch_header_and_blocks():
    text = format_medicine_lookup(_lookup(trade_records=[_rec(st.STATUS_ACTIVE), _rec(st.STATUS_EXPIRED, expires_at="31.12.2025")]))
    assert text.startswith("Найдено в ГРЛС (2; реестр от 2026-08-17):")
    assert "--- 1 ---" in text and "--- 2 ---" in text
    assert "Статус РУ: Действующий (РУ ЛП-000001, бессрочно)" in text


def test_supplement_and_not_found():
    s = DietarySupplement(product_name="Бак-Сет", registration_number="RU.77.99.11.003.Е.000001",
                          status="действует")
    text = format_medicine_lookup(_lookup(supplements=[s]))
    assert "Найдено как БАД" in text and "Бак-Сет" in text
    assert format_medicine_lookup(_lookup()) == NOT_FOUND
```

Проверить сигнатуру `DietarySupplement` в `src/storage/models/dietary_supplement.py` (поля `product_name`, `registration_number`, `status`, `label_info`…); при расхождении подправить конструктор в тесте, не модель.

- [ ] **Step 2: Запустить — должен упасть**

Run: `pytest tests/test_grls_format.py -q`
Expected: FAIL (`ModuleNotFoundError: grls.format`).

- [ ] **Step 3: Реализация `format.py`**

```python
# src/grls/format.py
"""Human/LLM-readable rendering of a GRLS lookup (shared by the tool and the graph node)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from grls.status import (LIVE_STATUSES, STATUS_ANNULLED, STATUS_CONFIRMING, STATUS_EXPIRED,
                         STATUS_FOREIGN_PACK, STATUS_SUSPENDED, StatusAtVisit, status_at)
from storage.models.dietary_supplement import DietarySupplement
from storage.models.grls_record import GrlsRecord

MAX_TRADE_RECORDS = 6
MAX_LIST_ITEMS = 5
TRADE_THRESHOLD = 0.85
NOT_FOUND = "Препарат или БАД не найден в реестрах."

_NOTES = {
    STATUS_SUSPENDED: "приостановлено применение (предупреждение, не запрет назначения)",
    STATUS_CONFIRMING: "на подтверждении регистрации",
    STATUS_FOREIGN_PACK: "в иностранной упаковке",
}


@dataclass
class MedicineLookup:
    query: str
    on: date | None
    registry_date: date | None
    inn_records: list[GrlsRecord] = field(default_factory=list)
    inn_counts: dict[str, int] = field(default_factory=dict)
    trade_records: list[GrlsRecord] = field(default_factory=list)
    supplements: list[DietarySupplement] = field(default_factory=list)


def _iso(d: date | None) -> str:
    return d.isoformat() if d else ""


def status_line(record: GrlsRecord, on: date | None) -> str:
    sv = status_at(record, on)
    ru = f"РУ {record.reg_number}"
    if sv is StatusAtVisit.ACTIVE:
        term = f"действует до {_iso(record.expires_at)}" if record.expires_at else "бессрочно"
        return f"{record.status} ({ru}, {term})"
    if sv is StatusAtVisit.ACTIVE_WITH_NOTE:
        term = f", срок до {_iso(record.expires_at)}" if record.expires_at else ""
        return f"Действующий, {_NOTES[record.status]} ({ru}{term})"
    if sv is StatusAtVisit.UNKNOWN_END:
        return f"{record.status} (дата неизвестна; {ru})"
    if record.status == STATUS_EXPIRED:
        event = f"истекло {_iso(record.expires_at)}"
    else:  # STATUS_ANNULLED
        event = f"аннулировано {_iso(record.annulled_at or record.expires_at)}"
    if sv is StatusAtVisit.VALID_AT_VISIT:
        return f"{record.status} ({event}; на дату визита {_iso(on)} действовало; {ru})"
    return f"{record.status} ({event}; {ru})"


def _join_capped(items: list[str]) -> str:
    head = "; ".join(items[:MAX_LIST_ITEMS])
    rest = len(items) - MAX_LIST_ITEMS
    return f"{head} (+ ещё {rest})" if rest > 0 else head


def format_record(record: GrlsRecord, on: date | None) -> str:
    parts = [f"Торговое наименование: {record.trade_name}"]
    if record.inn_name:
        parts.append(f"МНН: {record.inn_name}")
    parts.append(f"Статус РУ: {status_line(record, on)}")
    if record.dosage_forms:
        parts.append(f"Лекарственные формы: {_join_capped(record.dosage_forms)}")
    if record.dispensing:
        parts.append(f"Отпуск: {_join_capped(record.dispensing)}")
    if record.pharm_group:
        parts.append(f"ФТГ: {record.pharm_group}")
    if record.is_vital is not None:
        parts.append(f"ЖНВЛП: {'да' if record.is_vital else 'нет'}")
    if record.narcotic_list:
        parts.append(f"ПКУ: {record.narcotic_list}")
    return "\n".join(parts)


def _format_supplement(s: DietarySupplement) -> str:
    parts = [f"Наименование: {s.product_name}"]
    if s.registration_number:
        parts.append(f"Свидетельство: {s.registration_number}")
    if s.status:
        parts.append(f"Статус: {s.status}")
    if s.label_info:
        parts.append(f"Информация на этикетке: {s.label_info}")
    return "\n".join(parts)


def _registry_note(lookup: MedicineLookup) -> str:
    return f"реестр от {_iso(lookup.registry_date)}" if lookup.registry_date else "дата реестра неизвестна"


def format_medicine_lookup(lookup: MedicineLookup) -> str:
    if lookup.inn_records:
        total = sum(lookup.inn_counts.values()) or len(lookup.inn_records)
        live = sum(n for s, n in lookup.inn_counts.items() if s in LIVE_STATUSES)
        names: list[str] = []
        for r in lookup.inn_records:
            if r.trade_name not in names:
                names.append(r.trade_name)
        lines = [f"В ГРЛС «{lookup.query}» — это МНН. "
                 f"Регистраций: {total}, из них действующих: {live}. "
                 f"Примеры торговых наименований: {', '.join(names[:MAX_LIST_ITEMS])} "
                 f"({_registry_note(lookup)})."]
        if live == 0:
            lines.append("Внимание: все РУ по этому МНН истекли или аннулированы.")
        return "\n".join(lines)
    if lookup.trade_records:
        recs = lookup.trade_records[:MAX_TRADE_RECORDS]
        lines = [f"Найдено в ГРЛС ({len(recs)}; {_registry_note(lookup)}):\n"]
        lines += [f"--- {i} ---\n{format_record(r, lookup.on)}" for i, r in enumerate(recs, 1)]
        return "\n\n".join(lines)
    if lookup.supplements:
        lines = [f"Найдено как БАД в Едином реестре свидетельств о государственной регистрации ({len(lookup.supplements)}):\n"]
        lines += [f"--- {i} ---\n{_format_supplement(s)}" for i, s in enumerate(lookup.supplements, 1)]
        return "\n\n".join(lines)
    return NOT_FOUND
```

- [ ] **Step 4: Реализация `lookup.py`**

```python
# src/grls/lookup.py
"""Open the registries and assemble a MedicineLookup (I/O lives here, formatting in format.py)."""
from __future__ import annotations

import logging
from datetime import date

from grls.format import TRADE_THRESHOLD, MedicineLookup
from storage.dietary_supplements_storage import DietarySupplementsStorage
from storage.grls_storage import GrlsStorage

logger = logging.getLogger(__name__)


async def lookup_medicine(query: str, *, on: date | None = None) -> MedicineLookup:
    """INN first, then trade name, then dietary supplements — same order as the old tool."""
    async with GrlsStorage() as grls:
        imp = await grls.latest_import()
        inn_records = await grls.search_by_inn(query)
        inn_counts = await grls.inn_status_counts(query) if inn_records else {}
        trade_records = [] if inn_records else await grls.search_by_trade_name(query, threshold=TRADE_THRESHOLD)
    supplements = []
    if not inn_records and not trade_records:
        async with DietarySupplementsStorage() as supps:
            supplements = await supps.search(query)
    logger.info('💊 GRLS lookup "%s": inn=%d trade=%d supplements=%d',
                query, len(inn_records), len(trade_records), len(supplements))
    return MedicineLookup(query=query, on=on, registry_date=imp.registry_date if imp else None,
                          inn_records=inn_records, inn_counts=inn_counts,
                          trade_records=trade_records, supplements=supplements)
```

- [ ] **Step 5: Запустить тесты**

Run: `pytest tests/test_grls_format.py -q`
Expected: 8 passed.

- [ ] **Step 6: Коммит**

```bash
git add src/grls/format.py src/grls/lookup.py tests/test_grls_format.py
git commit -m "feat(grls): format_medicine_lookup with status_at lines; lookup_medicine"
```

---

### Task 9: `SearchMedicineTool` на ГРЛС, промпт, удаление `drugs`

**Files:**
- Modify: `src/LLM/tools.py` (импорты строки 38/41, блок «Drug lookup helpers» строки 78–101, класс `SearchMedicineTool` строки 183–247)
- Modify: `src/LLM/prompts/treatment_checker.txt`
- Modify: `src/storage/models/__init__.py`, `scripts/seed-reference-lists.sh`
- Delete: `src/storage/drugs_storage.py`, `src/storage/models/drug.py`, `resources/Drugs list.csv`
- Test: `tests/test_search_medicine_tool.py`

**Interfaces:**
- Consumes: `grls.lookup.lookup_medicine`, `grls.format.format_medicine_lookup`.
- Produces: тул `search_medicine` с прежними именем/`args_schema`/местом подключения (`get_tools_for`, `get_treatment_tools_for`).

- [ ] **Step 1: Тест тула с подменой `lookup_medicine`**

```python
# tests/test_search_medicine_tool.py
from datetime import date

import pytest

from grls import status as st
from grls.format import MedicineLookup
from grls.parser import build_record
from grls_fixtures import sample_row
import LLM.tools as tools


async def test_search_medicine_tool_delegates_to_lookup(monkeypatch):
    seen = {}

    async def fake_lookup(query, *, on=None):
        seen["query"], seen["on"] = query, on
        return MedicineLookup(query=query, on=on, registry_date=date(2026, 8, 17),
                              trade_records=[build_record(st.STATUS_ACTIVE, sample_row())])

    monkeypatch.setattr(tools, "lookup_medicine", fake_lookup)
    text = await tools.SearchMedicineTool()._arun("тестин")
    assert seen == {"query": "тестин", "on": None}
    assert text.startswith("Найдено в ГРЛС (1; реестр от 2026-08-17):")
    assert "Статус РУ: Действующий (РУ ЛП-000001, бессрочно)" in text


def test_tool_description_mentions_grls_and_status():
    d = tools.SearchMedicineTool().description
    assert "ГРЛС" in d and "статус" in d.lower()
    assert "ЕСКЛП" not in d


def test_drugs_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        import storage.drugs_storage  # noqa: F401
    import storage.models as m
    assert not hasattr(m, "Drug")
```

- [ ] **Step 2: Запустить — должен упасть**

Run: `pytest tests/test_search_medicine_tool.py -q`
Expected: FAIL (`AttributeError: module 'LLM.tools' has no attribute 'lookup_medicine'`, `Drug` ещё есть).

- [ ] **Step 3: Переписать тул**

В `src/LLM/tools.py`:

1. Удалить `from storage.drugs_storage import DrugsStorage` и `from storage.models.drug import Drug`; удалить `_DRUG_SCORE_THRESHOLD`, `_format_drug`, `_format_supplement` (форматирование БАД теперь в `grls/format.py`); если `DietarySupplementsStorage`/`DietarySupplement` больше нигде в файле не используются — удалить и эти импорты (проверить `grep -n "DietarySupplement" src/LLM/tools.py`).
2. Добавить импорты:
   ```python
   from grls.format import format_medicine_lookup
   from grls.lookup import lookup_medicine
   ```
3. Класс:
   ```python
   class SearchMedicineTool(BaseTool):
       """Look up a drug (GRLS, with certificate status) or a dietary supplement by name."""

       name: str = "search_medicine"
       description: str = (
           "Поиск препарата в ГРЛС (Государственный реестр лекарственных средств) по торговому "
           "названию или МНН, с фолбэком в реестр БАД. Передавай только название, без лишних слов. "
           "Ответ содержит статус регистрационного удостоверения (действующее / истёкшее / "
           "аннулированное / с предупреждением), лекарственные формы и условия отпуска. "
           "Используй, чтобы связать торговое название с действующим веществом и проверить, "
           "что назначенный препарат зарегистрирован."
       )
       args_schema: Type[BaseModel] = _QueryInput

       async def _arun(self, query: str) -> str:  # type: ignore[override]
           return format_medicine_lookup(await lookup_medicine(query))

       def _run(self, query: str) -> str:  # type: ignore[override]
           raise NotImplementedError("Use async invocation (_arun).")
   ```

- [ ] **Step 4: Промпт `treatment_checker.txt`**

В разделе «Инструменты» строку про `search_medicine` заменить на:

```
- ``search_medicine`` — справка по препарату из ГРЛС (Государственный реестр лекарственных средств): МНН, статус регистрационного удостоверения, формы выпуска, условия отпуска; при отсутствии в ГРЛС — поиск в реестре БАД.
Обязателен к применению для оценки предписаний врача, т.к. врачи часто пишут торговые названия лекарств,
а клинические рекомендации — только действующие вещества. Инструмент позволяет связать их между собой.
Инструмент пишет, откуда берётся информация о лекарствах (ГРЛС, дата реестра). Указывай это тоже как source.
```

В раздел «Обрати внимание» добавить пункты 5–6:

```
5. Статус РУ из справки трактуй так: «Истёкший»/«Исключённый» без пометки «на дату визита действовало» — назначенный препарат не имел действующей регистрации, это замечание; «на дату визита действовало» — не замечание; «приостановлено применение», «на подтверждении регистрации», «в иностранной упаковке» — препарат легален, можно упомянуть как предупреждение, но не как нарушение.
6. «Препарат или БАД не найден в реестрах» — повод перепроверить написание другим запросом (МНН вместо торгового названия), а не утверждать, что препарат не зарегистрирован.
```

- [ ] **Step 5: Удалить старое**

```bash
git rm src/storage/drugs_storage.py src/storage/models/drug.py "resources/Drugs list.csv"
```

`src/storage/models/__init__.py`: убрать `from .drug import Drug` и `"Drug"` из `__all__`.

`scripts/seed-reference-lists.sh`: удалить всё, что относится к drugs — переменную `DRUGS_CSV`, её проверку в цикле `for f in …` (оставить проверку только `SUPPLEMENTS_CSV`), `DRUGS_SQL`/`mktemp`/`trap` (оставить `SUPPS_SQL`), блок «── drugs ──» до `psql_cmd -f "$DRUGS_SQL"` включительно, и в хвостовом «Rows loaded» — строку по `drugs`, если она там есть (посмотреть `sed -n 140,160p`). Обновить шапку-комментарий: «Truncates and reloads dietary_supplements from CSV. Drugs now come from GRLS: scripts/import-grls.py».

Проверить `bash -n scripts/seed-reference-lists.sh` (синтаксис).

- [ ] **Step 6: Запустить тесты и грепнуть хвосты**

Run: `pytest tests/test_search_medicine_tool.py tests/test_grls_*.py tests/test_migration_027.py tests/test_import_grls_script.py -q`
Expected: все passed.

Run: `grep -rn "DrugsStorage\|models.drug\|Drugs list\|ЕСКЛП" src scripts tests --include=*.py --include=*.sh --include=*.txt`
Expected: пусто (кроме, возможно, комментария-истории в seed-скрипте).

- [ ] **Step 7: Коммит**

```bash
git add -A src/LLM/tools.py src/LLM/prompts/treatment_checker.txt src/storage/models/__init__.py scripts/seed-reference-lists.sh tests/test_search_medicine_tool.py
git commit -m "feat(grls): search_medicine on GRLS with certificate status; drop drugs/ЕСКЛП"
```

---

### Task 10: Документация, журнал, хозяйство, стендовый гейт

**Files:**
- Create: `docs/grls.md`
- Modify: `docs/revision-log.md`, `CLAUDE.md` (строка дерева `drugs_storage.py`), `.gitignore`, `docs/rag.md` (если упоминает `search_medicine`/ЕСКЛП — `grep -n "search_medicine\|ЕСКЛП" docs/*.md`)

- [ ] **Step 1: `docs/grls.md`**

```markdown
# ГРЛС — реестр лекарственных средств со статусами РУ

Источник: выгрузка Государственного реестра лекарственных средств
(grls.rosminzdrav.ru → «Выгрузка реестра», xlsx-архив `grls<YYYY-MM-DD>-1.zip`,
~19 МБ, 8 файлов — по одному на состояние записи). Архив в git не кладём.

## Что в БД

- `grls_registry` — одна строка = одна строка выгрузки; 7 статусов (файл
  «Изменённый» — журнал редакций — не грузится). Номер РУ **не уникален**;
  ключ дедупликации/синка — `row_hash`. Смысл колонок — `COMMENT ON COLUMN`
  в `migrations/027_grls_registry.sql`.
- `grls_imports` — журнал загрузок; последняя строка = версия реестра
  (`registry_date` = «по состоянию на …» из выгрузки).
- Производные колонки: `dosage_forms`, `dispensing` (из «Формы выпуска»),
  `is_substance` (фармсубстанции `ФС-…`/«субстанция…» — не препараты; поиск
  их не показывает без `include_substances=True`).

## Импорт

    python scripts/import-grls.py grls2026-08-17-1.zip --dry-run     # счётчики без записи
    python scripts/import-grls.py grls2026-08-17-1.zip               # полная замена в одной транзакции
    python scripts/import-grls.py grls2026-08-17-1.zip --make-dump grls.jsonl.gz   # дамп для engine

Проверка версии: `SELECT registry_date, status_counts FROM grls_imports ORDER BY id DESC LIMIT 1;`

## Как читать статус

Статус берётся из файла выгрузки; даты его не переопределяют (реестр сам себе
противоречит: есть «Действующие» с датой окончания в прошлом). Даты используются
только чтобы смягчить мёртвый статус относительно даты визита
(`grls.status.status_at`): «Истёкший» с `expires_at ≥ дата визита` → «на дату
визита действовало» — не замечание. «Приостановлено применение», «на
подтверждении», «в иностранных упаковках» — предупреждения, не запрет.
Историю перерегистраций по снимку восстановить нельзя (старая запись затёрта).

## Поиск (`storage.grls_storage.GrlsStorage`, `grls.lookup.lookup_medicine`)

Хранение — как есть; нормализация в запросе (`grls.normalize.normalize_query`
↔ SQL `grls_norm()`: lower, без кавычек/®/™, ё→е). Порядок: МНН (точно или
trgm ≥ 0.6) → торговое название (trgm ≥ 0.85) → БАД. Результаты сортируются
по рангу статуса (живые → приостановлено → истёкший → исключённый).

## Синк в «Искру» (engine)

Контракт — спека `docs/superpowers/specs/2026-08-17-grls-registry-design.md`
§7: канон = `grls_registry`/`grls_imports`, ключ = `row_hash` (алгоритм
`grls.normalize.row_hash`), дамп = JSONL(.gz) с `_meta` первой строкой
(`grls.dump`). Реализация engine-стороны — отдельная ветка.
```

- [ ] **Step 2: Журнал, CLAUDE.md, .gitignore**

`docs/revision-log.md`, раздел «Лекарства (ГРЛС)»: строку «— | Первый импорт ГРЛС на стенде — дописать…» заменить на реальную запись **после** стендового импорта (Step 4): дата, `registry_date`, счётчики из `grls_imports.status_counts`, команда, коммит.

`CLAUDE.md`: строку `│   ├── drugs_storage.py     # Drug reference data` заменить на `│   ├── grls_storage.py      # GRLS registry (drug certificates + statuses), see docs/grls.md`; в разделе про скрипты (если есть перечень) добавить `scripts/import-grls.py`.

`.gitignore`: добавить строку `grls*.zip`.

`grep -n "search_medicine\|ЕСКЛП\|drugs" docs/rag.md docs/diagnosis_validator.md README.md` — где упоминается ЕСКЛП/`drugs`, заменить на ГРЛС/`grls_registry` со ссылкой на `docs/grls.md`.

- [ ] **Step 3: Коммит доков**

```bash
git add docs/grls.md docs/revision-log.md CLAUDE.md .gitignore docs/rag.md docs/diagnosis_validator.md README.md
git commit -m "docs(grls): docs/grls.md, revision log, CLAUDE.md tree, ignore archives"
```

- [ ] **Step 4: Стендовый гейт (DoD спеки §11) — вручную на стенде**

1. `bash migrations/migrate.sh` → 027 применилась (в `schema_migrations` есть `027_grls_registry.sql`; `\d grls_registry` показывает 23 колонки; `SELECT grls_norm('"ЭФКУРИЯ®"')` → `эфкурия`; `\dt drugs` — таблицы нет).
2. `python scripts/import-grls.py <архив> --dry-run`, затем без `--dry-run`: `inserted:` ≈ 39 тыс.; `SELECT status, count(*) FROM grls_registry GROUP BY 1` совпадает со `status_counts` последней строки `grls_imports`; `skipped_files` содержит «Изменённый».
3. `pytest tests/test_grls_storage.py -q` (подменяет содержимое таблицы! после него — **повторить импорт** п.2).
4. `search_medicine` на 5 препаратах из реальных карт (в т.ч. с истёкшим РУ, с `®`/кавычками, одно МНН, у которого есть субстанция, например «амоксициллин») — статус есть, субстанций в выдаче нет; treatment-чекер прогнан на 3–5 кешированных картах (`scripts/audit-file.py` или как принято на стенде) — карты не broken, замечания про РУ появляются только у истёкших/аннулированных.
5. `EXPLAIN` для `search_by_trade_name` — используется `grls_registry_trade_name_trgm_idx` (Bitmap Index Scan); если Seq Scan — проверить, что функция `IMMUTABLE` и индекс создан.
6. Записать в `docs/revision-log.md` строку первого импорта (Step 2), закоммитить.

- [ ] **Step 5: Финальный прогон тестов без БД и коммит**

Run: `pytest tests/test_grls_*.py tests/test_migration_027.py tests/test_import_grls_script.py tests/test_search_medicine_tool.py -q`
Expected: все passed. Затем `git status` чистый; ветка `grls-registry` готова к ревью (не пушить без команды).

---

## Self-review (выполнено при написании)

**Покрытие спеки:** §2 источник/формат — Task 4 (парсер, маркер, дата из строки 3, хвостовая строка); §3 схема+комменты+индексы+`grls_norm`+drop drugs — Task 1; §4.1–4.3 чтение/нормализация/хеш — Tasks 3–4; §4.4 запись одной транзакцией — Task 6 (`replace_all`) + Task 7 (CLI, dry-run, дамп); §5 storage+`normalize_query`+фильтр субстанций — Task 6, §5.1 `status_at` — Task 2; §6 тул и текст справки, промпт — Tasks 8–9; §7 контракт синка/дамп — Task 5 + `--make-dump` в Task 7 + `docs/grls.md`; §8 тесты — по задачам, стендовые в Task 6/10; §9 доки/журнал — Task 10; §10 границы соблюдены (формы — только два производных массива, история не восстанавливается, БАДы не тронуты); §11 DoD — Task 10 Step 4.

**Отклонения от спеки, зафиксированы в ней же:** дата выгрузки — из строки 3, а не 6; запись — DELETE+INSERT в транзакции вместо staging-swap; докфайл `docs/grls.md` вместо отсутствующего в ветке `docs/storage.md`.

**Согласованность имён:** `GrlsRecord`/`GrlsImport` (Task 2) используются в 4–9; `status_at`/`StatusAtVisit` (2) — в 8; `normalize_query`/`row_hash` (3) — в 6, 5; `build_record`/`read_archive`/`SheetResult` (4) — в 5, 7; `write_dump`/`read_dump` (5) — в 7; `GrlsStorage.search_by_inn/search_by_trade_name/inn_status_counts/latest_import/replace_all` (6) — в 7, 8; `MedicineLookup`/`format_medicine_lookup`/`lookup_medicine` (8) — в 9.
