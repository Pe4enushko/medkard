# Таблица `guidelines` — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Перенести справочник клинреков из `resources/manifest.csv` в таблицу БД `guidelines`, связать её с `docs` внешним ключом, убрать дублирование колонок манифеста в `docs.metadata`, и перевести всех четырёх читателей CSV на БД.

**Architecture:** Новая таблица `guidelines` (PK = `file_id`, множественные поля `mkb`/`age_category` как `TEXT[]`) — единственный источник истины. Модель `Guideline` + `GuidelinesStorage` (psycopg3, паттерн `BaseStorage`). Матчинг МКБ уходит в SQL. Читаемые манифест-поля в `Doc` приходят через `LEFT JOIN docs → guidelines`. **`Guideline`-объекты передаются по всей цепочке как есть** — `clinic_recs`, `pipeline`, `icd_check._render_manifest_table`, `LLM.decider`, `LLM.icd_prefix_picker` все работают с `Guideline`, а не с dict-строками манифеста. Единственное место, где объект превращается в dict, — сериализация в JSON-промпт LLM внутри `decider`/`icd_prefix_picker` (через `dataclasses.asdict`), потому что это специфика LLM-вызова, а не контракт между модулями.

**Tech Stack:** Python 3, psycopg3 (async, `dict_row`), PostgreSQL, pgvector, pytest (`asyncio_mode=auto`, `pythonpath=src`), openpyxl.

## Global Constraints

- `pythonpath = src` (pytest.ini) — импорты вида `from storage.guidelines_storage import GuidelinesStorage`.
- Все SQL-миграции идемпотентны: `migrate.sh` гоняет **все** `[0-9]*.sql` при каждом прогоне с `ON_ERROR_STOP=1`. Использовать `IF NOT EXISTS` / guard на constraint.
- Порядок имён миграций критичен (лексикографическая сортировка): `019_guidelines` → `020_docs_metadata_cleanup` → `021_docs_guideline_fk`.
- **На машине разработки нет доступа к БД и нет `pip install`.** DB-миграции, seed, бэкфилл и storage-тесты против БД прогоняются **только на стенде**. Локально прогоняемы юнит-тесты чистых функций.
- Два бэкенда БД сосуществуют: `psycopg3` (storage/) и `asyncpg` (RAG/retrieval/). Новый код — в `storage/`, psycopg3.
- Возрастная семантика: только `Дети` → пациент-ребёнок (age ≤ 15); только `Взрослые` → взрослый; `Взрослые, дети` или пусто → пропускаем. Регистронезависимо.
- `age_category` хранится **дословно как в CSV** (`Взрослые`, `Дети`).
- **Никаких промежуточных dict-адаптеров манифеста.** Всюду, где раньше передавался `dict[str, str]` («строка манифеста»), теперь передаётся `Guideline`. Dict возникает только на самой границе с LLM-клиентом (JSON-сериализация промпта), не раньше.
- Коммитить часто, каждый таск — отдельный самодостаточный коммит на ветке `guidelines-table`.

---

## Файловая структура

Новые:
- `migrations/019_guidelines.sql` — DDL таблицы + GIN-индекс
- `migrations/020_docs_metadata_cleanup.sql` — удаление манифест-ключей из `docs.metadata`
- `migrations/021_docs_guideline_fk.sql` — FK `docs.file_id → guidelines.file_id` + проверка сирот
- `src/storage/models/guideline.py` — dataclass `Guideline` + `from_manifest_row`
- `src/storage/guidelines_storage.py` — `GuidelinesStorage`
- `scripts/seed-guidelines.py` — заливка `manifest.csv` → таблица
- `tests/test_guideline_model.py` — юнит-тесты `Guideline` (локальные, без БД)
- `tests/test_guidelines_storage.py` — интеграционные тесты storage (стенд)
- `tests/test_doc_format_chunk.py` — регрессия шапки чанка (локальный)
- `tests/test_clinic_recs_age.py` — юнит `_is_age_eligible` (локальный)
- `tests/test_report_meta.py` — юнит `build_manifest_meta` (локальный)
- `tests/test_decider_guideline_input.py` — юнит: `decider`/`icd_prefix_picker` принимают `Guideline` (локальный)

Изменяемые:
- `src/storage/models/__init__.py` — экспорт `Guideline`
- `src/storage/models/doc.py` — поля `name/mkb/age_category`, чтение из них
- `src/storage/docs_storage.py` — JOIN guidelines в `get`/`get_many`
- `src/RAG/ingestion/data_loader.py` — убрать splat манифеста в metadata
- `src/audit/diagnosis/clinic_recs.py` — матчинг/возраст через `GuidelinesStorage`, работает с `Guideline`
- `src/audit/pipeline.py` — источник `manifest_rows` (`list[Guideline]`) + `TODO(guidelines-sql)`
- `src/audit/icd_check/validator.py` — `_render_manifest_table` и `check_icd_codes` принимают `list[Guideline]`
- `src/LLM/decider.py` — `decide_file_id` принимает `list[Guideline]`, сериализует в JSON сам
- `src/LLM/icd_prefix_picker.py` — `IcdPrefixPicker.pick` принимает `list[Guideline]`, сериализует сам
- `src/reporting/result_parser.py` — убрать `load_manifest_meta` (CSV), добавить `build_manifest_meta(list[Guideline])`
- `src/reporting/api_formatter.py` — meta из `GuidelinesStorage` + `TODO`
- `src/audit/excel_formatter.py` — meta из `GuidelinesStorage`

---

## Task 1: Модель `Guideline` (чистые функции, без БД)

**Files:**
- Create: `src/storage/models/guideline.py`
- Modify: `src/storage/models/__init__.py`
- Test: `tests/test_guideline_model.py`

**Interfaces:**
- Consumes: ничего (первый таск).
- Produces:
  - `@dataclass Guideline` с полями: `file_id: str`, `name: str | None`, `mkb: list[str]`, `age_category: list[str]`, `developer: str | None`, `nps_status: str | None`, `published_at: str | None`, `usage_status: str | None`.
  - `Guideline.from_manifest_row(row: dict[str, str]) -> Guideline` — парсит CSV-строку (ключи: `ID`, `Наименование`, `МКБ-10`, `Возрастная категория`, `Разработчик`, `Статус одобрения НПС`, `Дата размещения`, `Статус применения`). `mkb`/`age_category` — split по запятой + strip; для `mkb` дополнительно `.upper()`; пустые ячейки → `[]`.

Никакого `to_manifest_row()` — объект передаётся дальше как есть; JSON-сериализация (если понадобится) делается точечно на границе с LLM (см. Task 6).

- [ ] **Step 1: Написать падающие тесты**

Создать `tests/test_guideline_model.py`:

```python
"""Юнит-тесты storage.models.guideline.Guideline — чистые функции, без БД."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from storage.models.guideline import Guideline


def _row(**overrides) -> dict[str, str]:
    base = {
        "ID": "581_2",
        "Наименование": "Острый бронхит",
        "МКБ-10": "J20.0, J20.1",
        "Возрастная категория": "Взрослые, дети",
        "Разработчик": "Минздрав",
        "Статус одобрения НПС": "Одобрено",
        "Дата размещения": "01.01.2020",
        "Статус применения": "Действует",
    }
    base.update(overrides)
    return base


def test_from_manifest_row_splits_mkb_into_upper_list():
    assert Guideline.from_manifest_row(_row()).mkb == ["J20.0", "J20.1"]


def test_from_manifest_row_uppercases_and_strips_mkb():
    g = Guideline.from_manifest_row(_row(**{"МКБ-10": " j20.0 ,j20.1 "}))
    assert g.mkb == ["J20.0", "J20.1"]


def test_from_manifest_row_splits_age_category_verbatim():
    assert Guideline.from_manifest_row(_row()).age_category == ["Взрослые", "дети"]


def test_from_manifest_row_single_values():
    g = Guideline.from_manifest_row(_row(**{"МКБ-10": "A15", "Возрастная категория": "Дети"}))
    assert g.mkb == ["A15"]
    assert g.age_category == ["Дети"]


def test_from_manifest_row_empty_cells_become_empty_lists():
    g = Guideline.from_manifest_row(_row(**{"МКБ-10": "", "Возрастная категория": ""}))
    assert g.mkb == []
    assert g.age_category == []


def test_from_manifest_row_maps_all_scalar_fields():
    g = Guideline.from_manifest_row(_row())
    assert g.file_id == "581_2"
    assert g.name == "Острый бронхит"
    assert g.developer == "Минздрав"
    assert g.nps_status == "Одобрено"
    assert g.published_at == "01.01.2020"
    assert g.usage_status == "Действует"
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `pytest tests/test_guideline_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'storage.models.guideline'`

- [ ] **Step 3: Реализовать модель**

Создать `src/storage/models/guideline.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

_ID = "ID"
_NAME = "Наименование"
_MKB = "МКБ-10"
_AGE = "Возрастная категория"
_DEVELOPER = "Разработчик"
_NPS = "Статус одобрения НПС"
_PUBLISHED = "Дата размещения"
_USAGE = "Статус применения"


def _split_csv_cell(cell: str, *, upper: bool = False) -> list[str]:
    """Разбить ячейку манифеста по запятой; strip; опционально upper. Пусто → []."""
    parts = [p.strip() for p in (cell or "").split(",")]
    parts = [p for p in parts if p]
    return [p.upper() for p in parts] if upper else parts


@dataclass
class Guideline:
    """Строка справочника клинреков (зеркало строки manifest.csv)."""

    file_id: str
    name: str | None = None
    mkb: list[str] = field(default_factory=list)
    age_category: list[str] = field(default_factory=list)
    developer: str | None = None
    nps_status: str | None = None
    published_at: str | None = None
    usage_status: str | None = None

    @classmethod
    def from_manifest_row(cls, row: dict[str, str]) -> "Guideline":
        return cls(
            file_id=(row.get(_ID) or "").strip(),
            name=row.get(_NAME) or None,
            mkb=_split_csv_cell(row.get(_MKB, ""), upper=True),
            age_category=_split_csv_cell(row.get(_AGE, "")),
            developer=row.get(_DEVELOPER) or None,
            nps_status=row.get(_NPS) or None,
            published_at=row.get(_PUBLISHED) or None,
            usage_status=row.get(_USAGE) or None,
        )
```

Добавить в `src/storage/models/__init__.py` (рядом с существующими экспортами):

```python
from .guideline import Guideline  # noqa: F401
```

- [ ] **Step 4: Запустить — убедиться, что проходит**

Run: `pytest tests/test_guideline_model.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Коммит**

```bash
git add src/storage/models/guideline.py src/storage/models/__init__.py tests/test_guideline_model.py
git commit -m "feat: модель Guideline с парсингом строки манифеста"
```

---

## Task 2: Миграция `019_guidelines.sql`

**Files:**
- Create: `migrations/019_guidelines.sql`

**Interfaces:**
- Consumes: ничего.
- Produces: таблица `guidelines(file_id PK, name, mkb TEXT[], age_category TEXT[], developer, nps_status, published_at, usage_status)` + GIN-индекс `guidelines_mkb_idx` на `mkb`.

Автотеста нет (DDL против БД — только стенд).

- [ ] **Step 1: Написать миграцию**

Создать `migrations/019_guidelines.sql`:

```sql
-- Migration 019: таблица guidelines — справочник клинреков (зеркало manifest.csv).
--
-- Единственный канонический источник справочника. Связывается с docs
-- внешним ключом по file_id (см. 021_docs_guideline_fk.sql).
--
-- mkb / age_category — массивы: один клинрек покрывает несколько кодов МКБ-10
-- и может относиться к нескольким возрастным категориям.
-- age_category хранится дословно как в CSV ('Взрослые', 'Дети').

CREATE TABLE IF NOT EXISTS guidelines (
    file_id       TEXT   PRIMARY KEY,               -- манифестный ID (= docs.file_id)
    name          TEXT,                             -- Наименование
    mkb           TEXT[] NOT NULL DEFAULT '{}',     -- МКБ-10: ['J20.0','J20.1']
    age_category  TEXT[] NOT NULL DEFAULT '{}',     -- ['Взрослые','Дети']
    developer     TEXT,                             -- Разработчик
    nps_status    TEXT,                             -- Статус одобрения НПС
    published_at  TEXT,                             -- Дата размещения (строка как в CSV)
    usage_status  TEXT                              -- Статус применения
);

-- GIN — матчинг по коду МКБ (code = ANY(mkb)) это горячий путь аудита.
CREATE INDEX IF NOT EXISTS guidelines_mkb_idx ON guidelines USING GIN (mkb);
```

- [ ] **Step 2: Проверить синтаксис локально (без БД)**

Run: `grep -c "CREATE TABLE IF NOT EXISTS guidelines" migrations/019_guidelines.sql`
Expected: `1`

- [ ] **Step 3: Коммит**

```bash
git add migrations/019_guidelines.sql
git commit -m "feat: миграция 019 — таблица guidelines"
```

---

## Task 3: `GuidelinesStorage` + seed-скрипт

**Files:**
- Create: `src/storage/guidelines_storage.py`
- Create: `scripts/seed-guidelines.py`
- Test: `tests/test_guidelines_storage.py`

**Interfaces:**
- Consumes: `Guideline`, `Guideline.from_manifest_row` (Task 1); таблица `guidelines` (Task 2).
- Produces: `class GuidelinesStorage(BaseStorage)` с async-методами:
  - `upsert_many(rows: list[Guideline]) -> int` — число записанных строк (upsert по PK `file_id`).
  - `get(file_id: str) -> Guideline | None`
  - `all() -> list[Guideline]`
  - `find_by_code(code: str) -> list[Guideline]` — `WHERE %(code)s = ANY(mkb)`, `code` нормализуется `.strip().upper()`.
  - `find_by_prefix(prefix: str) -> list[Guideline]` — `WHERE EXISTS (SELECT 1 FROM unnest(mkb) c WHERE split_part(c,'.',1) = %(prefix)s)`.

- [ ] **Step 1: Написать падающие интеграционные тесты**

Создать `tests/test_guidelines_storage.py` (паттерн из `tests/test_api_keys_storage.py`):

```python
"""Интеграционные тесты storage.guidelines_storage.GuidelinesStorage.

Требует настроенный Postgres (.env) с применённой миграцией 019.
Запускается на стенде — на dev-машине нет доступа к БД.
Каждый тест чистит вставленные им строки.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from storage.guidelines_storage import GuidelinesStorage
from storage.models.guideline import Guideline


def _g(file_id: str, mkb: list[str], age: list[str], name: str = "test") -> Guideline:
    return Guideline(file_id=file_id, name=name, mkb=mkb, age_category=age)


async def _cleanup(storage: GuidelinesStorage, file_ids: list[str]) -> None:
    async with storage._pool.connection() as conn:
        await conn.execute(
            "DELETE FROM guidelines WHERE file_id = ANY(%(ids)s)", {"ids": file_ids}
        )


async def test_upsert_and_get_roundtrip():
    fid = f"test_{uuid.uuid4().hex}"
    async with GuidelinesStorage() as storage:
        try:
            n = await storage.upsert_many([_g(fid, ["J20.0", "J20.1"], ["Взрослые"])])
            assert n == 1
            got = await storage.get(fid)
            assert got is not None
            assert got.mkb == ["J20.0", "J20.1"]
            assert got.age_category == ["Взрослые"]
        finally:
            await _cleanup(storage, [fid])


async def test_upsert_is_idempotent_by_file_id():
    fid = f"test_{uuid.uuid4().hex}"
    async with GuidelinesStorage() as storage:
        try:
            await storage.upsert_many([_g(fid, ["A15"], ["Дети"], name="one")])
            await storage.upsert_many([_g(fid, ["A16"], ["Взрослые"], name="two")])
            got = await storage.get(fid)
            assert got.name == "two"
            assert got.mkb == ["A16"]
        finally:
            await _cleanup(storage, [fid])


async def test_find_by_code_matches_array_member():
    fid = f"test_{uuid.uuid4().hex}"
    async with GuidelinesStorage() as storage:
        try:
            await storage.upsert_many([_g(fid, ["J20.0", "J20.1"], ["Взрослые"])])
            found = await storage.find_by_code("j20.1")  # нижний регистр — нормализуется
            assert fid in {g.file_id for g in found}
            assert not any(g.file_id == fid for g in await storage.find_by_code("Z99.9"))
        finally:
            await _cleanup(storage, [fid])


async def test_find_by_prefix_strips_subcategory():
    fid = f"test_{uuid.uuid4().hex}"
    async with GuidelinesStorage() as storage:
        try:
            await storage.upsert_many([_g(fid, ["J20.9"], ["Взрослые"])])
            found = await storage.find_by_prefix("J20")
            assert fid in {g.file_id for g in found}
        finally:
            await _cleanup(storage, [fid])
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `pytest tests/test_guidelines_storage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'storage.guidelines_storage'` (на dev-машине без БД тоже упадёт — зелёный прогон на стенде).

- [ ] **Step 3: Реализовать `GuidelinesStorage`**

Создать `src/storage/guidelines_storage.py`:

```python
"""GuidelinesStorage — async psycopg3 интерфейс к таблице guidelines."""
from __future__ import annotations

from .base import BaseStorage
from .models.guideline import Guideline

_COLS = "file_id, name, mkb, age_category, developer, nps_status, published_at, usage_status"


def _row_to_guideline(row: dict) -> Guideline:
    return Guideline(
        file_id=row["file_id"],
        name=row["name"],
        mkb=list(row["mkb"] or []),
        age_category=list(row["age_category"] or []),
        developer=row["developer"],
        nps_status=row["nps_status"],
        published_at=row["published_at"],
        usage_status=row["usage_status"],
    )


class GuidelinesStorage(BaseStorage):
    """Async context-manager для таблицы guidelines (общий пул BaseStorage)."""

    async def upsert_many(self, rows: list[Guideline]) -> int:
        if not rows:
            return 0
        written = 0
        async with self._pool.connection() as conn:
            for g in rows:
                await conn.execute(
                    """
                    INSERT INTO guidelines
                        (file_id, name, mkb, age_category, developer,
                         nps_status, published_at, usage_status)
                    VALUES
                        (%(file_id)s, %(name)s, %(mkb)s, %(age_category)s, %(developer)s,
                         %(nps_status)s, %(published_at)s, %(usage_status)s)
                    ON CONFLICT (file_id) DO UPDATE SET
                        name         = EXCLUDED.name,
                        mkb          = EXCLUDED.mkb,
                        age_category = EXCLUDED.age_category,
                        developer    = EXCLUDED.developer,
                        nps_status   = EXCLUDED.nps_status,
                        published_at = EXCLUDED.published_at,
                        usage_status = EXCLUDED.usage_status
                    """,
                    {
                        "file_id": g.file_id,
                        "name": g.name,
                        "mkb": g.mkb,
                        "age_category": g.age_category,
                        "developer": g.developer,
                        "nps_status": g.nps_status,
                        "published_at": g.published_at,
                        "usage_status": g.usage_status,
                    },
                )
                written += 1
        return written

    async def get(self, file_id: str) -> Guideline | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_COLS} FROM guidelines WHERE file_id = %(file_id)s",
                {"file_id": file_id},
            )
            row = await cur.fetchone()
        return _row_to_guideline(row) if row else None

    async def all(self) -> list[Guideline]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(f"SELECT {_COLS} FROM guidelines ORDER BY file_id")
            rows = await cur.fetchall()
        return [_row_to_guideline(r) for r in rows]

    async def find_by_code(self, code: str) -> list[Guideline]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_COLS} FROM guidelines WHERE %(code)s = ANY(mkb)",
                {"code": code.strip().upper()},
            )
            rows = await cur.fetchall()
        return [_row_to_guideline(r) for r in rows]

    async def find_by_prefix(self, prefix: str) -> list[Guideline]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_COLS} FROM guidelines "
                "WHERE EXISTS (SELECT 1 FROM unnest(mkb) c WHERE split_part(c, '.', 1) = %(prefix)s)",
                {"prefix": prefix.strip().upper()},
            )
            rows = await cur.fetchall()
        return [_row_to_guideline(r) for r in rows]
```

- [ ] **Step 4: Реализовать seed-скрипт**

Создать `scripts/seed-guidelines.py`:

```python
"""seed-guidelines.py — залить resources/manifest.csv в таблицу guidelines.

Запускать после миграции 019 и ДО FK-миграции 021 (см. spec §4).
Идемпотентно: upsert по file_id.

    python scripts/seed-guidelines.py
"""
from __future__ import annotations

import asyncio
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from storage.guidelines_storage import GuidelinesStorage
from storage.models.guideline import Guideline

_MANIFEST = ROOT / "resources" / "manifest.csv"


async def main() -> None:
    with open(_MANIFEST, newline="", encoding="utf-8") as fh:
        rows = [Guideline.from_manifest_row(r) for r in csv.DictReader(fh) if (r.get("ID") or "").strip()]
    async with GuidelinesStorage() as storage:
        written = await storage.upsert_many(rows)
    print(f"seeded {written} guideline(s) from {_MANIFEST.name}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 5: Прогнать storage-тесты (стенд)**

Run (на стенде, после `migrations/migrate.sh`): `pytest tests/test_guidelines_storage.py -v`
Expected: PASS (4 passed). Локально без БД — пропустить.

- [ ] **Step 6: Коммит**

```bash
git add src/storage/guidelines_storage.py scripts/seed-guidelines.py tests/test_guidelines_storage.py
git commit -m "feat: GuidelinesStorage + seed-скрипт манифеста"
```

---

## Task 4: Ingestion — убрать splat манифеста в `docs.metadata`

**Files:**
- Modify: `src/RAG/ingestion/data_loader.py`

**Interfaces:**
- Consumes: ничего нового.
- Produces: чанки, в `metadata` которых только chunk-intrinsic ключи (`section`, `content_type`, `chunk_index`, `page`, `table_index`) — без колонок манифеста. `DocumentLoader.metadata` (полная строка манифеста) НЕ трогаем — `ingest-pdfs.py` читает из неё `["ID"]` для `file_id`.

- [ ] **Step 1: Найти splat-места**

Run: `grep -n "\*\*self.metadata\|\*\*base_meta" src/RAG/ingestion/data_loader.py`
Expected: строки ~221 (text chunk `**self.metadata`), ~233 (`base_meta` с `**self.metadata`), ~271 (table chunk `**base_meta`).

- [ ] **Step 2: Убрать splat манифеста в text-чанке**

В `src/RAG/ingestion/data_loader.py`, блок text-чанка (около 217-226) заменить:

```python
                    yield {
                        "type": "text",
                        "content": sub_chunk,
                        "metadata": {
                            **self.metadata,
                            "section": section_title,
                            "content_type": "text",
                            "chunk_index": chunk_counter,
                        },
                    }
```

на:

```python
                    yield {
                        "type": "text",
                        "content": sub_chunk,
                        "metadata": {
                            "section": section_title,
                            "content_type": "text",
                            "chunk_index": chunk_counter,
                        },
                    }
```

- [ ] **Step 3: Убрать splat манифеста в `base_meta` (table-чанки)**

В том же файле блок `base_meta` (около 232-237) заменить:

```python
            base_meta = {
                **self.metadata,
                "page": page_idx,
                "section": section,
                "content_type": "table",
            }
```

на:

```python
            base_meta = {
                "page": page_idx,
                "section": section,
                "content_type": "table",
            }
```

(`**base_meta` в table-чанке ниже — оставить; теперь там только intrinsic-ключи.)

- [ ] **Step 4: Проверить, что манифест-ключей в metadata чанков не осталось**

Run: `grep -n "\"metadata\":" src/RAG/ingestion/data_loader.py`
Затем визуально убедиться, что ни в одном из этих словарей нет `**self.metadata`.
Expected: `**self.metadata` внутри `"metadata": {...}` отсутствует.

- [ ] **Step 5: Коммит**

```bash
git add src/RAG/ingestion/data_loader.py
git commit -m "refactor: не дублировать манифест в metadata чанков docs"
```

---

## Task 5: `Doc` + `DocsStorage` JOIN — читаемые поля из guidelines

**Files:**
- Modify: `src/storage/models/doc.py`
- Modify: `src/storage/docs_storage.py`
- Test: `tests/test_doc_format_chunk.py`

**Interfaces:**
- Consumes: таблица `guidelines` (Task 2).
- Produces:
  - `Doc` получает поля: `name: str | None = None`, `mkb: list[str] = field(default_factory=list)`, `age_category: list[str] = field(default_factory=list)`.
  - `Doc._format_chunk` читает `self.name` / `self.mkb` / `self.age_category` (массивы → `", ".join`) вместо `self.metadata[...]`. Шапка байт-в-байт прежняя.
  - `DocsStorage.get` / `get_many` делают `LEFT JOIN guidelines g ON g.file_id = docs.file_id`, выбирая `g.name AS g_name, g.mkb AS g_mkb, g.age_category AS g_age_category`; `_row_to_doc` кладёт их в новые поля `Doc`.

- [ ] **Step 1: Написать падающий тест шапки чанка**

Создать `tests/test_doc_format_chunk.py`:

```python
"""Регрессия Doc._format_chunk — шапка чанка при чтении полей из guidelines."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from storage.models.doc import Doc


def test_header_from_guideline_fields():
    doc = Doc(
        chunk="тело фрагмента",
        file_id="581_2",
        name="Острый бронхит",
        mkb=["J20.0", "J20.1"],
        age_category=["Взрослые", "дети"],
        metadata={"section": "Диагностика", "content_type": "text", "chunk_index": 3},
    )
    out = doc._format_chunk()
    assert "Острый бронхит | МКБ-10: J20.0, J20.1 | Взрослые, дети" in out
    assert "Диагностика | фрагмент 3" in out
    assert out.endswith("тело фрагмента")


def test_header_omits_absent_fields():
    doc = Doc(chunk="тело", file_id="x", metadata={"section": "S", "content_type": "text"})
    out = doc._format_chunk()
    assert out.endswith("тело")
    assert "МКБ-10" not in out
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `pytest tests/test_doc_format_chunk.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'name'`.

- [ ] **Step 3: Добавить поля в `Doc` и переключить `_format_chunk`**

В `src/storage/models/doc.py`, после поля `metadata`:

```python
    metadata: dict = field(default_factory=dict)

    # Денормализованные из guidelines через JOIN по file_id (populated on read).
    name: str | None = None
    mkb: list[str] = field(default_factory=list)
    age_category: list[str] = field(default_factory=list)
```

В `_format_chunk` заменить:

```python
        name: str | None = self.metadata.get("Наименование")
        mkb: str | None = self.metadata.get("МКБ-10")
        age_cat: str | None = self.metadata.get("Возрастная категория")
```

на:

```python
        name: str | None = self.name
        mkb: str | None = ", ".join(self.mkb) if self.mkb else None
        age_cat: str | None = ", ".join(self.age_category) if self.age_category else None
```

(`section` и `chunk_idx` по-прежнему из `self.metadata` — они chunk-intrinsic.)

- [ ] **Step 4: Запустить тест шапки**

Run: `pytest tests/test_doc_format_chunk.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Добавить JOIN в `DocsStorage` и `_row_to_doc`**

В `src/storage/docs_storage.py`, `_row_to_doc`:

```python
def _row_to_doc(row: dict) -> Doc:
    return Doc(
        id=row["id"],
        file_id=row["file_id"],
        chunk=row["chunk"],
        metadata=row["metadata"],
        fact_q=row.get("fact_q"),
        procedure_q=row.get("procedure_q"),
        constraint_q=row.get("constraint_q"),
        name=row.get("g_name"),
        mkb=list(row.get("g_mkb") or []),
        age_category=list(row.get("g_age_category") or []),
    )
```

В `get` — SELECT:

```python
                """
                SELECT
                    docs.id::text AS id, docs.file_id, docs.chunk, docs.metadata,
                    docs.fact_q, docs.procedure_q, docs.constraint_q,
                    g.name AS g_name, g.mkb AS g_mkb, g.age_category AS g_age_category
                FROM docs
                LEFT JOIN guidelines g ON g.file_id = docs.file_id
                WHERE docs.id = %(id)s::uuid
                """
```

В `get_many` — SELECT:

```python
                """
                SELECT
                    docs.id::text AS id, docs.file_id, docs.chunk, docs.metadata,
                    docs.fact_q, docs.procedure_q, docs.constraint_q,
                    g.name AS g_name, g.mkb AS g_mkb, g.age_category AS g_age_category
                FROM docs
                LEFT JOIN guidelines g ON g.file_id = docs.file_id
                WHERE docs.id = ANY(%(ids)s::uuid[])
                """
```

(`by_id = {r["id"]: ...}` в `get_many` продолжает работать — `id` явно алиасится.)

- [ ] **Step 6: Прогнать локальные тесты**

Run: `pytest tests/test_doc_format_chunk.py tests/test_guideline_model.py -v`
Expected: PASS. (JOIN-путь — на стенде через `test_vector_store.py` / RAG-тесты.)

- [ ] **Step 7: Коммит**

```bash
git add src/storage/models/doc.py src/storage/docs_storage.py tests/test_doc_format_chunk.py
git commit -m "feat: Doc читает name/mkb/age из guidelines через JOIN"
```

---

## Task 6: `decider` и `icd_prefix_picker` — принимают `list[Guideline]`

Эти два модуля — единственное место, где кандидаты реально нужно сериализовать в dict/JSON (промпт LLM). Переводим их сигнатуру на `Guideline` первыми, чтобы Task 7 (`clinic_recs`) мог сразу передавать объекты, не городя адаптер.

**Files:**
- Modify: `src/LLM/decider.py`
- Modify: `src/LLM/icd_prefix_picker.py`
- Test: `tests/test_decider_guideline_input.py`

**Interfaces:**
- Consumes: `Guideline` (Task 1).
- Produces:
  - `decide_file_id(patient: dict, diagnosis: dict, candidates: list[Guideline]) -> tuple[str | None, int]` — сериализует `candidates` через `dataclasses.asdict` перед `json.dumps`; валидирует ответ по `{g.file_id for g in candidates}`.
  - `IcdPrefixPicker.pick(patient: dict, diagnosis: dict, candidates: list[Guideline]) -> tuple[str | None, int]` — аналогично.

- [ ] **Step 1: Написать падающие тесты**

Создать `tests/test_decider_guideline_input.py` — проверяем только сериализацию и валидацию ответа, не реальный вызов LLM (мокаем `_client.call` / `self._client.call`):

```python
"""decider/icd_prefix_picker принимают list[Guideline] напрямую (без dict-адаптера)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from storage.models.guideline import Guideline

_CANDIDATES = [
    Guideline(file_id="581_2", name="Острый бронхит", mkb=["J20.0"], age_category=["Взрослые"]),
    Guideline(file_id="581_3", name="Хронический бронхит", mkb=["J20.1"], age_category=["Взрослые"]),
]


async def test_decide_file_id_accepts_guideline_list_and_serializes_them():
    from LLM.decider import decide_file_id

    with patch("LLM.decider._client") as mock_client:
        mock_client.call = AsyncMock(return_value=("581_2", 10))
        chosen, tokens = await decide_file_id({"Возраст": 30}, {"КодМКБ": "J20.0"}, _CANDIDATES)

    assert chosen == "581_2"
    assert tokens == 10
    # candidates были сериализованы в промпт (JSON с их именами)
    user_msg = mock_client.call.call_args.kwargs["messages"][1]["content"]
    assert "Острый бронхит" in user_msg
    assert "581_2" in user_msg


async def test_decide_file_id_rejects_answer_outside_candidates():
    from LLM.decider import decide_file_id

    with patch("LLM.decider._client") as mock_client:
        mock_client.call = AsyncMock(return_value=("not_a_real_id", 5))
        chosen, tokens = await decide_file_id({}, {"КодМКБ": "J20.0"}, _CANDIDATES)

    assert chosen is None
    assert tokens == 5


async def test_icd_prefix_picker_accepts_guideline_list():
    from LLM.icd_prefix_picker import IcdPrefixPicker

    picker = IcdPrefixPicker()
    with patch.object(picker, "_client") as mock_client:
        mock_client.call = AsyncMock(return_value=("581_3", 7))
        chosen, tokens = await picker.pick({}, {"КодМКБ": "J20.1"}, _CANDIDATES)

    assert chosen == "581_3"
    assert tokens == 7
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `pytest tests/test_decider_guideline_input.py -v`
Expected: FAIL — текущий код читает `row.get("ID", "")` из dict; `Guideline` не поддерживает `.get()` → `AttributeError`.

- [ ] **Step 3: Переписать `decider.py`**

В `src/LLM/decider.py` изменить сигнатуру и тело:

```python
"""
decider.py — LLM-based selection of the most relevant clinical-guideline
document when multiple candidates match an МКБ-10 code.

Given patient metadata, the diagnosis record, and the list of matching
Guideline candidates, the LLM picks the single most relevant file_id.

Usage::
    from LLM.decider import decide_file_id

    file_id, tokens = await decide_file_id(patient, diagnosis, candidates)
"""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

from LLM.client import LLMClient
from storage.models.guideline import Guideline

_client = LLMClient()


async def decide_file_id(
    patient: dict[str, Any],
    diagnosis: dict[str, Any],
    candidates: list[Guideline],
) -> tuple[str | None, int]:
    """Ask the LLM to pick the most relevant guideline file_id.

    Args:
        patient:    The «Пациент» dict from the raw visit JSON.
        diagnosis:  A single entry from «Диагнозы».
        candidates: Guideline objects that matched the МКБ code.

    Returns:
        (chosen file_id or None, tokens spent).
    """
    candidate_json = json.dumps([asdict(c) for c in candidates], ensure_ascii=False, indent=2)
    diagnosis_json = json.dumps(diagnosis, ensure_ascii=False, indent=2)
    patient_json = json.dumps(patient, ensure_ascii=False, indent=2)

    system = (
        "Ты — медицинский эксперт. Тебе даны данные о пациенте, диагноз и список "
        "клинических рекомендаций, подходящих по коду МКБ-10. "
        "Выбери ОДНУ наиболее подходящую рекомендацию для данного пациента и диагноза. "
        "Ответь ТОЛЬКО значением поля file_id выбранной рекомендации, без пояснений."
    )

    user = (
        f"## Пациент\n{patient_json}\n\n"
        f"## Диагноз\n{diagnosis_json}\n\n"
        f"## Кандидаты (клинические рекомендации)\n{candidate_json}"
    )

    raw_content, tokens = await _client.call(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
    )

    logger.debug("[decider] raw LLM answer: %s", raw_content)
    chosen = raw_content.strip().strip('"').strip("'")
    valid_ids = {c.file_id for c in candidates}
    return (chosen if chosen in valid_ids else None), tokens
```

(Промпт теперь просит `file_id` вместо `ID` — согласовано с полем `Guideline.file_id`, которое и сериализуется в JSON через `asdict`.)

- [ ] **Step 4: Переписать `icd_prefix_picker.py`**

В `src/LLM/icd_prefix_picker.py` изменить сигнатуру и тело:

```python
"""
icd_prefix_picker.py — LLM-based selection of a clinical guideline when the
exact МКБ-10 code had no match and a prefix-only lookup returned candidates.

Usage::
    from LLM.icd_prefix_picker import IcdPrefixPicker

    picker = IcdPrefixPicker()
    file_id, tokens = await picker.pick(patient, diagnosis, candidates)
"""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from LLM.client import LLMClient
from storage.models.guideline import Guideline

_PROMPT_PATH = Path(__file__).parent / "prompts" / "icd_prefix_picker.txt"


class IcdPrefixPicker:
    """Pick a guideline file_id from prefix-matched candidates using an LLM."""

    def __init__(self) -> None:
        self._system = _PROMPT_PATH.read_text(encoding="utf-8").strip()
        self._client = LLMClient()

    async def pick(
        self,
        patient: dict[str, Any],
        diagnosis: dict[str, Any],
        candidates: list[Guideline],
    ) -> tuple[str | None, int]:
        """Return the most relevant guideline file_id among prefix-matched *candidates*.

        Args:
            patient:    Patient info dict.
            diagnosis:  Diagnosis dict with at least ``КодМКБ`` key.
            candidates: Guideline objects matched by the ICD prefix (e.g. ``J20``).

        Returns:
            (chosen file_id or None, tokens spent).
        """
        candidate_json = json.dumps([asdict(c) for c in candidates], ensure_ascii=False, indent=2)
        user = (
            f"## Пациент\n{json.dumps(patient, ensure_ascii=False, indent=2)}\n\n"
            f"## Диагноз\n{json.dumps(diagnosis, ensure_ascii=False, indent=2)}\n\n"
            f"## Кандидаты (клинические рекомендации)\n{candidate_json}"
        )

        raw, tokens = await self._client.call(
            messages=[
                {"role": "system", "content": self._system},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
        )

        logger.debug("[icd_prefix_picker] raw answer: %s", raw)
        chosen = raw.strip().strip('"').strip("'")
        if chosen.lower() == "none":
            return None, tokens
        valid_ids = {c.file_id for c in candidates}
        return (chosen if chosen in valid_ids else None), tokens
```

- [ ] **Step 5: Проверить промпт-файл на упоминание поля `ID`**

Run: `grep -n "\bID\b" src/LLM/prompts/icd_prefix_picker.txt`
Если найдено упоминание поля `ID` как ключа ответа — заменить на `file_id` для согласованности с сериализацией через `asdict` (ключ dataclass — `file_id`, не `ID`).

- [ ] **Step 6: Запустить тесты**

Run: `pytest tests/test_decider_guideline_input.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Коммит**

```bash
git add src/LLM/decider.py src/LLM/icd_prefix_picker.py tests/test_decider_guideline_input.py
git commit -m "refactor: decider/icd_prefix_picker принимают Guideline вместо dict"
```

---

## Task 7: `clinic_recs` — матчинг и возраст через `GuidelinesStorage`

**Files:**
- Modify: `src/audit/diagnosis/clinic_recs.py`
- Test: `tests/test_clinic_recs_age.py`

**Interfaces:**
- Consumes: `GuidelinesStorage.find_by_code`, `find_by_prefix` (Task 3); `decide_file_id`, `IcdPrefixPicker.pick` принимающие `list[Guideline]` (Task 6).
- Produces:
  - `_is_age_eligible(guideline: Guideline, age: int | None) -> bool` — принимает `Guideline`, читает `guideline.age_category` (список), регистронезависимо.
  - `ClinicRecs.pick_recs` — без изменений сигнатуры (`async`, `-> tuple[str | None, int]`); внутри работает с `Guideline` от начала до конца, без dict-конверсий.
  - `ClinicRecs.__init__()` — больше не принимает `manifest_path`.
  - `_patient_age`, `_ADULT_THRESHOLD`, `_SKIP_CODES` — сохраняются.

- [ ] **Step 1: Написать падающий юнит-тест**

Создать `tests/test_clinic_recs_age.py`:

```python
"""Юнит-тесты clinic_recs._is_age_eligible (по Guideline)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.diagnosis.clinic_recs import _is_age_eligible
from storage.models.guideline import Guideline


def _g(age_category: list[str]) -> Guideline:
    return Guideline(file_id="x", age_category=age_category)


@pytest.mark.parametrize("age,cats,expected", [
    (None, ["Дети"], True),
    (10, ["Дети"], True),
    (30, ["Дети"], False),
    (30, ["Взрослые"], True),
    (10, ["Взрослые"], False),
    (10, ["Взрослые", "Дети"], True),
    (30, ["Взрослые", "дети"], True),
    (10, [], True),
    (10, ["дети"], True),
])
def test_is_age_eligible(age, cats, expected):
    assert _is_age_eligible(_g(cats), age) is expected
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `pytest tests/test_clinic_recs_age.py -v`
Expected: FAIL — сейчас `_is_age_eligible` ждёт `dict` (`AttributeError`/неверный результат).

- [ ] **Step 3: Переписать `_is_age_eligible`**

В `src/audit/diagnosis/clinic_recs.py` заменить функцию:

```python
def _is_age_eligible(guideline: "Guideline", age: int | None) -> bool:
    """Return False if the guideline's age category contradicts the patient's age."""
    if age is None:
        return True
    cats = {c.strip().lower() for c in guideline.age_category}
    is_child = age <= _ADULT_THRESHOLD
    has_child = "дети" in cats
    has_adult = "взрослые" in cats
    if has_child and not has_adult:
        return is_child
    if has_adult and not has_child:
        return not is_child
    return True  # оба или неизвестно — пропускаем
```

Добавить импорты вверху:

```python
from storage.guidelines_storage import GuidelinesStorage
from storage.models.guideline import Guideline
```

- [ ] **Step 4: Переписать матчинг в `ClinicRecs`**

Заменить `__init__` и тело `pick_recs`, удалить `_load_manifest` / `_find_matching_rows` / `_find_matching_rows_by_prefix`:

```python
    def __init__(self) -> None:
        self._prefix_picker = IcdPrefixPicker()

    async def pick_recs(
        self,
        patient: dict[str, Any],
        diagnosis: dict[str, Any],
    ) -> tuple[str | None, int]:
        icd_raw: str = diagnosis.get("КодМКБ", "")
        normalised = icd_raw.strip().upper()

        if not normalised or normalised in _SKIP_CODES:
            return None, 0

        age = _patient_age(patient)
        async with GuidelinesStorage() as store:
            matched = [g for g in await store.find_by_code(normalised) if _is_age_eligible(g, age)]

            if not matched:
                prefix = normalised.split(".")[0]
                if prefix != normalised:
                    candidates = [g for g in await store.find_by_prefix(prefix) if _is_age_eligible(g, age)]
                    if candidates:
                        return await self._prefix_picker.pick(patient, diagnosis, candidates)
                return None, 0

        if len(matched) == 1:
            return matched[0].file_id or None, 0

        diag_name: str = diagnosis.get("НаименованиеМКБ", "").lower()
        diag_tokens = set(diag_name.split())
        scores = [len(diag_tokens & set((g.name or "").lower().split())) for g in matched]
        best_score = max(scores)
        if best_score > 0:
            best = matched[scores.index(best_score)]
            return best.file_id or None, 0

        return await decide_file_id(patient, diagnosis, matched)
```

Удалить константы `_ICD_COLUMN`, `_ID_COLUMN`, `_NAME_COLUMN`, `_AGE_COLUMN`, `_MANIFEST_PATH` и `import csv` (больше не нужны).

- [ ] **Step 5: Запустить возрастной юнит-тест**

Run: `pytest tests/test_clinic_recs_age.py -v`
Expected: PASS (9 passed)

- [ ] **Step 6: Прогнать смежные тесты (стенд)**

Run: `pytest tests/test_validations.py tests/test_pipeline_multiple_diagnoses.py -v`
Expected: PASS на стенде (бьют БД/LLM). Локально без БД — пропустить.

- [ ] **Step 7: Коммит**

```bash
git add src/audit/diagnosis/clinic_recs.py tests/test_clinic_recs_age.py
git commit -m "refactor: clinic_recs матчит МКБ через GuidelinesStorage, работает с Guideline"
```

---

## Task 8: `pipeline` + `icd_check` — источник `manifest_rows` из БД, тип `list[Guideline]`

**Files:**
- Modify: `src/audit/pipeline.py:211-215,31`
- Modify: `src/audit/icd_check/validator.py` — `_render_manifest_table` и `check_icd_codes` типизация

**Interfaces:**
- Consumes: `GuidelinesStorage.all` (Task 3); `_is_age_eligible(Guideline, age)` (Task 7).
- Produces: `check_icd_codes(..., manifest_rows: list[Guideline])`. `_render_manifest_table(rows: list[Guideline])` рендерит `g.file_id`/`g.name`/`", ".join(g.mkb)`/`", ".join(g.age_category)` напрямую — без промежуточного dict.

- [ ] **Step 1: Переписать `_render_manifest_table` под `Guideline`**

В `src/audit/icd_check/validator.py` заменить:

```python
def _render_manifest_table(rows: list[dict[str, str]]) -> str:
    """Render age-filtered manifest rows as a plain text table for the agent."""
    header = "ID | Наименование | МКБ-10 | Возрастная категория"
    sep = "-" * len(header)
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"{row.get('ID', '')} | {row.get('Наименование', '')} | "
            f"{row.get('МКБ-10', '')} | {row.get('Возрастная категория', '')}"
        )
    return "\n".join(lines)
```

на:

```python
def _render_manifest_table(rows: list["Guideline"]) -> str:
    """Render age-filtered guideline rows as a plain text table for the agent."""
    header = "ID | Наименование | МКБ-10 | Возрастная категория"
    sep = "-" * len(header)
    lines = [header, sep]
    for g in rows:
        lines.append(
            f"{g.file_id} | {g.name or ''} | "
            f"{', '.join(g.mkb)} | {', '.join(g.age_category)}"
        )
    return "\n".join(lines)
```

Добавить импорт в начало файла:

```python
from storage.models.guideline import Guideline
```

- [ ] **Step 2: Обновить сигнатуру и докстринг `check_icd_codes`**

В том же файле изменить:

```python
async def check_icd_codes(
    patient: dict[str, Any],
    diagnoses: list[dict[str, Any]],
    manifest_rows: list[dict[str, str]],
    inspection_data: list[dict[str, Any]] | None = None,
) -> tuple[list[IcdCodingIssue], int]:
```

на:

```python
async def check_icd_codes(
    patient: dict[str, Any],
    diagnoses: list[dict[str, Any]],
    manifest_rows: list[Guideline],
    inspection_data: list[dict[str, Any]] | None = None,
) -> tuple[list[IcdCodingIssue], int]:
```

В докстринге заменить `manifest_rows: Age-filtered rows from manifest.csv.` на `manifest_rows: Age-filtered Guideline rows from GuidelinesStorage.`.

- [ ] **Step 3: Заменить загрузку манифеста в pipeline**

В `src/audit/pipeline.py`, блок около 211-215:

```python
        clinic_recs = ClinicRecs()
        age = _patient_age(patient)
        all_manifest_rows = clinic_recs._load_manifest()
        manifest_rows = [r for r in all_manifest_rows if _is_age_eligible(r, age)]
```

заменить на:

```python
        clinic_recs = ClinicRecs()
        age = _patient_age(patient)
        # TODO(guidelines-sql): фильтрацию по возрасту вынести в SQL —
        # GuidelinesStorage.all_age_eligible(age) вместо загрузки всего
        # справочника и фильтра в Python. См. spec §5.
        async with GuidelinesStorage() as _store:
            manifest_rows = [g for g in await _store.all() if _is_age_eligible(g, age)]
```

- [ ] **Step 4: Добавить импорт в pipeline**

В `src/audit/pipeline.py` добавить:

```python
from storage.guidelines_storage import GuidelinesStorage
```

- [ ] **Step 5: Прогнать pipeline-тест (стенд)**

Run: `pytest tests/test_pipeline_multiple_diagnoses.py -v`
Expected: PASS на стенде.

- [ ] **Step 6: Коммит**

```bash
git add src/audit/pipeline.py src/audit/icd_check/validator.py
git commit -m "refactor: pipeline/icd_check работают с list[Guideline] вместо dict-манифеста"
```

---

## Task 9: Отчёт — meta из `GuidelinesStorage` вместо CSV

**Files:**
- Modify: `src/reporting/result_parser.py`
- Modify: `src/reporting/api_formatter.py:20,90`
- Modify: `src/audit/excel_formatter.py:26,141-179`
- Test: `tests/test_report_meta.py`

**Interfaces:**
- Consumes: `GuidelinesStorage.all` (Task 3), `Guideline` (Task 1).
- Produces:
  - `result_parser.build_manifest_meta(guidelines: list[Guideline]) -> dict[str, dict]` — заменяет `load_manifest_meta()`; строит `{file_id: {"name", "date", "age_group"}}`. `date` = `published_at`, `age_group` = `", ".join(age_category)`.
  - `load_manifest_meta()` удаляется. `parse_diagnosis(data, manifest_meta)` — без изменений сигнатуры (принимает готовый dict, не `Guideline`).
  - `ExcelFormatter` и `api_formatter` подгружают meta через `GuidelinesStorage` в async-контексте и передают dict в `parse_diagnosis`.

- [ ] **Step 1: Написать падающий тест**

Создать `tests/test_report_meta.py`:

```python
"""Юнит-тест reporting.result_parser.build_manifest_meta (из Guideline)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reporting.result_parser import build_manifest_meta
from storage.models.guideline import Guideline


def test_build_manifest_meta_shape():
    guidelines = [
        Guideline(file_id="581_2", name="Острый бронхит",
                  age_category=["Взрослые", "Дети"], published_at="01.01.2020"),
    ]
    meta = build_manifest_meta(guidelines)
    assert meta["581_2"] == {
        "name": "Острый бронхит",
        "date": "01.01.2020",
        "age_group": "Взрослые, Дети",
    }
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `pytest tests/test_report_meta.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_manifest_meta'`.

- [ ] **Step 3: Заменить `load_manifest_meta` на `build_manifest_meta`**

В `src/reporting/result_parser.py` удалить `load_manifest_meta` (и `import csv`, `_MANIFEST_PATH`), добавить:

```python
from storage.models.guideline import Guideline


def build_manifest_meta(guidelines: list["Guideline"]) -> dict[str, dict]:
    """Return {file_id: {name, date, age_group}} from Guideline objects."""
    return {
        g.file_id: {
            "name": g.name or "",
            "date": g.published_at or "",
            "age_group": ", ".join(g.age_category),
        }
        for g in guidelines
        if g.file_id
    }
```

- [ ] **Step 4: Обновить `ExcelFormatter`**

В `src/audit/excel_formatter.py` заменить импорт (строка 26):

```python
from reporting.result_parser import load_manifest_meta as _load_manifest_meta
```

на:

```python
from reporting.result_parser import build_manifest_meta as _build_manifest_meta
from storage.guidelines_storage import GuidelinesStorage
```

Изменить `export_*` и `_write_rows` (meta грузится в async-контексте, передаётся в sync-`_write_rows`):

```python
    async def export_all(self) -> int:
        rows = await self._reader.fetch_all()
        return self._write_rows(rows, await self._load_meta())

    async def export_period(self, date_from, date_to, organization_id) -> int:
        rows = await self._reader.fetch_by_period(date_from, date_to, organization_id)
        return self._write_rows(rows, await self._load_meta())

    async def export_guids(self, guids: set[str]) -> int:
        rows = await self._reader.fetch_by_guids(guids)
        return self._write_rows(rows, await self._load_meta())

    async def _load_meta(self) -> dict:
        async with GuidelinesStorage() as store:
            return _build_manifest_meta(await store.all())

    def _write_rows(self, rows, manifest_meta) -> int:
        existing = _existing_guids_in_excel(self._excel)
        written = 0
        for row in rows:
            guid = (row["card_guid"] or "").lower()
            if guid and guid in existing:
                logger.debug("📊 skipping already exported card guid=%s", guid)
                continue
            visit = row["card_data"]
            formal = _parse_formal(row["formal_result"])
            diagnosis = _parse_diagnosis(row["diag_result"], manifest_meta)
            icd_check = _parse_icd_check(row.get("icd_check_result") or [])
            self._excel.append(visit=visit, formal=formal, diagnosis=diagnosis, icd_check=icd_check)
            written += 1
        logger.info("📊 ExcelFormatter exported %d row(s)", written)
        return written
```

(Убрать прежнюю строку `manifest_meta = _load_manifest_meta()` из тела `_write_rows`.)

- [ ] **Step 5: Обновить `api_formatter`**

В `src/reporting/api_formatter.py` заменить в импорте `load_manifest_meta` на `build_manifest_meta`, добавить `from storage.guidelines_storage import GuidelinesStorage`.

Строку 90 `manifest_meta = load_manifest_meta()` заменить на:

```python
        async with GuidelinesStorage() as _store:
            manifest_meta = build_manifest_meta(await _store.all())
```

- [ ] **Step 6: Запустить локальный тест meta**

Run: `pytest tests/test_report_meta.py -v`
Expected: PASS (1 passed)

- [ ] **Step 7: Прогнать API-тесты (стенд)**

Run: `pytest tests/test_cards_api.py -v`
Expected: PASS на стенде.

- [ ] **Step 8: Коммит**

```bash
git add src/reporting/result_parser.py src/reporting/api_formatter.py src/audit/excel_formatter.py tests/test_report_meta.py
git commit -m "refactor: отчёт берёт meta клинреков из GuidelinesStorage"
```

---

## Task 10: Миграции бэкфилла — cleanup metadata + FK

**Files:**
- Create: `migrations/020_docs_metadata_cleanup.sql`
- Create: `migrations/021_docs_guideline_fk.sql`

**Interfaces:**
- Consumes: таблица `guidelines` (Task 2), заполненная seed-скриптом (Task 3).
- Produces: `docs.metadata` без манифест-ключей; FK `docs_file_id_fkey`. FK-миграция **падает**, если seed не выполнен (сироты), останавливая `migrate.sh`.

- [ ] **Step 1: cleanup-миграция**

Создать `migrations/020_docs_metadata_cleanup.sql`:

```sql
-- Migration 020: удалить дублирующие колонки манифеста из docs.metadata.
--
-- Читаемые поля (Наименование, МКБ-10, Возрастная категория) теперь резолвятся
-- через JOIN docs → guidelines по file_id. В metadata остаётся только
-- chunk-intrinsic: section, content_type, chunk_index, page, table_index.
--
-- WHERE делает миграцию идемпотентной: повторный прогон трогает 0 строк.

UPDATE docs SET metadata = metadata
        - 'ID' - 'Наименование' - 'МКБ-10' - 'Возрастная категория'
        - 'Разработчик' - 'Статус одобрения НПС'
        - 'Дата размещения' - 'Статус применения'
WHERE metadata ?| array['ID','Наименование','МКБ-10','Возрастная категория',
                        'Разработчик','Статус одобрения НПС',
                        'Дата размещения','Статус применения'];
```

- [ ] **Step 2: FK-миграция с проверкой сирот**

Создать `migrations/021_docs_guideline_fk.sql`:

```sql
-- Migration 021: FK docs.file_id → guidelines.file_id.
--
-- Требует, чтобы guidelines была заполнена seed-скриптом
-- (scripts/seed-guidelines.py) ДО этой миграции. Если seed не выполнен,
-- docs.file_id окажутся сиротами и миграция ЯВНО падает — migrate.sh
-- (ON_ERROR_STOP=1) остановится здесь. Это штатная защита порядка:
-- выполните seed и повторите migrate.sh.
--
-- Guard: constraint добавляется, только если его ещё нет (идемпотентно).

DO $$
DECLARE
    orphan_count integer;
BEGIN
    SELECT count(DISTINCT file_id) INTO orphan_count
    FROM docs
    WHERE file_id IS NOT NULL
      AND file_id NOT IN (SELECT file_id FROM guidelines);

    IF orphan_count > 0 THEN
        RAISE EXCEPTION
            'docs содержит % file_id без строки в guidelines — выполните scripts/seed-guidelines.py перед этой миграцией',
            orphan_count;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'docs_file_id_fkey'
    ) THEN
        ALTER TABLE docs
            ADD CONSTRAINT docs_file_id_fkey
            FOREIGN KEY (file_id) REFERENCES guidelines(file_id);
    END IF;
END$$;
```

- [ ] **Step 3: Проверить порядок имён**

Run: `ls migrations/0{19,20,21}*.sql`
Expected: `019_guidelines.sql`, `020_docs_metadata_cleanup.sql`, `021_docs_guideline_fk.sql`.

- [ ] **Step 4: Коммит**

```bash
git add migrations/020_docs_metadata_cleanup.sql migrations/021_docs_guideline_fk.sql
git commit -m "feat: миграции бэкфилла metadata + FK docs→guidelines"
```

---

## Task 11: Прогон стенда и финальная проверка

**Files:**
- Проверка: `resources/manifest.csv` остаётся в репо как seed-данные (НЕ удалять).

**Interfaces:**
- Consumes: всё выше.
- Produces: рабочая система на стенде; `manifest.csv` не читается рантаймом; нигде в рантайме не осталось dict-адаптеров манифеста.

- [ ] **Step 1: Убедиться, что рантайм больше не читает manifest.csv**

Run: `grep -rn "manifest.csv\|MANIFEST_PATH\|load_manifest_meta\|_load_manifest\b" src/`
Expected: пусто (упоминания допустимы только в комментариях/докстрингах). `scripts/seed-guidelines.py` и `resources/manifest.csv` — легитимные seed-места.

- [ ] **Step 2: Убедиться, что не осталось dict-строк манифеста в сигнатурах**

Run: `grep -rn "list\[dict\[str, str\]\]" src/audit/diagnosis/clinic_recs.py src/audit/icd_check/validator.py src/LLM/decider.py src/LLM/icd_prefix_picker.py`
Expected: пусто — все эти сигнатуры теперь типизированы как `list[Guideline]`.

- [ ] **Step 3: Локальные (без-БД) тесты**

Run: `pytest tests/test_guideline_model.py tests/test_doc_format_chunk.py tests/test_clinic_recs_age.py tests/test_report_meta.py tests/test_decider_guideline_input.py -v`
Expected: PASS (все).

- [ ] **Step 4: Стенд — миграции + seed (runbook)**

На стенде, по порядку:

```bash
bash migrations/migrate.sh        # применит 019, 020; на 021 УПАДЁТ (сироты — seed ещё не сделан)
python scripts/seed-guidelines.py # зальёт manifest.csv в guidelines
bash migrations/migrate.sh        # теперь 021 (FK) пройдёт; 019/020 идемпотентны
```

Expected: первый прогон останавливается на 021 с сообщением про сирот; после seed второй прогон — «All migrations applied.»

- [ ] **Step 5: Стенд — проверить очистку metadata**

Run: `psql "host=$POSTGRES_HOST dbname=$POSTGRES_DB user=$POSTGRES_USER" -c "SELECT count(*) FROM docs WHERE metadata ?| array['Наименование','МКБ-10'];"`
Expected: `0`.

- [ ] **Step 6: Стенд — полный pytest**

Run: `pytest -v`
Expected: PASS (весь набор против БД и LLM).

- [ ] **Step 7: Финальный коммит (если были правки на стенде)**

```bash
git add -A
git commit -m "chore: guidelines-таблица — прогон стенда зелёный" || echo "нет изменений"
```

---

## Self-Review

**Покрытие спеки:**
- §1 схема → Task 2 ✓
- §2 модель/storage/ingestion/seed → Tasks 1, 3, 4 ✓
- §3 читатели: doc.py/docs_storage JOIN → Task 5; decider/icd_prefix_picker (граница с LLM) → Task 6; clinic_recs → Task 7; icd_check/pipeline → Task 8; result_parser/api/excel → Task 9 ✓
- §4 миграции/seed/бэкфилл + защита порядка → Tasks 2, 10, 11 ✓
- §5 TODO(guidelines-sql) → Task 8 (pipeline age-фильтр помечен); отчёт (Task 9) переведён на `all()` с загрузкой всего справочника — соответствует «оставить как есть, пометить» из спеки ✓
- §6 тесты + verification-разрыв → тест-шаги в каждом таске, стенд-прогон Task 11 ✓
- Апдейт от пользователя (dict → объекты без костылей): `to_manifest_row()` убран полностью; `decider`/`icd_prefix_picker`/`_render_manifest_table`/`clinic_recs`/`pipeline` работают с `Guideline` от начала до конца; единственная точка перехода в dict — `dataclasses.asdict()` внутри `decider`/`icd_prefix_picker` непосредственно перед `json.dumps` для LLM-промпта, что является границей с внешней системой (LLM API), а не внутренним контрактом ✓

**Типы/имена согласованы:** `Guideline` (`file_id`/`mkb`/`age_category`/`name`) едины во всех тасках; `GuidelinesStorage` методы (`all`/`find_by_code`/`find_by_prefix`/`upsert_many`/`get`) названы одинаково в Tasks 3/7/8/9; `decide_file_id`/`IcdPrefixPicker.pick` типизированы `list[Guideline]` (Task 6) и вызываются с этим типом в Task 7; `_render_manifest_table`/`check_icd_codes` типизированы `list[Guideline]` (Task 8) и заполняются из `GuidelinesStorage.all()` там же; `build_manifest_meta` (Task 9) остаётся на dict-выходе намеренно — это формат, ожидаемый `parse_diagnosis`, отдельный от «манифест как dict-строка», трогать не требовалось; `Doc` поля `name`/`mkb`/`age_category` + алиасы `g_name`/`g_mkb`/`g_age_category` согласованы между Task 5 SELECT и `_row_to_doc`.

**Изменённый порядок тасков:** Task 6 (decider/icd_prefix_picker) теперь идёт ДО Task 7 (clinic_recs), потому что `clinic_recs.pick_recs` вызывает оба и должен сразу передавать им `Guideline` — без этого порядка Task 7 пришлось бы писать против ещё dict-based сигнатур и переписывать заново.
```