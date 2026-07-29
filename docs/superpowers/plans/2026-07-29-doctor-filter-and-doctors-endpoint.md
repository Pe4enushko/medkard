# Medkard: фильтр по коду врача и список врачей — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Опциональный `doctor_code`-фильтр в `GET /visits/pull` (с xlsx-заглушкой вместо 404) и новый авторизованный `GET /visits/doctors` со списком врачей организации.

**Architecture:** Фильтр вставляется в общий CTE `_VISIT_DATE_CTE` (`src/reporting/api_formatter.py`), поэтому 404-проверка (`check`) и генерация книги (`make_xlsx`) отфильтровываются согласованно. Список врачей — `DISTINCT ON` по `card_data->'Прием'->>'Врач_код'` (full-scan по организации — осознанно, объёмы позволяют). Роуты остаются тонкими (`src/api/routes/cards.py`, префикс уже `/visits`).

**Tech Stack:** FastAPI, psycopg3 (`BaseStorage`), openpyxl, pytest (integration через `TestClient` + реальный Postgres из `.env`).

**Спека:** `docs/superpowers/specs/2026-07-29-medcheck-doctor-mapping-design.md` в ветке `medcheck-doctor-map` репозитория engine (worktree `/home/savoy/projects/engine-medcheck-doctor-map`).

## Global Constraints

- Ветка `medkard-doctor-reports` (worktree `/home/savoy/projects/medkard-doctor-reports`), база `origin/release` (e60ad16 — префикс роутера уже `/visits`).
- Тесты гонять точечно: `pytest tests/<файл> -v` (pytest.ini: `pythonpath=src`, asyncio_mode=auto). Интеграционные тесты ходят в Postgres из `.env` и требуют организаций `Alenka` и `MDS` в таблице `organizations`.
- Поле кода врача в данных: `card_data -> 'Прием' ->> 'Врач_код'`; ФИО: `card_data -> 'Прием' ->> 'Врач'`. НЕ путать с верхнеуровневым блоком `Врач` (там только `SPECIALIZATION`).
- Имя query-параметра — ровно `doctor_code`; текст заглушки — ровно «За {DD.MM.YYYY} приёмов врача с кодом {код} не обнаружено».
- Никаких новых индексов и миграций. Файл роутов остаётся `cards.py` (переименование файла — не наша тема).
- Пушить нельзя; коммитить после каждой задачи.

---

### Task 1: Ретаргет существующих тестов на `/visits`

Коммит e60ad16 переименовал только префикс роутера; `tests/test_cards_api.py` всё ещё ходит на `/cards/*` и падает 404-ми. Механическая правка путей.

**Files:**
- Modify: `tests/test_cards_api.py` (все литералы `"/cards/...` → `"/visits/...`; строки 90, 97, 102, 107, 112, 117, 126–128, 132, 137, 140, 156–157, 160, 169, 175)

**Interfaces:**
- Consumes: существующие фикстуры `client`, `test_key`, `alenka_only_key` (не меняются).
- Produces: зелёный baseline для последующих задач; фикстуры этого файла остаются источником паттерна для нового тест-файла Task 2.

- [ ] **Step 1: Убедиться, что тесты падают на /cards**

Run: `pytest tests/test_cards_api.py -v -x`
Expected: FAIL — 404 от TestClient (роут `/cards/check` больше не существует).

- [ ] **Step 2: Заменить пути**

Во всех строках файла заменить подстроку `"/cards/` на `"/visits/`. Ничего больше не менять.

- [ ] **Step 3: Прогнать тесты**

Run: `pytest tests/test_cards_api.py -v`
Expected: PASS (все 10).

- [ ] **Step 4: Commit**

```bash
git add tests/test_cards_api.py
git commit -m "test: retarget pull-API tests to the renamed /visits root"
```

---

### Task 2: `doctor_code`-фильтр в reader/formatter

**Files:**
- Modify: `src/reporting/api_formatter.py` (CTE строки 24–33; `count_by_date` 44–51; `fetch_by_date` 53–64; `ApiFormatter.check` 138–140; `make_xlsx` 142–159)
- Test: `tests/test_visits_doctor_filter.py` (новый)

**Interfaces:**
- Consumes: `BaseStorage` (psycopg3-пул, `dict_row`), существующий `_VISIT_DATE_CTE`.
- Produces: `ApiFormatter.check(visit_date: date, organization_id: str, doctor_code: str | None = None) -> int` и `ApiFormatter.make_xlsx(visit_date: date, organization_id: str, doctor_code: str | None = None) -> bytes`. Task 4 зовёт их из роута; фикстура `_seed_cards` этого файла переиспользуется в Task 4 и Task 5.

- [ ] **Step 1: Написать падающий тест с фикстурой карт**

```python
"""Integration tests for the doctor_code filter and /visits/doctors.

Seeds its own done_cards rows for MDS on a far-future date (2044-01-01) so
existing stand data can't collide, and deletes them afterwards.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from reporting.api_formatter import ApiFormatter
from storage.base import BaseStorage
from storage.organizations_storage import OrganizationsStorage

FIXTURE_DATE = date(2044, 1, 1)          # DD.MM.YYYY в карте: 01.01.2044
DOC_A = "00001"
DOC_B = "00002"


class _CardsWriter(BaseStorage):
    async def insert_card(self, guid: str, card_data: dict[str, Any], org_id: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "INSERT INTO done_cards (card_guid, card_data, formal_result, diag_result,"
                " icd_check_result, ignored, broken, status, organization_id)"
                " VALUES (%(guid)s, %(data)s::jsonb, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb,"
                " FALSE, FALSE, 'done', %(org)s::uuid)",
                {"guid": guid, "data": json.dumps(card_data, ensure_ascii=False), "org": org_id},
            )

    async def delete_cards(self, guids: list[str]) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "DELETE FROM done_cards WHERE card_guid = ANY(%(guids)s)", {"guids": guids}
            )


def _card(doctor_code: str | None, doctor_name: str | None) -> dict[str, Any]:
    priem: dict[str, Any] = {"GUID": str(uuid.uuid4()), "DATE": "01.01.2044"}
    if doctor_name is not None:
        priem["Врач"] = doctor_name
    if doctor_code is not None:
        priem["Врач_код"] = doctor_code
    return {
        "Прием": priem,
        "Врач": {"SPECIALIZATION": "Невролог"},
        "Пациент": {"CODE": "Т-000001", "GENDER": "Мужской", "AGE": 40},
        "Диагнозы": [],
    }


@pytest.fixture
async def mds_org_id() -> str:
    async with OrganizationsStorage() as organizations:
        return await organizations.get_id_by_name("MDS")


@pytest.fixture
async def seeded_cards(mds_org_id: str):
    """2 карты врача 00001, 1 карта 00002, 1 без Врач_код — все на 2044-01-01."""
    cards = [
        _card(DOC_A, "Иванов Иван Иванович"),
        _card(DOC_A, "Иванов Иван Иванович"),
        _card(DOC_B, "Петрова Анна Сергеевна"),
        _card(None, None),
    ]
    guids = [c["Прием"]["GUID"].lower() for c in cards]
    async with _CardsWriter() as writer:
        for guid, card in zip(guids, cards):
            await writer.insert_card(guid, card, mds_org_id)
    yield mds_org_id
    async with _CardsWriter() as writer:
        await writer.delete_cards(guids)


async def test_check_counts_only_that_doctor(seeded_cards: str):
    async with ApiFormatter() as formatter:
        assert await formatter.check(FIXTURE_DATE, seeded_cards, DOC_A) == 2
        assert await formatter.check(FIXTURE_DATE, seeded_cards, DOC_B) == 1
        assert await formatter.check(FIXTURE_DATE, seeded_cards, "99999") == 0


async def test_check_without_filter_counts_all(seeded_cards: str):
    async with ApiFormatter() as formatter:
        assert await formatter.check(FIXTURE_DATE, seeded_cards) == 4


async def test_make_xlsx_filters_rows(seeded_cards: str):
    import io
    import openpyxl

    async with ApiFormatter() as formatter:
        content = await formatter.make_xlsx(FIXTURE_DATE, seeded_cards, DOC_A)
    ws = openpyxl.load_workbook(io.BytesIO(content)).active
    assert ws.max_row - 1 == 2  # минус заголовок
```

- [ ] **Step 2: Прогнать — убедиться, что падает**

Run: `pytest tests/test_visits_doctor_filter.py -v`
Expected: FAIL — `TypeError: check() takes 3 positional arguments but 4 were given`.

- [ ] **Step 3: Реализовать фильтр**

В `src/reporting/api_formatter.py` заменить `_VISIT_DATE_CTE`:

```python
_VISIT_DATE_CTE = (
    "WITH cards AS ("
    "  SELECT id, card_guid, card_data, formal_result, diag_result, icd_check_result, "
    "         to_date(card_data -> 'Прием' ->> 'DATE', 'DD.MM.YYYY') AS visit_date "
    "  FROM done_cards "
    "  WHERE ignored = FALSE "
    "    AND broken = FALSE "
    "    AND organization_id = %(org_id)s::uuid "
    "    AND (%(doctor_code)s::text IS NULL "
    "         OR card_data -> 'Прием' ->> 'Врач_код' = %(doctor_code)s)"
    ") "
)
```

`count_by_date` и `fetch_by_date` получают параметр и кладут его в `params`:

```python
    async def count_by_date(
        self, visit_date: date, organization_id: str, doctor_code: str | None = None
    ) -> int:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                _VISIT_DATE_CTE + "SELECT count(*) AS n FROM cards WHERE visit_date = %(date)s::date",
                {"org_id": organization_id, "date": visit_date, "doctor_code": doctor_code},
            )
            row = await cur.fetchone()
        return row["n"]
```

В `fetch_by_date` — та же сигнатура, `params` дополняется `"doctor_code": doctor_code`. В `ApiFormatter`:

```python
    async def check(
        self, visit_date: date, organization_id: str, doctor_code: str | None = None
    ) -> int:
        """Return the number of audited cards for *organization_id* on *visit_date*,
        optionally narrowed to one doctor (Прием.Врач_код)."""
        return await self._reader.count_by_date(visit_date, organization_id, doctor_code)
```

`make_xlsx` — аналогично: параметр `doctor_code: str | None = None`, пробрасывается в `fetch_by_date`.

- [ ] **Step 4: Прогнать тесты**

Run: `pytest tests/test_visits_doctor_filter.py tests/test_cards_api.py -v`
Expected: PASS (новые + старые — фильтр по умолчанию `None` ничего не меняет).

- [ ] **Step 5: Commit**

```bash
git add src/reporting/api_formatter.py tests/test_visits_doctor_filter.py
git commit -m "feat: optional doctor_code filter in the visit-date CTE"
```

---

### Task 3: xlsx-заглушка «приёмов не обнаружено»

**Files:**
- Modify: `src/parsers/excel.py` (новая функция после `build_workbook_bytes`, строка ~218)
- Test: `tests/test_excel_empty_report.py` (новый)

**Interfaces:**
- Consumes: `openpyxl.Workbook`, `io` (уже импортированы в модуле).
- Produces: `build_empty_report_bytes(message: str) -> bytes` — Task 4 зовёт её из роута.

- [ ] **Step 1: Написать падающий тест**

```python
"""build_empty_report_bytes: single-cell placeholder workbook."""
import io

import openpyxl

from parsers.excel import build_empty_report_bytes


def test_empty_report_has_message_in_a1_and_nothing_else():
    message = "За 01.01.2044 приёмов врача с кодом 00001 не обнаружено"
    content = build_empty_report_bytes(message)
    ws = openpyxl.load_workbook(io.BytesIO(content)).active
    assert ws.cell(row=1, column=1).value == message
    assert ws.max_row == 1 and ws.max_column == 1
```

- [ ] **Step 2: Прогнать — убедиться, что падает**

Run: `pytest tests/test_excel_empty_report.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_empty_report_bytes'`.

- [ ] **Step 3: Реализовать**

В `src/parsers/excel.py`, сразу после `build_workbook_bytes`:

```python
def build_empty_report_bytes(message: str) -> bytes:
    """Single-cell workbook stating that a filtered report has no visits.

    Used by /visits/pull with a doctor_code filter: the integrating service
    expects a file either way, so absence is stated inside the workbook
    instead of a 404 (the unfiltered pull keeps its 404 contract).
    """
    wb = Workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value=message)
    ws.column_dimensions["A"].width = 100
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()
```

- [ ] **Step 4: Прогнать тест**

Run: `pytest tests/test_excel_empty_report.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/parsers/excel.py tests/test_excel_empty_report.py
git commit -m "feat: single-cell placeholder workbook for empty filtered reports"
```

---

### Task 4: роут `/visits/pull` — параметр, заглушка, имя файла

**Files:**
- Modify: `src/api/routes/cards.py` (роут `pull`, строки 45–65)
- Test: `tests/test_visits_doctor_filter.py` (дополнить)

**Interfaces:**
- Consumes: `ApiFormatter.check/make_xlsx` c `doctor_code` (Task 2), `build_empty_report_bytes` (Task 3), фикстура `seeded_cards` (Task 2).
- Produces: контракт API — `GET /visits/pull?date&org&doctor_code`; при фильтре имя файла `report_{org}_{date}_doc{code}.xlsx`, при нуле карт с фильтром — 200 + заглушка. Его потребляет engine-ветка `medcheck-doctor-map`.

- [ ] **Step 1: Дописать падающие тесты**

В `tests/test_visits_doctor_filter.py` добавить (фикстуры `client`/`test_key` — по образцу `tests/test_cards_api.py`, но ключ скоупится только на MDS):

```python
import io

import openpyxl
from fastapi.testclient import TestClient

from api.app import create_app
from storage.api_keys_storage import ApiKeysStorage


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
async def test_key(mds_org_id: str) -> str:
    raw_key = f"medkard_test_{uuid.uuid4().hex}"
    async with ApiKeysStorage() as api_keys:
        key_id = await api_keys.create_key("pytest-doctor-filter", raw_key, [mds_org_id])
    yield raw_key
    async with ApiKeysStorage() as api_keys:
        await api_keys.revoke_key(key_id)


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_pull_with_doctor_code_filters_and_renames(client, test_key, seeded_cards):
    resp = client.get(
        f"/visits/pull?date=2044-01-01&org=MDS&doctor_code={DOC_A}", headers=_auth(test_key)
    )
    assert resp.status_code == 200
    assert 'filename="report_MDS_2044-01-01_doc00001.xlsx"' in resp.headers["content-disposition"]
    ws = openpyxl.load_workbook(io.BytesIO(resp.content)).active
    assert ws.max_row - 1 == 2


def test_pull_unknown_doctor_returns_placeholder_not_404(client, test_key, seeded_cards):
    resp = client.get(
        "/visits/pull?date=2044-01-01&org=MDS&doctor_code=99999", headers=_auth(test_key)
    )
    assert resp.status_code == 200
    ws = openpyxl.load_workbook(io.BytesIO(resp.content)).active
    assert ws.cell(row=1, column=1).value == "За 01.01.2044 приёмов врача с кодом 99999 не обнаружено"
    assert ws.max_row == 1


def test_pull_without_filter_keeps_404_contract(client, test_key):
    resp = client.get("/visits/pull?date=1999-01-01&org=MDS", headers=_auth(test_key))
    assert resp.status_code == 404
```

- [ ] **Step 2: Прогнать — убедиться, что падают**

Run: `pytest tests/test_visits_doctor_filter.py -v`
Expected: новые тесты FAIL (нет параметра → фильтр игнорируется: не тот счёт строк / нет `_doc` в имени / 404 вместо заглушки); тесты Task 2 — PASS.

- [ ] **Step 3: Реализовать роут**

В `src/api/routes/cards.py` заменить `pull` (импорт заглушки добавить к остальным: `from parsers.excel import build_empty_report_bytes`):

```python
@router.get("/pull")
async def pull(
    date_: date = Query(..., alias="date"),
    doctor_code: str | None = Query(default=None, min_length=1),
    org_access: tuple[str, str] = Depends(require_org_access),
) -> Response:
    org_id, org_name = org_access
    async with ApiFormatter() as formatter:
        count = await formatter.check(date_, org_id, doctor_code)
        if count == 0 and doctor_code is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No visits for {org_name} on {date_.isoformat()}",
            )
        if count == 0:
            # Personal report: the caller delivers a file per doctor either way,
            # so "no visits" is stated inside the workbook, not as a 404.
            xlsx_bytes = build_empty_report_bytes(
                f"За {date_.strftime('%d.%m.%Y')} приёмов врача с кодом {doctor_code} не обнаружено"
            )
        else:
            xlsx_bytes = await formatter.make_xlsx(date_, org_id, doctor_code)

    suffix = f"_doc{doctor_code}" if doctor_code else ""
    filename = f"report_{org_name}_{date_.isoformat()}{suffix}.xlsx"
    return Response(
        content=xlsx_bytes,
        media_type=_XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
```

- [ ] **Step 4: Прогнать тесты**

Run: `pytest tests/test_visits_doctor_filter.py tests/test_cards_api.py -v`
Expected: PASS (все).

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/cards.py tests/test_visits_doctor_filter.py
git commit -m "feat: doctor_code filter on /visits/pull with placeholder workbook"
```

---

### Task 5: `GET /visits/doctors`

**Files:**
- Modify: `src/api/models.py` (новая модель), `src/reporting/api_formatter.py` (метод reader + метод formatter), `src/api/routes/cards.py` (новый роут после `check_updates`)
- Test: `tests/test_visits_doctor_filter.py` (дополнить)

**Interfaces:**
- Consumes: `require_org_access`, `_ApiCardsReader`, фикстуры `seeded_cards`/`client`/`test_key`.
- Produces: `DoctorEntry(code: str, name: str)`; `ApiFormatter.doctors(organization_id: str) -> list[dict]`; контракт `GET /visits/doctors?org=` → `[{"code","name"}]`, сортировка по имени. Его потребляет операторский скрипт engine-ветки.

- [ ] **Step 1: Дописать падающие тесты**

```python
def test_doctors_lists_unique_codes_sorted_by_name(client, test_key, seeded_cards):
    resp = client.get("/visits/doctors?org=MDS", headers=_auth(test_key))
    assert resp.status_code == 200
    doctors = resp.json()
    ours = [d for d in doctors if d["code"] in (DOC_A, DOC_B)]
    assert ours == [
        {"code": DOC_A, "name": "Иванов Иван Иванович"},
        {"code": DOC_B, "name": "Петрова Анна Сергеевна"},
    ]
    # карта без Врач_код не рождает пустого врача
    assert all(d["code"] for d in doctors)


def test_doctors_requires_auth(client):
    resp = client.get("/visits/doctors?org=MDS")
    assert resp.status_code in (401, 403)
```

- [ ] **Step 2: Прогнать — убедиться, что падают**

Run: `pytest tests/test_visits_doctor_filter.py -v`
Expected: новые FAIL с 404 (роут не существует).

- [ ] **Step 3: Реализовать**

`src/api/models.py`:

```python
class DoctorEntry(BaseModel):
    code: str
    name: str
```

`src/reporting/api_formatter.py`, в `_ApiCardsReader` (после `fetch_changed`):

```python
    async def fetch_doctors(self, organization_id: str) -> list[dict[str, Any]]:
        """Unique doctors of an org from card data: Прием.Врач_код + Прием.Врач.

        DISTINCT ON keeps the freshest card's spelling of the name per code.
        broken cards are excluded; ignored are NOT — an ignored card still
        names a real doctor. Full JSONB scan per org — accepted at current
        volumes (no card_data indexes exist).
        """
        query = (
            "SELECT code, name FROM ("
            "  SELECT DISTINCT ON (card_data -> 'Прием' ->> 'Врач_код')"
            "         card_data -> 'Прием' ->> 'Врач_код'          AS code,"
            "         COALESCE(card_data -> 'Прием' ->> 'Врач', '') AS name"
            "  FROM done_cards"
            "  WHERE organization_id = %(org_id)s::uuid"
            "    AND broken = FALSE"
            "    AND COALESCE(card_data -> 'Прием' ->> 'Врач_код', '') <> ''"
            "  ORDER BY card_data -> 'Прием' ->> 'Врач_код', updated_at DESC"
            ") AS d ORDER BY name, code"
        )
        async with self._pool.connection() as conn:
            cur = await conn.execute(query, {"org_id": organization_id})
            return await cur.fetchall()
```

В `ApiFormatter`:

```python
    async def doctors(self, organization_id: str) -> list[dict[str, Any]]:
        """Unique (code, name) doctors of the org, sorted by name."""
        return await self._reader.fetch_doctors(organization_id)
```

`src/api/routes/cards.py` (импорт `DoctorEntry` из `api.models`; роут после `check_updates`):

```python
@router.get("/doctors", response_model=list[DoctorEntry])
async def doctors(
    org_access: tuple[str, str] = Depends(require_org_access),
) -> list[DoctorEntry]:
    org_id, _ = org_access
    async with ApiFormatter() as formatter:
        rows = await formatter.doctors(org_id)
    return [DoctorEntry(code=row["code"], name=row["name"]) for row in rows]
```

- [ ] **Step 4: Прогнать тесты**

Run: `pytest tests/test_visits_doctor_filter.py tests/test_cards_api.py -v`
Expected: PASS (все).

- [ ] **Step 5: Commit**

```bash
git add src/api/models.py src/reporting/api_formatter.py src/api/routes/cards.py tests/test_visits_doctor_filter.py
git commit -m "feat: GET /visits/doctors — unique doctors of an org"
```

---

### Task 6: документация API

**Files:**
- Modify: `docs/cards-api.md`

**Interfaces:**
- Consumes: контракты Task 4/5.
- Produces: актуальный документ; кода не производит.

- [ ] **Step 1: Обновить документ**

1. Заголовок и все пути `/cards/*` → `/visits/*` (в первой строке: `# Visits API (\`/visits\`)`); упоминание `scripts/test-cards-route.sh` оставить как есть.
2. В секции `GET /visits/pull` добавить параметр и поведение:

```markdown
Опциональный параметр `doctor_code` — код врача из 1С (`Прием.Врач_код`):
отчёт сужается до карт одного врача, имя файла получает суффикс
`_doc<код>` (`report_<org>_<date>_doc<код>.xlsx`). При нуле карт с фильтром
возвращается **200** и одноклеточный xlsx «За <дата> приёмов врача с кодом
<код> не обнаружено» — интегрирующийся сервис доставляет врачу файл в любом
случае. Без фильтра контракт прежний: `404` при нуле карт.
```

3. Новая секция после `GET /visits/check_updates`:

```markdown
## GET /visits/doctors

Уникальные врачи организации по данным карт (`Прием.Врач_код` +
`Прием.Врач`). Параметры запроса: `org`. Карты с `broken = TRUE` и без
`Врач_код` не учитываются; при разных написаниях ФИО у одного кода берётся
самая свежая карта. Сортировка по имени.

​```json
[{"code": "00012", "name": "Губарева Елена Александровна"}]
​```
```

- [ ] **Step 2: Commit**

```bash
git add docs/cards-api.md
git commit -m "docs: /visits rename, doctor_code filter and /visits/doctors"
```
