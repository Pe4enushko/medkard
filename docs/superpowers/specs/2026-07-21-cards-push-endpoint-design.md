# API для пуша обновлённых карт от 1С-организаций

**Дата:** 2026-07-21 · **Статус:** утверждён · **Ветка:** `guideline-filename-fallback` (текущая рабочая)
**Связано:** `src/api/routes/cards.py`, `src/storage/done_cards_storage.py`, `src/audit/pipeline.py`,
`migrations/004_done_cards.sql` и последующие миграции done_cards.

## Зачем

Сейчас единственный способ получить карты от 1С-организации — ночной batch-забор
(`scripts/audit-one-c-period.py` → `AuditPipeline.run_batched`), который тянет карты за период и
сразу их аудирует. Организации иногда обновляют уже отправленные карты в течение дня, а мы узнаём
об этом только на следующую ночь (если вообще узнаём — если дата визита выпадает за пределы окна
следующего ночного пулла, обновление никогда не будет переаудировано).

Нужен способ, которым организация может **протолкнуть** к нам одну обновлённую карту сразу, не
дожидаясь ночи. Обновлённая карта сохраняется как есть (сырые данные), без аудита — аудит
результаты предыдущей версии карты становятся неактуальными и должны быть стёрты. Сам аудит
по-прежнему происходит ночью, вместе с обычным пуллом.

Уже существует «pull API» (`src/api`) — FastAPI-сервис в отдельном docker-контейнере
(`docker-compose.yml`, сервис `api`), который организации используют, чтобы забирать готовые
отчёты (`GET /cards/check`, `/cards/pull`, `/cards/export`), с bearer-токеном, скоупнутым на
организации (`api/auth.py::require_org_access`). Пуш обновлённых карт — обратное направление того
же сервиса.

## Решение

### 1. Новая колонка `done_cards.status`

Миграция `025_done_cards_status.sql`:

```sql
ALTER TABLE done_cards
    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'done'));

-- Бэкфилл: все существующие строки уже прошли пайплайн (аудит, ignore или broken).
UPDATE done_cards SET status = 'done';

CREATE INDEX IF NOT EXISTS done_cards_status_idx ON done_cards (status);
```

Семантика:
- `pending` — в `card_data` лежат сырые данные карты, ещё не аудированные (или аудированные, но
  устаревшие после пуша обновления). Все audit-колонки (`formal_result`, `diag_result`,
  `icd_check_result`), а также `ignored`/`broken`/`stacktrace` — сброшены.
- `done` — карта прошла пайплайн: либо полноценно аудирована, либо помечена `ignored`
  (ICD-игнор-лист), либо `broken` (упала с исключением). Дальнейших действий не требует до
  следующего пуша.

### 2. `DoneCardsStorage` — новые методы

`src/storage/done_cards_storage.py`:

- **`upsert_pending(*, card_guid, card_data, organization_id=None) -> str`** — пуш-путь.
  `INSERT ... ON CONFLICT (card_guid) DO UPDATE`, аналогично существующему `upsert()`/
  `upsert_ignored()`. Устанавливает:
  - `card_data = EXCLUDED.card_data` (новые сырые данные)
  - `status = 'pending'`
  - `formal_result = NULL`, `diag_result = NULL`, `icd_check_result = NULL`
  - `ignored = FALSE`, `broken = FALSE`, `stacktrace = NULL`
  - `organization_id = EXCLUDED.organization_id`

  Работает одинаково для нового `card_guid` (первый пуш карты, которую мы раньше не видели) и для
  уже существующего (обновление) — отдельной ветки «создать новую» не нужно.

- **`get_pending(organization_id=None) -> list[dict]`** — возвращает `card_guid` + `card_data` всех
  строк со `status = 'pending'` для организации, для ночного джоба.

Существующий `upsert()` (успешный аудит) дополнительно ставит `status = 'done'`. Существующие
`upsert_ignored()` / `upsert_broken()` также ставят `status = 'done'` — это тоже терминальные
состояния обработки.

### 3. Новый роут — `POST /cards/push`

`src/api/routes/cards.py`, рядом с `check`/`pull`/`export`:

```
POST /cards/push?org=<name>
Authorization: Bearer <key>
Body: JSON одной карты (та же форма, что один элемент батча из 1С — с полем "Прием": {"GUID": ...})
```

- Авторизация — существующий `Depends(require_org_access)`, без нового auth-кода.
- `card_guid` достаётся тем же способом, что и в пайплайне (`Прием.GUID`); если GUID отсутствует —
  `422 Unprocessable Entity`.
- Дальше — без валидации структуры визита (то же доверие к данным, что и у ночного пулла из 1С,
  который тоже не валидирует форму визита на этом этапе).
- Вызывает `DoneCardsStorage.upsert_pending(...)`.
- Ответ: `200 OK`, `{"card_guid": "...", "status": "pending"}`.

Новых Pydantic-моделей для тела запроса не нужно — тело принимается как произвольный JSON (`dict`),
как есть, и передаётся в `card_data`.

### 4. Ночной джоб — подхват pending-карт

В `AuditPipeline.run_batched` (или в вызывающем скрипте `scripts/audit-one-c-period.py`) — рядом с
обычным забором из 1С за период:

- Отдельным запросом получить `DoneCardsStorage.get_pending(organization_id=org_id)` — сырые
  `card_data` карт, ожидающих обработки для этой организации.
- Смешать их со списком визитов из 1С **перед** вызовом `run_batched` (оба списка — обычные визит-
  dict'ы одной формы, `run_batched` их не различает).
- Существующая дедупликация по `card_guid`/`done_guids` внутри `run_batched`/`CardFilter` гарантирует,
  что карта, которая одновременно и пришла из 1С, и была запушена, аудируется один раз.
- По итогам успешного аудита `upsert()` уже сегодня делает `ON CONFLICT DO UPDATE` по всем колонкам
  — добавление `status = 'done'` в этот же `SET` не меняет структуру запроса.

### 5. Контейнер

Отдельный контейнер не нужен — `POST /cards/push` добавляется в тот же FastAPI-сервис `api`
(`src/api`, `docker-compose.yml`), который уже обслуживает `/cards/check|pull|export` за
WireGuard-туннелем. Просто ещё один роут в том же `APIRouter(prefix="/cards")`.

## Не в скоупе

- Дальнейшая ретрансляция обновлённых/аудированных карт другим сервисам — отдельная задача
  («мы будем как-то релеить их дальше», следующий шаг по словам заказчика).
- Валидация структуры визита при пуше (форма `Прием`/`Пациент`/`Диагнозы`/`ДанныеОсмотра`) —
  выполняется, как и сегодня, только на этапе аудита, не на этапе приёма пуша.
- Rate-limiting / защита от повторных пушей одной и той же карты — вне скоупа, пуш идемпотентен
  по `card_guid` и просто перезаписывает.
