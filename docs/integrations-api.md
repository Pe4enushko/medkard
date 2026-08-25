# Интеграции и pull-API: 1С, FTP, FastAPI, организации, деплой

> Актуально на 2026-07-16, ветка `release`. Сверено с кодом.

## Два контура выдачи результатов

1. **Push:** `audit-one-c-period.py` → 1С → аудит → `done_cards` → Excel на диске → FTP.
   На проде запускается **по cron** (crontab на стенде, вне репозитория); повторные
   запуски безопасны (дедуп + кеш + идемпотентный Excel).
2. **Pull:** внешнее интегрирующее приложение по HTTP тянет из `done_cards`
   (`/visits/check|pull|export`), Bearer API-ключ + `?org=`, трафик — приватный
   WireGuard-туннель.

Оба контура читают одну таблицу и общий парсер `reporting/result_parser.py`.

## Источник данных: 1С — `src/integrations/one_c.py`

- `OneCClient` — базовый; `AlenkaOneCClient` (env `ALENKA_ONE_C_*`, пароль обязателен),
  `MdsOneCClient` (env `MDS_ONE_C_*`, `requires_password=False`).
- HTTP GET c Basic-auth (`base64(login:password)`), `Accept: application/json`,
  query `datebegin`/`dateend` в формате `DD.MM.YYYY`, `urllib.request`, таймаут из env
  (дефолт 15 с, в `.env.example` — 25).
- Ответ возвращается как есть (`json.loads`); сетевые ошибки → `RuntimeError`.
- Данные приходят **обезличенными** (контур данных: вход из 1С обезличен, модели локальные).
- Диагностика: `scripts/checks/check-one-c-curl.sh {Alenka|MDS} FROM TO` — воспроизводит
  запрос curl'ом с теми же env.

### Кеш `data_snapshots/`

Реализован в `scripts/audit-one-c-period.py` (не в клиенте):
`data_snapshots/one_c_<ORG>_<from>_to_<to>.json` — есть файл → читается из кеша,
нет → запрос к 1С + запись. `--date` перепроигрывает закешированный день.

### Формат входного JSON

`{"appointments": [<визит>, …]}` или голый список; один визит:

```jsonc
{
  "Прием":        { "GUID": "…",            // card_guid: дедуп
                    "DATE": "DD.MM.YYYY" },  // дата визита (парсится в SQL)
  "Пациент":      { "AGE"/"Возраст": 45, "SEX": "…", … },   // обезличено
  "Врач":         { "SPECIALIZATION": "…" },
  "Диагнозы":     [ { "КодМКБ": "J06.9", "НаименованиеМКБ": "…",
                      "Детализация": "…", "ВыявленВпервые": "…" } ],
  "ДанныеОсмотра":[ { "Параметр": "Жалобы…", "Значение": "…" }, … ],  // порядок произвольный
  "Услуги":       [ { "Наименование": "…", "Код": "B01.058.001",
                      "Артикул": "…", "КодЕГИСЗ": "…", "УИДЕГИСЗ": "…" } ]
}
```

## FTP — `src/integrations/ftp.py`

- Креды из файла `key=value` (`ip, port, username, password` обязательны).
- `upload(local, filename, creds)` → пассивный обычный FTP (без TLS), путь
  `/YYYY/MM/<filename>` по **текущей дате запуска** (org — только в имени файла).
- Вызывают: `audit-one-c-period.py --ftpcreds` и `scripts/operator/send_report_ftp.py`.

## Pull-API — `src/api/` (FastAPI)

Назначение: единственный внешний клиент (интегрирующее приложение) забирает отчёты.
`pull` отдаёт **сырые байты xlsx** — клиент прогоняет файл через свой ingestion;
`check` — дешёвая сверка числа строк; `export` — инкрементальная выгрузка сырых
строк для аналитической реплики engine. Браузерного контура нет (нет CORS),
сервис публикуется только на WireGuard-IP.

`create_app()` (`app.py`) — фабрика, подключает единственный роутер `cards`.

### Эндпоинты (`src/api/routes/cards.py`, префикс `/cards`)

| Эндпоинт | Параметры | Ответ | Ошибки |
|---|---|---|---|
| `GET /visits/check` | `date` (ISO), `org` | JSON `{date, count}` — число аудированных (не ignored/broken) карт за дату визита | 401/403/404 auth, 422 |
| `GET /visits/pull` | `date`, `org` | xlsx-байты, `Content-Disposition: attachment; filename="report_<org>_<date>.xlsx"` | **404 если карт нет**; 401/403/422 |
| `GET /visits/export` | `since` (ISO ts, опц.), `limit` (0=∞), `cursor` (OFFSET), `org` | JSON-массив сырых строк done_cards: `card_guid, card_data, formal_result, diag_result, icd_check_result, updated_at` (контракт закреплён тестом; token_count/org_id исключены) | 401/403/422 |

`export` сортирует по `updated_at, card_guid` и фильтрует `updated_at > since` —
курсор — триггерная колонка 022 (обновляется на любой апдейт строки).

### Аутентификация — `src/api/auth.py`

`Authorization: Bearer <ключ>` + обязательный `?org=<имя>`:

1. Резолв организации case-insensitive → нет → **404** `Unknown organization`.
2. `is_key_authorized_for_org(raw, org_id)` — ключ активен И заскоуплен на org.
3. Не прошло: ключ валиден, но не для этой org → **403**; невалиден/отозван → **401**.

Ключи: формат `medkard_<token_urlsafe(32)>`, в БД — только SHA-256 (`key_hash`) +
`key_prefix`; модель «ключ = доверенное приложение», scope — M:M на организации
(`api_key_organizations`). Управление:

```
python scripts/operator/create-api-key.py "<label>" --orgs Alenka MDS   # печатает ключ ОДИН раз
python scripts/operator/revoke-api-key.py <uuid | сырой ключ>            # мягкий revoke (revoked_at)
```

## Отчётность — `src/reporting/`

- `result_parser.py` — **единственное** место маппинга JSONB-колонок done_cards →
  датаклассы (`parse_formal`, `parse_diagnosis`, `parse_icd_check`,
  `build_manifest_meta`). Общий для Excel-экспорта и API.
- `api_formatter.py` — `ApiFormatter`: `check` (count по дате визита), `make_xlsx`
  (workbook в памяти через `parsers/excel.build_workbook_bytes`, тот же layout, что и
  дисковые отчёты), `export` (курсорная выгрузка по `updated_at`). Всё жёстко
  scoped `organization_id` и `ignored=FALSE AND broken=FALSE`.
- Скрипты: `create_report.py` (`--from/--to/--org`, диапазон инклюзивный, файл
  `report_<org>_<from>_to_<to>.xlsx`), `send_report_ftp.py` (то же + FTP),
  `metrics.py ORG [--csv]` (читает вьюху `done_cards_metrics`).

```
1С JSON → AuditPipeline → done_cards
  ├─ [диск] ExcelFormatter.export_period → .xlsx → ftp.upload → /YYYY/MM/
  ├─ [API]  ApiFormatter.make_xlsx → байты xlsx (pull)
  ├─ [API]  ApiFormatter.export → сырые строки (реплика engine)
  └─ [метрики] done_cards_metrics → scripts/checks/metrics.py
```

## Мультиарендность

Организации — `organizations`; каждая `done_cards`-строка привязана к
`organization_id`; все отчётные/API-запросы жёстко scoped по нему. Имена организаций
в скриптах фиксированы `choices=("Alenka","MDS")`. Per-org фильтрация входных карт —
`filterconfig.json` (Alenka: IcdFilter Z11.1/Z11.8; MDS: KDLFilter + AnalysisFilter).

## Деплой

- **Dockerfile**: python:3.11-slim + uv + `requirements-api.txt` (минимальный набор,
  без LLM/RAG); копируются только `src/` и `resources/manifest.csv`;
  CMD `uvicorn api.app:create_app --factory --port 8000`. В образе — **только pull-API**;
  аудит, ингест и миграции запускаются вне контейнера.
- **docker-compose.yml**: один сервис `api`, `env_file: .env`,
  порт **`${WG_BIND_IP}:8000:8000`** — только на WireGuard-интерфейсе. Postgres внешний.
  ⚠ `POSTGRES_HOST=localhost` изнутри контейнера не достучится до хостового Postgres.
- asyncpg/pgvector в requirements-api — только из-за жадного импорта
  `storage/__init__.py` (API их не использует); openpyxl — для xlsx-ответа.

## Замечание про `egisz-mis-upload-guide.md`

Этот документ — доменное исследование (как медкарты попадают в ЕГИСЗ: РЭМД, СЭМД,
СМЭВ и стратегии интеграции), **не спецификация** данного API. Реализованная
интеграция — приватный pull-API поверх WireGuard, минуя контур ЕГИСЗ.
