# E2E-хелперы для времянок + smoke-тест push_log

**Дата:** 2026-08-20 · **Статус:** утверждён · **Ветка:** `push-log-e2e-tests` (поверх `audit-overwrite-journal`)
**Связано:** `scripts/smoke-cards-push.sh`, `scripts/smoke-push-check-updates.py`, `scripts/create-api-key.py`,
`migrations/027_audit_overwrite_journal.sql` (`push_log`, `push_metrics_by_date`), `docker-compose.yml`, `.env.example`

## Зачем

Существующие smoke-тесты (`scripts/smoke-cards-push.sh`, `scripts/smoke-push-check-updates.py`) каждый
сам создаёт и убирает за собой временную организацию/API-ключ/карту — логика дублируется между
скриптами (см. класс `_Fixtures` внутри `smoke-push-check-updates.py`). Нужна общая переиспользуемая
инфраструктура для e2e-тестов, которые создают времянки (организация, API-ключ, карта) и гарантированно
убирают их за собой, плюс новый smoke-тест конкретно для фичи push_log (`migrations/027_audit_overwrite_journal.sql`
с этой ветки — журналирует каждый push с флагом `overrode_audit` и вьюхой `push_metrics_by_date`
для разбивки по датам).

Запуск — против уже поднятого API (докер-контейнер или локальный `uvicorn`), опция `local` — просто
короткий алиас на `http://localhost:{API_PORT}`, без автозапуска сервера.

## Решение

### 1. `.env` / `.env.example` — новая переменная `API_PORT`

Порт, на который docker-compose публикует контейнер, сейчас захардкожен в `docker-compose.yml`
(`13742`) и нигде не вынесен в конфиг. Добавляем:

```env
# Docker: bind the published pull-API port to this address only (e.g. the host's
# WireGuard interface IP) so the API is unreachable except through the tunnel.
WG_BIND_IP=
API_PORT=8000
```

`API_PORT=8000` в `.env.example` — дефолт для локальной разработки без докера (`uvicorn` слушает
`8000` по умолчанию). В реальном `.env` пользователь ставит своё значение сам (например, `13742` —
внешний порт, на который смотрит WireGuard-туннель).

### 2. `docker-compose.yml` — порт из `.env`

```yaml
services:
  api:
    build: .
    env_file: .env
    ports:
      - "${WG_BIND_IP}:${API_PORT}:8000"
    restart: unless-stopped
```

Внешний (host) порт теперь берётся из `.env` вместо `13742`, зашитого в файл. Внутренний порт
контейнера остаётся `8000` — это порт, на котором `uvicorn` слушает внутри контейнера
(`Dockerfile:15`), е не меняется.

### 3. `e2e/` — новый top-level каталог

```
e2e/
  __init__.py
  tests/
    __init__.py
    helpers/
      __init__.py
      organizations.py
      api_keys.py
      cards.py
    test_push_log_smoke.py
```

Каталог на верхнем уровне репозитория (не под `src/`, не под `tests/`) — это e2e-инфраструктура
против живого HTTP API, а не unit/integration-тесты кода на `pythonpath=src` (`pytest.ini`). Все
модули добавляют `ROOT / "src"` в `sys.path` сами при импорте, как это уже делает
`scripts/smoke-push-check-updates.py` — паттерн, а не новое решение.

### 4. `e2e/tests/helpers/organizations.py`

Прямой INSERT/DELETE в `organizations`, минуя API — тестовой организации не нужно проходить через
никакой публичный контракт создания, его и не существует (organizations сейчас создаются только
вручную/миграциями).

```python
class OrganizationFixtures(BaseStorage):
    async def create_org(self, name: str) -> str:
        """INSERT INTO organizations (name) VALUES (...) RETURNING id::text"""

    async def delete_org(self, org_id: str) -> None:
        """DELETE FROM organizations WHERE id = ...::uuid"""
```

Идентично тому, что уже есть в `_Fixtures.create_org`/`delete_org` внутри
`scripts/smoke-push-check-updates.py` — переносится один в один, только в общий модуль.

### 5. `e2e/tests/helpers/api_keys.py`

```python
async def issue_key(label: str, org_id: str) -> tuple[str, str]:
    """Через ApiKeysStorage.create_key — (key_id, raw_key)."""

class ApiKeyFixtures(BaseStorage):
    async def delete_key(self, label: str) -> int:
        """DELETE FROM api_keys WHERE label = ... — по label, не по id (см. ниже)."""

    async def count_key_scopes(self, key_id: str) -> int:
        """Для проверки в teardown, что api_key_organizations действительно каскадно удалились."""
```

По label, а не по id — та же причина, что уже описана в docstring `_Fixtures.delete_keys_by_label`:
`create_key` делает вставку ключа и его org-скоупа двумя отдельными операциями без явной транзакции,
так что если между ними что-то упадёт, у вызывающего кода не будет id, но label он всегда знает
заранее (сам его сгенерировал).

### 6. `e2e/tests/helpers/cards.py`

```python
async def push_card(client: httpx.AsyncClient, base_url: str, org: str, raw_key: str, card: dict) -> httpx.Response:
    """POST {base_url}/visits/push?org={org}, Authorization: Bearer {raw_key}."""

class CardFixtures(BaseStorage):
    async def stage_audited(self, card_guid: str) -> None:
        """
        Прямой UPDATE done_cards: status='done', formal_result = одна фиктивная
        находка (валидный JSON, форма как из FormalStructureResult), ignored=FALSE,
        broken=FALSE — имитирует уже прошедшую формальную проверку без реальных
        LLM-вызовов, чтобы следующий push поверх этой строки дал overrode_audit=true
        в push_log. Аналог шага 6 в scripts/smoke-cards-push.sh, но как переиспользуемый
        хелпер, а не inline SQL внутри теста.
        """

    async def card_row(self, card_guid: str) -> dict | None:
        """SELECT * FROM done_cards WHERE card_guid = ..."""

    async def push_log_rows(self, card_guid: str) -> list[dict]:
        """SELECT * FROM push_log WHERE card_guid = ... ORDER BY pushed_at."""

    async def delete_cards(self, card_guid: str) -> int:
        """DELETE FROM done_cards WHERE card_guid = ..."""

    async def delete_push_log(self, card_guid: str) -> int:
        """DELETE FROM push_log WHERE card_guid = ... — обязательный teardown-шаг,
        которого не хватало в старых тестах done_cards (см. фикс 08bff77/2dd9812
        на этой же ветке — забытый DELETE FROM push_log утекал строки в
        push_metrics_by_date на реальной БД)."""
```

### 7. `e2e/tests/test_push_log_smoke.py`

Самостоятельный скрипт с `argparse`, тот же паттерн, что `scripts/smoke-push-check-updates.py`:
создаёт временные org+key, гоняет сценарий, разбирает в `finally` даже при падении/Ctrl-C, печатает
PASS/FAIL по каждой проверке, ненулевой exit code при провале.

```
python e2e/tests/test_push_log_smoke.py local
python e2e/tests/test_push_log_smoke.py https://medkard.example --keep
```

- `local` (или `localhost`) как позиционный URL-аргумент разворачивается в
  `http://localhost:{API_PORT}` (`API_PORT` читается из `.env` тем же способом, что и `POSTGRES_*`).
  Любая другая строка используется как есть, как полноценный URL — так же, как сейчас работает
  позиционный `url` в `smoke-push-check-updates.py`.
- `--keep` — не убирать за собой, напечатать что осталось (тот же контракт, что уже есть).

Сценарий (проверяет именно `push_log`/`push_metrics_by_date` с этой ветки, не переоткрывает то, что
уже проверяет `scripts/smoke-cards-push.sh` про сам `/visits/push`):

1. Создать org + ключ через хелперы.
2. Push новой карты (уникальный GUID) → 200. В `push_log` для этого guid — 0 строк: первый пуш
   создаёт строку в `done_cards`, это `INSERT`, а не перезапись, триггер не срабатывает
   (`WHEN (NEW.status = 'pending')` есть, но триггер `BEFORE UPDATE`, не `BEFORE INSERT`).
3. Повторный push той же карты (она всё ещё `pending`, аудит-колонки пустые) → в `push_log`
   появляется ровно одна новая строка с `overrode_audit = false`.
4. `stage_audited(guid)` — карта переводится в `status='done'` с фиктивным `formal_result`.
5. Push поверх аудированной карты → в `push_log` — ещё одна строка, `overrode_audit = true`.
6. Запрос к `push_metrics_by_date` за сегодняшнюю дату и эту организацию: `pushes_total == 2`,
   `pushes_overrode_audit == 1`, `pushes_no_override == 1` (дельта от снятого до теста baseline —
   как в `tests/test_push_log.py::test_multiple_pushes_in_one_day_aggregate_in_metrics_view` на этой
   же ветке, чтобы не падать от параллельно идущих прогонов на общей БД).
7. Teardown в `finally`: `delete_cards`, `delete_push_log`, `delete_key` (по label), `delete_org` —
   всегда, включая падение сценария и Ctrl-C; при `--keep` — пропустить и напечатать, что оставлено
   (org name, raw key, card_guid).

## Не в скоупе

- Автозапуск API-сервера (`uvicorn`/докер) самим скриптом — `local` только формирует URL, сервер
  должен быть поднят заранее вручную, как и раньше.
- Миграция существующих `scripts/smoke-cards-push.sh` и `scripts/smoke-push-check-updates.py` на
  новые общие хелперы — они продолжают работать как есть; общая инфраструктура вводится для нового
  кода, обратная миграция старых скриптов — отдельная задача, если понадобится.
- pytest-обвязка / запуск через `pytest e2e/` — по решению пользователя это самостоятельный скрипт
  с `argparse`, не pytest-тест.
- Проверки самого `/visits/push` (авторизация, 422 на пустышку и т.д.) — уже покрыты
  `scripts/smoke-cards-push.sh`, здесь не дублируются.
