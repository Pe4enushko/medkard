# Журнал пушей и метрики перезаписей аудита

**Дата:** 2026-08-20 · **Статус:** утверждён · **Ветка:** `audit-overwrite-journal` (правится на месте)
**Связано:** `migrations/027_audit_overwrite_journal.sql`, `src/storage/done_cards_storage.py::upsert_pending`,
`src/api/routes/visits.py::push`, `src/api/routes/stats.py`, `done_cards_metrics` view

## Зачем

`POST /visits/push` уходит в `DoneCardsStorage.upsert_pending`, чей `ON CONFLICT`-branch сбрасывает
`card_data` и обнуляет все audit-колонки (`formal_result`/`diag_result`/`icd_check_result`), если
строка уже существовала. Если карта уже была проаудирована, вывод LLM пропадает бесследно, без
какого-либо следа, что перезапись вообще произошла.

Нужно знать по каждой организации, сколько раз в конкретную дату она пушила карты вообще, и сколько
из этих пушей затёрли уже готовый аудит (`overrode_audit`) — а сколько были холостыми (карта ещё не
была аудирована, либо была `ignored`/`broken`, либо это не пуш вовсе, а честная переаудит-запись).

На ветке `audit-overwrite-journal` уже было реализовано близкое решение (журнал только для
деструктивных перезаписей с полным сохранением старых `card_data`/результатов, отдельный счётчик
`push_count` на живой строке для общего числа, оргфлаг отключения журнала). У этого подхода есть
разрыв: `push_count` — счётчик без даты каждого инкремента, так что «сколько пушей за дату» можно
посчитать только для деструктивных (у которых есть `overwritten_at` в журнале), а не для всех пушей
целиком. Этот документ описывает объединение обоих механизмов в один — единый датированный журнал
каждого пуша, лёгкий по умолчанию, с местом под payload на будущее.

## Решение

### 1. Миграция 027 (переписывается) — таблица `push_log`

```sql
CREATE TABLE push_log (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    card_guid        TEXT,
    organization_id  UUID REFERENCES organizations(id),
    pushed_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    overrode_audit   BOOLEAN NOT NULL,
    card_data        JSONB
);

COMMENT ON COLUMN push_log.card_data IS
    'Резерв под снимок card_data, которую этот пуш затёр. Сейчас всегда NULL — '
    'колонка существует заранее, чтобы начать её заполнять без новой миграции.';

CREATE INDEX push_log_org_date_idx ON push_log (organization_id, pushed_at);
CREATE INDEX push_log_card_guid_idx ON push_log (card_guid);
```

Семантика `overrode_audit`:
- `TRUE` — на момент пуша у строки уже был хотя бы один непустой результат аудита
  (`formal_result`/`diag_result`/`icd_check_result` IS NOT NULL) — пуш их уничтожил.
- `FALSE` — строка была `pending` (ещё не аудирована), `ignored` или `broken` — терять было нечего.

Никакого организационного флага опт-аута: таблица лёгкая (`card_data` всегда `NULL` сегодня), гасить
нечего. Колонки `done_cards.push_count` и `organizations.audit_overwrite_journal_enabled` из старой
версии ветки — убираются, посчитанное в `push_log` их заменяет полностью.

### 2. Триггер — один инсерт на каждый пуш

```sql
CREATE OR REPLACE FUNCTION done_cards_log_push()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO push_log (card_guid, organization_id, overrode_audit)
    VALUES (
        OLD.card_guid,
        OLD.organization_id,
        OLD.formal_result IS NOT NULL
            OR OLD.diag_result IS NOT NULL
            OR OLD.icd_check_result IS NOT NULL
    );
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS done_cards_journal_overwrite ON done_cards;
DROP TRIGGER IF EXISTS done_cards_log_push ON done_cards;
CREATE TRIGGER done_cards_log_push
    BEFORE UPDATE ON done_cards
    FOR EACH ROW
    WHEN (NEW.status = 'pending')
    EXECUTE FUNCTION done_cards_log_push();
```

- `WHEN (NEW.status = 'pending')` — как и раньше, срабатывает только на переходах в `pending`, то есть
  только на пути `upsert_pending` (пуш). Обычный `upsert()` (успешный аудит, `done -> done`) не
  зацепляет триггер вовсе — переаудит не пуш, ничего не логируется.
- `OLD.status` намеренно не проверяется — повторный пуш поверх уже `pending`-строки тоже считается
  пушем (`overrode_audit = FALSE`, терять было нечего).
- `BEFORE UPDATE`, не `AFTER` — не блокирует и не меняет сам `UPDATE`, только пишет журнал попутно
  (как существующий `done_cards_set_updated_at`).

### 3. Вьюха `push_metrics_by_date`

```sql
CREATE VIEW push_metrics_by_date AS
SELECT
    organizations.name                                       AS organization_name,
    (push_log.pushed_at AT TIME ZONE 'UTC')::date             AS push_date,
    count(*)                                                  AS pushes_total,
    count(*) FILTER (WHERE push_log.overrode_audit)           AS pushes_overrode_audit,
    count(*) FILTER (WHERE NOT push_log.overrode_audit)       AS pushes_no_override
FROM push_log
LEFT JOIN organizations ON organizations.id = push_log.organization_id
GROUP BY organizations.name, (push_log.pushed_at AT TIME ZONE 'UTC')::date
ORDER BY organization_name, push_date;
```

Дата группировки — `pushed_at::date` в UTC (дата самого API-вызова push, не `visit_date` карты из
`card_data`, и не по местному времени клиники). Это отдельная вьюха от существующей
`done_cards_metrics` (которая про токены/время аудита, группированную по `visit_date` карты) —
разный смысл даты, объединять было бы путаницей.

### 4. `/stats/storage` — переименование колонки таблицы

`StatsStorage.storage_kb` и `StorageStatsResponse` (`src/storage/stats_storage.py`,
`src/api/models.py`) — заменить `audit_overwrite_journal_kb` на `push_log_kb`, источник —
`pg_column_size()` по строкам `push_log` для организации. Остальной эндпоинт (`GET /stats/storage`,
авторизация через `require_org_access`, `done_cards_kb` + `total_kb`) — без изменений.

### 5. Тесты

`tests/test_audit_overwrite_journal.py` → `tests/test_push_log.py`, переписывается под новую схему:
- пуш поверх аудированной карты → одна строка `push_log` с `overrode_audit = TRUE`;
- пуш поверх `pending`/`ignored`/`broken` карты → строка с `overrode_audit = FALSE`;
- переаудит (`done -> done` через `upsert()`) → ничего не логируется;
- несколько пушей за один день от одной организации агрегируются в `push_metrics_by_date` в одну
  строку с правильной суммой `pushes_total`/`pushes_overrode_audit`/`pushes_no_override`;
- пуши в разные дни / от разных организаций не смешиваются во вьюхе.

`tests/test_stats_api.py` — обновить проверки под `push_log_kb` вместо `audit_overwrite_journal_kb`.

## Не в скоупе

- Заполнение `push_log.card_data` — колонка зарезервирована, но сама запись значения в неё (и любая
  логика восстановления карты по журналу) откладывается на будущее без новой миграции.
- Дедупликация повторных идентичных пушей (одна и та же карта, тот же payload) — любой пуш поверх
  существующей строки логируется как есть, без сравнения старого и нового `card_data`.
- Ретеншен/чистка старых записей `push_log` — таблица растёт неограниченно, план по архивации не
  входит в этот документ.
