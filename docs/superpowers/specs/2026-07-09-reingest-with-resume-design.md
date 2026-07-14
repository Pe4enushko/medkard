# Реингест RAG-документов с resume-состоянием

**Дата:** 2026-07-09 · **Ревизия:** 2026-07-14 · **Статус:** утверждён · **Ветка:** `reingest-with-resume`
**Связано:** `2026-07-08-guidelines-table-design.md` (таблица `guidelines` — уже влита в main,
хранит колонки манифеста денормализованно), коммит 3-уровневого regex подсекций (`\d+\.\d+(?:\.\d+)?`,
уже в main — причина, по которой существующие чанки в `docs` нужно переписать).

## Зачем

Две причины перегнать документы в БД:

1. **Сменился код чанкинга.** 3-уровневый regex подсекций меняет разбивку `section` в metadata уже
   занесённых чанков: где раньше была одна секция «3.1 …» на ~67к токенов, теперь «3.1» + «3.1.1» +
   «3.1.2». Меняется набор чанков внутри секции — нужен не точечный UPDATE, а перечанкинг.
2. **Приехал новый манифест / новые/изменённые PDF.** Reingest синхронизирует БД с текущим
   состоянием `resources/manifest.csv` и файлов `pdfs/`.

`scripts/ingest-pdfs.py` для этого не подходит: он читает `get_ingested_file_ids()` и **скипает** файлы,
уже присутствующие в `docs` — то есть именно те, которые надо переписать.

Reingest — устойчивый к прерываниям инструмент синка: `docs` (чанки PDF) + `guidelines` (метаданные
манифеста) приводятся в соответствие текущим манифесту и PDF, с resume-состоянием в служебной таблице.

## Что делаем с каждым файлом манифеста (work-list)

Для каждого `file_id` (= колонка `ID` манифеста) считаем `current_hash = sha256(PDF на диске)` и берём
строку `ingest_runs[file_id]` (её может не быть). Дальше — по приоритету:

1. **Полный reingest** (re-chunk `docs` + upsert `guidelines`), если выполнено **любое**:
   - нет записи в `ingest_runs` — новый файл;
   - `status != 'done'` — `pending`/`failed` (resume после прерывания / долечивание упавших);
   - `current_hash != ingest_runs.content_hash` — **PDF изменился** (в т.ч. откат к другой версии).
2. **Только метаданные** (upsert строки `guidelines`, БЕЗ re-chunk и LLM), если файл `done`, хеш
   совпал, но строка манифеста **отличается** от строки в `guidelines`. Сравнение — на уровне
   нормализованного `Guideline` (`Guideline.from_manifest_row(new)` vs строка из `guidelines`: МКБ в
   upper, list-поля разбиты по запятой), чтобы косметические различия CSV не считались diff'ом.
3. Иначе (`done` + хеш совпал + метаданные совпали) → **skip**.

Ключевое: **re-chunk (дорогая LLM-генерация) делаем только при изменении хеша PDF.** Изменение одних
метаданных манифеста при том же PDF обходится дешёвым upsert'ом в `guidelines` — лишней работы нет.

`content_hash` — хеш PDF на момент **последнего успешного** (`done`) reingest этого файла (одна строка
на `file_id`, PK; истории прежних хешей не держим). Сравниваем всегда с ним: если файл откатили к
прежней версии, `current_hash` не совпадёт с последним `done` → делаем reingest, синкаясь к тому, что
на диске сейчас.

## Действия

### Путь 1 — полный reingest (сменился хеш / новый / не-done)

```
upsert ingest_runs(file_id, status='pending')      # content_hash НЕ трогаем

попытаться:
    reader = единственный из load_documents(only={file_id})
    chunks = list(reader.iter_chunks())
    docs   = process(chunks)           # LLM hypothetical queries + embeddings (общий пайплайн)

    replace_by_file_id(file_id, docs)  # атомарно: DELETE docs + INSERT bulk в ОДНОЙ транзакции
    guidelines.upsert_many([new_guideline])
    mark_done(file_id, content_hash=current_hash)   # status='done', hash=current — ПОСЛЕДНИМ

кроме Exception as e:
    mark_failed(file_id, error=str(e))              # status='failed', content_hash СОХРАНЯЕМ
    залогировать и идти дальше — ошибка одного файла не останавливает прогон
    (per-file try/except; LLM-генерация — самая медленная и хрупкая часть)
```

Замена `docs` и upsert `guidelines` — **две отдельные атомарные операции**, не одна кросс-табличная
транзакция (это два разных storage-класса на общем пуле — плести общую транзакцию хрупко и не нужно).
Корректность держится порядком: `mark_done` идёт **последним**, поэтому крах между заменой docs и
`mark_done` оставит `status='pending'` → следующий запуск переделает файл целиком (идемпотентно).

### Путь 2 — только метаданные (хеш тот же, изменились колонки манифеста)

```
upsert guidelines ← новая строка манифеста        # дёшево, без LLM/эмбеддингов
# ingest_runs не трогаем: чанки актуальны, status остаётся 'done', hash тот же
```

## Таблица `ingest_runs`

Миграция `migrations/023_ingest_runs.sql` (следующий свободный номер: guidelines заняли 019/020/021,
export — 022):

```sql
CREATE TABLE IF NOT EXISTS ingest_runs (
    file_id      TEXT PRIMARY KEY,
    status       TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'done' | 'failed'
    content_hash TEXT,                             -- sha256 PDF на момент успешного reingest
    error        TEXT,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Идемпотентна (`IF NOT EXISTS`). **Не** связана FK с `docs`/`guidelines`: реингест перезаписывает файл,
которого временно нет в `docs` между delete и insert внутри транзакции — жёсткая связь создала бы
паразитную зависимость. Прогресс смотрится обычным SQL:

```sql
SELECT status, count(*) FROM ingest_runs GROUP BY status;
SELECT file_id, error FROM ingest_runs WHERE status = 'failed';
```

## Возобновление и прерывание

Возобновление — просто повторный запуск: work-list заново вычисляется по правилам выше (`pending`/
`failed`/отсутствующие/изменённые подбираются, неизменённые `done` скипаются).

Прерывание (Ctrl+C) между файлами безопасно. Прерывание **посреди** транзакции по файлу: транзакция
docs+guidelines атомарна (частичной записи не будет), но `ingest_runs` останется `pending` — следующий
запуск увидит не-`done` и переделает файл заново.

## Флаги CLI

- `--only-failed` — перегнать только `status='failed'` (долечить упавшие, не трогая остальное).
- `--file-id <id>` — принудительный реингест одного файла **вне** diff/status-логики (точечная проверка).

## Изменения кода

1. **`src/RAG/ingestion/pipeline.py`** (новый) — вынести из `scripts/ingest-pdfs.py` чистую часть
   пайплайна: `chunk_text(chunk)`, `process_chunk(chunk, file_id) -> Doc | None` (LLM queries +
   embeddings), батч-обёртку. `ingest-pdfs.py` начинает импортировать оттуда — **поведение не меняется**,
   покрываем тестами до и после.
2. **`load_documents(..., only: set[str] | None = None)`** в `src/RAG/ingestion/data_loader.py` — новый
   параметр, симметрично существующему `exceptions`. При заданном `only` yield-им только эти `file_id`.
3. **`DocsStorage.replace_by_file_id(file_id, docs)`** — атомарно `delete_by_file_id` + `insert_many`
   в одном `async with self._pool.connection()` (psycopg3 оборачивает блок в транзакцию).
4. **`IngestRunsStorage`** (новый, `src/storage/ingest_runs_storage.py`, паттерн `BaseStorage`) —
   `get_all() -> dict[file_id, (status, content_hash)]`, `upsert_pending(file_id)`,
   `mark_done(file_id, content_hash)`, `mark_failed(file_id, error)`. Экспорт в `storage/__init__.py`.
   Инвариант: `content_hash` пишется **только** в `mark_done`; `upsert_pending`/`mark_failed`
   существующий `content_hash` сохраняют — чтобы он всегда отражал последний успешный ingest.
5. **`GuidelinesStorage.upsert_many`** уже есть — используем для одной строки при полном reingest.
6. **`scripts/reingest-pdfs.py`** (новый) — оркестрация: собрать манифест, прочитать статусы/хеши/
   `guidelines`, вычислить work-list, прогнать по файлам с per-file try/except и логированием (по образцу
   `ingest-pdfs.py`), поддержать флаги.

## Область охвата этой сессии

Входит: механизм reingest + resume-таблица + триггеры (PDF-hash-diff → полный reingest; manifest-diff
при том же хеше → metadata-only upsert `guidelines`) + вынос общего пайплайна + `only=` +
`replace_by_file_id` + `IngestRunsStorage`.

Не входит: удаление из БД файлов, пропавших из манифеста (только add/update); версионная история хешей
(храним только последний `done`).

## Тесты

**Чистые (без БД):** решает `classify(file_id) -> full | metadata_only | skip`:
- матрица (status ∈ {отсутствует, pending, failed, done}) × (hash-diff да/нет) × (manifest-diff да/нет)
  → ожидаемое решение. Ключевые случаи: `done`+hash-diff → `full` (даже если метаданные совпали);
  `done`+hash-same+manifest-diff → `metadata_only`; `done`+всё совпало → `skip`; откат PDF (hash ≠
  последнему `done`) → `full`.
- Формирование записи `ingest_runs`: `mark_done` пишет hash; `upsert_pending`/`mark_failed` сохраняют
  прежний `content_hash`.
- Хеш-функция PDF (детерминизм, чувствительность к изменению байт).
- `pipeline.process_chunk` на фейковом чанке с моками `generate_queries`/`embed_queries` → корректный `Doc`.
- Регресс: `ingest-pdfs.py` после выноса `pipeline.py` собирает тот же `Doc` (мок LLM/embeddings).

**На стенде (с БД):**
- Путь 1: транзакционная замена `docs` + upsert `guidelines` для одного файла; `ingest_runs` → `done`+hash.
- Путь 2 (metadata-only): изменили только колонку манифеста при том же PDF → `guidelines` обновлён,
  строки `docs` НЕ тронуты (те же id/количество), LLM не вызывался.
- Откат PDF к прежней версии (hash ≠ последнему `done`) → полный reingest.
- Повторный запуск: подбирает `failed`, скипает неизменённые `done`.
- Прерывание посреди файла не оставляет частичных чанков; `pending` подбирается следующим запуском.
