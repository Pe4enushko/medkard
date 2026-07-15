# Реестр миграций + baseline + reconcile схемы docs

**Дата:** 2026-07-15 · **Статус:** утверждён · **Ветка:** `remove-reverse-hyde` (от `dev`)
**Связано:** `2026-07-14-remove-reverse-hyde-design.md` (эта работа переделывает миграцию 024
из той спеки и разблокирует её применение на стенде).

## Зачем

`migrate.sh` переигрывает **все** `[0-9]*.sql` с 001 при каждом прогоне под `ON_ERROR_STOP=1`
и не ведёт реестра применённых миграций. Живая схема БД дрифтанула от файлов:

- Файлы описывают 3-запросный reverse HyDE: `docs` с `fact_q_embedding/procedure_q_embedding/
  constraint_q_embedding` (002_docs.sql) + три HNSW-индекса по ним.
- Живая таблица `docs` на стенде (`localhost:5532/MedKard`) имеет другую, **нигде в git не
  зафиксированную** схему:
  ```
  id uuid PK · file_id text · chunk text · metadata jsonb
  chunk_embedding vector(1024)   -- мёртвая, all-null; индекс docs_chunk_embedding_hnsw
  hyde_reembedded boolean=false  -- флаг backfill'а chunk_embedding→embedding
  embedding       vector(1024)   -- живая; индекс docs_embedding_idx; reingest её заполняет
  FK docs_file_id_fkey → guidelines(file_id)   -- присутствует → 021 реально применена
  ```

**Симптом:** повторный `migrate.sh` падает на 002 — `CREATE TABLE IF NOT EXISTS docs` тихо
пропускается (таблица есть), затем `CREATE INDEX … ON docs (fact_q_embedding …)` бьёт в колонку,
которой в реальной таблице нет → ERROR → скрипт останавливается.

**Два ортогональных корня** (идемпотентность их не лечит — 002 уже идемпотентна в смысле
`IF NOT EXISTS`, но её индекс ссылается на несуществующую колонку):

1. У `migrate.sh` нет реестра → replay-from-001 хрупок by design.
2. Схема дрифтанула от файлов → файлы описывают не ту реальность.

Плюс следствие: миграция 024 из remove-hyde-спеки написана против неверной (`fact_q_*`) схемы —
на живой БД это **полный no-op** (дропает отсутствующие `fact_q_*`, добавляет уже существующую
`embedding`, создаёт уже существующий `docs_embedding_idx`, не трогает `chunk_embedding`/
`hyde_reembedded`). Её надо переделать.

## Решение (обзор)

1. **Реестр `schema_migrations`** — `migrate.sh` применяет только не-применённые файлы.
2. **Baseline** живой БД флагом `--skip-until` — пометить 001–023 applied без прогона.
3. **Reconcile-024** — переписать под реальную схему: дропнуть `chunk_embedding`/`hyde_reembedded`
   и HyDE-остатки обоих происхождений, оставить `embedding` + индекс. С гейтом от потери данных.

002 **не переписываем** — история иммутабельна, реестр гарантирует, что на стенде 002 больше не
запустится. Свежая БД прогонит `fact_q_*` через 002 и тут же дропнет их в 024 — небольшой оверхед
в обмен на отказ от переписывания истории.

## 1. Реестр `schema_migrations`

```sql
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);
```

`migrate.sh`:
- Первым делом бутстрапит эту таблицу (`CREATE TABLE IF NOT EXISTS …`) — реестр не может
  записать сам себя, поэтому создаётся вне цикла.
- В цикле по `[0-9]*.sql`: `SELECT 1 FROM schema_migrations WHERE filename = :f`. Если строка
  есть — печатаем `skip` и идём дальше. Иначе применяем файл **и** пишем строку в реестре
  **одной транзакцией**:
  ```
  psql --single-transaction --set=ON_ERROR_STOP=1 \
       -f "$sql_file" \
       -c "INSERT INTO schema_migrations(filename) VALUES ('$(basename "$sql_file")')"
  ```
  `--single-transaction` оборачивает `-f` и `-c` в одну транзакцию: файл упал → откат → строка
  реестра не появилась. Успех → файл применён и записан атомарно.

## 2. Baseline: `--skip-until FILE`

Разовое действие оператора на существующей (уже наполненной) БД.

`migrate.sh --skip-until 024_docs_reconcile.sql`:
- Для каждого файла, который сортируется **строго до** `FILE`, пишем строку в
  `schema_migrations` **без применения** (`INSERT … ON CONFLICT (filename) DO NOTHING`).
- Начиная с `FILE` — обычный режим (применить + записать).

Семантика имени: «skip everything until you reach FILE, start applying AT FILE». То есть
`--skip-until 024_docs_reconcile.sql` помечает 001–023 applied и применяет 024+.

На свежей БД baseline **не** делается — просто `migrate.sh`, всё применяется по порядку.

Безопасность baseline: FK `docs_file_id_fkey` присутствует на стенде → 021 реально применена →
пометка 001–023 applied честна. 002 «диверговала» (её `fact_q_*`-индексы никогда не создавались),
но реестр гарантирует, что 002 на стенде не перезапустится, а reconcile-024 не зависит от наличия
`fact_q_*` (везде `IF EXISTS`). Фикция локализована.

## 3. Reconcile: переделанная `024_docs_reconcile.sql`

Origin-agnostic + идемпотентная + с гейтом от потери данных. Сводит **обе** схемы (свежую
`fact_q_*` из 002 и дрифтнутый стенд `chunk_embedding/hyde_reembedded`) к чистому single-`embedding`.

```sql
-- Гейт от потери данных: срабатывает только когда ОБЕ колонки существуют (живой стенд).
-- На свежей БД chunk_embedding нет → проверка пропускается; таблица пуста → потери нет.
DO $$
DECLARE unmigrated int;
BEGIN
    IF (SELECT count(*) FROM information_schema.columns
        WHERE table_name = 'docs'
          AND column_name IN ('chunk_embedding', 'embedding')) = 2 THEN
        SELECT count(*) INTO unmigrated
        FROM docs
        WHERE embedding IS NULL AND chunk_embedding IS NOT NULL;
        IF unmigrated > 0 THEN
            RAISE EXCEPTION
                '% строк docs держат вектор только в chunk_embedding — прогони reingest --force-all до этой миграции',
                unmigrated;
        END IF;
    END IF;
END$$;

DROP INDEX IF EXISTS docs_fact_q_embedding_idx;
DROP INDEX IF EXISTS docs_procedure_q_embedding_idx;
DROP INDEX IF EXISTS docs_constraint_q_embedding_idx;
DROP INDEX IF EXISTS docs_chunk_embedding_hnsw;

ALTER TABLE docs
    DROP COLUMN IF EXISTS fact_q,
    DROP COLUMN IF EXISTS procedure_q,
    DROP COLUMN IF EXISTS constraint_q,
    DROP COLUMN IF EXISTS fact_q_embedding,
    DROP COLUMN IF EXISTS procedure_q_embedding,
    DROP COLUMN IF EXISTS constraint_q_embedding,
    DROP COLUMN IF EXISTS chunk_embedding,
    DROP COLUMN IF EXISTS hyde_reembedded,
    ADD COLUMN IF NOT EXISTS embedding VECTOR(1024);

CREATE INDEX IF NOT EXISTS docs_embedding_idx
    ON docs USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

Гейт **сам навязывает порядок раскатки**: 024 не пройдёт, пока `embedding` не заполнена везде.
Тихого дропа живых векторов быть не может.

Переименовать файл `024_docs_single_embedding.sql` → `024_docs_reconcile.sql` (содержимое
заменяется целиком). Индекс/колонки/имена — те же, что уже на стенде (`docs_embedding_idx`,
`VECTOR(1024)`, HNSW m=16/ef=64), поэтому на стенде ALTER/CREATE — no-op сверх дропов.

## 4. Порядок раскатки на стенде

1. Задеплоить код remove-hyde (уже читает/пишет `embedding`).
2. `bash migrations/migrate.sh --skip-until 024_docs_reconcile.sql` — реестр + 001–023 applied.
3. `python scripts/reingest-pdfs.py --force-all` — заполнить `embedding` во всех строках.
4. `bash migrations/migrate.sh` — применяет только 024 (гейт проходит, дропает следы hyde).
5. Проверка: `\d docs` чистый (нет `chunk_embedding`/`hyde_reembedded`/`fact_q_*`, есть
   `embedding` + `docs_embedding_idx`); spot-check `retrieve()`.

Гейт делает порядок 3→4 обязательным: пропустишь reingest — шаг 4 упадёт с понятной ошибкой.

## 5. Тестирование

**На dev-машине (без БД):**
- `migrate.sh`: логика «файл в реестре → skip» и «`--skip-until` помечает до FILE, применяет с
  FILE» — юнит-тест на bash-разбор без реального psql (мок psql-обёртки / проверка построенных
  команд).

**На стенде (с pgvector):**
- Реестр создаётся; второй прогон `migrate.sh` — все файлы `skip`, 0 изменений.
- `--skip-until 024_docs_reconcile.sql` на копии стенда: реестр получает 23 строки, применяется
  только 024.
- Reconcile-024 идемпотентна (двойной прогон без ошибок); после неё `\d docs` — без
  `chunk_embedding/hyde_reembedded/fact_q_*`, есть `embedding` + `docs_embedding_idx`.
- Гейт: при наличии строки `embedding IS NULL AND chunk_embedding IS NOT NULL` 024 падает с
  RAISE; при отсутствии — проходит.
- Свежая БД: полный `migrate.sh` с нуля → финальная схема `docs` = чистый single-`embedding`
  (сходимость обоих происхождений).

## Область охвата

**Входит:** таблица `schema_migrations`; переписанный `migrate.sh` (бутстрап реестра, skip
применённых, атомарные apply+record, режим `--skip-until`); переделанная `024_docs_reconcile.sql`
(реальные колонки + гейт); документация порядка раскатки.

**Не входит:** переписывание исторических миграций 001–023 (в т.ч. 002); down-миграции (весь
`migrate.sh` forward-only); авто-baseline (оператор явно зовёт `--skip-until` один раз);
изменение модели эмбеддинга/размерности; фильтр поиска по подсекциям (3.1→3.1.1/…) и пометка
невалидных OCR-блоков — отдельные темы, запаркованы.
