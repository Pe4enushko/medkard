# Вынос манифеста клинреков в таблицу `guidelines`

**Дата:** 2026-07-08
**Статус:** утверждён, готов к плану реализации

## Проблема

Справочник клинических рекомендаций хранится в файле `resources/manifest.csv` вне БД —
рудимент экспериментальной стадии проекта. Это порождает два дефекта:

1. **Файл-вне-БД как источник истины.** `manifest.csv` читается напрямую из файловой
   системы в четырёх независимых местах рантайма (аудит, ICD-чек, отчёт, ingestion).
   Нет единого канонического источника, нет ссылочной целостности с `docs`.

2. **Дублирование манифеста в каждый чанк.** При ingestion `data_loader` через
   `**self.metadata` копирует **все 8 колонок** строки манифеста в JSONB-поле
   `metadata` **каждого** чанка таблицы `docs`. Из них при чтении реально
   используются только три (`Наименование`, `МКБ-10`, `Возрастная категория`) —
   и те резолвятся по `file_id`, то есть дублировать их по чанкам не нужно.
   Остальные пять (`ID` дублирует `docs.file_id`; `Разработчик`,
   `Статус одобрения НПС`, `Дата размещения`, `Статус применения` не читаются нигде)
   — мёртвый груз.

## Цель

- Ввести таблицу `guidelines` — единственный канонический источник справочника,
  связанный с `docs` внешним ключом по `file_id`.
- Убрать копирование колонок манифеста в `docs.metadata`; читаемые поля брать
  через JOIN `docs → guidelines`.
- Перевести все четыре читателя `manifest.csv` на таблицу; убрать чтение CSV из
  рантайма. `manifest.csv` остаётся в репозитории **только** как seed-данные.

## Карта потребителей манифеста (текущее состояние)

Заголовок CSV:
`ID, Наименование, МКБ-10, Возрастная категория, Разработчик, Статус одобрения НПС, Дата размещения, Статус применения`

| Колонка | Кто читает | Откуда |
|---|---|---|
| `ID` | все (как `file_id`) | уже реальная колонка `docs.file_id` |
| `Наименование` | `doc.py` (рендер чанка), `icd_check` (таблица для агента), `result_parser` (отчёт), `decider`/`icd_prefix_picker` (кандидаты) | CSV + `docs.metadata` |
| `МКБ-10` | `clinic_recs` (матчинг), `doc.py`, `icd_check` | CSV + `docs.metadata` |
| `Возрастная категория` | `clinic_recs` (возр. фильтр), `doc.py`, `icd_check`, `result_parser` (age_group) | CSV + `docs.metadata` |
| `Дата размещения` | `result_parser` (поле «date» отчёта) | только CSV |
| `Разработчик` | **никто** | мёртвый груз |
| `Статус одобрения НПС` | **никто** | мёртвый груз |
| `Статус применения` | **никто** | мёртвый груз |

Четыре читателя `manifest.csv` из файла:
- `src/audit/diagnosis/clinic_recs.py` — матчинг МКБ→file_id + возрастной фильтр (аудит)
- `src/audit/icd_check/validator.py` — рендер таблицы манифеста в промпт агента
- `src/reporting/result_parser.py` — обогащение отчёта (name / published_at / age) для Excel и pull-API
- `src/RAG/ingestion/data_loader.py` — splat всех колонок в `metadata` чанка при ingestion

Данные манифеста (707 строк на момент проектирования):
- **МКБ-10:** 449/707 строк содержат несколько кодов через запятую
  (например `"J20.0, J20.1"`).
- **Возрастная категория:** ровно три значения — `Взрослые` (277), `Дети` (149),
  `Взрослые, дети` (281). Единственный разделитель — запятая.

## §1. Схема БД

Новая таблица `guidelines` — полное зеркало строки манифеста; множественные поля —
массивы.

```sql
-- migrations/019_guidelines.sql (идемпотентно: migrate.sh гоняет все файлы каждый раз)
CREATE TABLE IF NOT EXISTS guidelines (
    file_id       TEXT   PRIMARY KEY,                 -- манифестный ID (= docs.file_id)
    name          TEXT,                               -- Наименование
    mkb           TEXT[] NOT NULL DEFAULT '{}',       -- МКБ-10: ["J20.0","J20.1"]
    age_category  TEXT[] NOT NULL DEFAULT '{}',       -- ["Взрослые","Дети"] — дословно как в CSV
    developer     TEXT,                               -- Разработчик
    nps_status    TEXT,                               -- Статус одобрения НПС
    published_at  TEXT,                               -- Дата размещения (строка как в CSV)
    usage_status  TEXT                                -- Статус применения
);

CREATE INDEX IF NOT EXISTS guidelines_mkb_idx ON guidelines USING GIN (mkb);
```

Решения:
- **`file_id` — первичный ключ** (естественный ключ; `docs.file_id` уже ссылается на
  него; `ID` в манифесте уникален). Не вводим отдельный `id UUID`.
- **`mkb TEXT[]`** — ячейка `"J20.0, J20.1"` парсится в `['J20.0','J20.1']`
  (strip + upper). GIN-индекс — матчинг по коду это горячий путь аудита.
- **`age_category TEXT[]`** — значения хранятся **дословно как в CSV**
  (`Взрослые`, `Дети`), чтобы отчёты и промпты не меняли вид. Проверка возраста —
  регистронезависимая (в коде).
- **Все 8 колонок** переносятся, включая мёртвые. Хранить полную строку манифеста в
  одном экземпляре на гайдлайн дёшево; дублирование убирается только там, где оно
  болело — в `docs.metadata` по чанкам.
- `mkb` хранится как массив кодов; нормализация в отдельную таблицу кодов —
  за рамками задачи (YAGNI).

```sql
-- migrations/021_docs_guideline_fk.sql
-- Добавляется ПОСЛЕ seed (см. §4). Guard: добавить, если constraint ещё нет.
-- Перед добавлением — проверка на сирот; при наличии сирот миграция явно падает.
```

**Нумерация миграций важна:** `migrate.sh` применяет `[0-9]*.sql` в лексикографическом
порядке имён. FK обязан идти **после** cleanup, поэтому FK получает номер `021`, а
cleanup — `020`. Итоговый порядок по именам файлов: `019_guidelines` → `020_docs_metadata_cleanup`
→ `021_docs_guideline_fk`. Seed (Python-скрипт) выполняется оператором между прогонами
`migrate.sh` — см. §4.

## §2. Слой хранения и ingestion

**Модель `Guideline`** — `src/storage/models/guideline.py`:
- dataclass с полями таблицы;
- классметод `from_manifest_row(row: dict) -> Guideline` — **единственное место**,
  знающее формат CSV-ячеек: split по запятой, strip, upper для МКБ; массив возрастов
  дословно из ячейки.

**Класс `GuidelinesStorage`** — `src/storage/guidelines_storage.py`, паттерн
`BaseStorage`:

```python
class GuidelinesStorage(BaseStorage):
    async def upsert_many(self, rows: list[Guideline]) -> int      # seed / обновление
    async def get(self, file_id: str) -> Guideline | None
    async def all(self) -> list[Guideline]                          # clinic_recs / icd_check
    async def find_by_code(self, code: str) -> list[Guideline]      # WHERE code = ANY(mkb)
    async def find_by_prefix(self, prefix: str) -> list[Guideline]  # unnest(mkb)+split_part
```

**Seed:** `scripts/seed-guidelines.py` — читает `resources/manifest.csv` →
`Guideline.from_manifest_row` → `GuidelinesStorage.upsert_many`. `manifest.csv`
остаётся в репозитории **только** как seed-данные, из рантайма не читается.

**Ingestion (`scripts/ingest-pdfs.py` + `src/RAG/ingestion/data_loader.py`):**
- `data_loader` перестаёт splat-ить `**self.metadata` в metadata чанка. В `metadata`
  остаётся только chunk-intrinsic: `section`, `content_type`, `chunk_index`, `page`,
  `table_index`. `file_id` — уже отдельная колонка `docs.file_id`.
- FK `docs.file_id → guidelines.file_id` требует наличия строки-родителя. Seed
  заливает весь манифест заранее; ingest дополнительно проверяет
  `GuidelinesStorage.get(file_id)` перед вставкой чанков и падает с внятной ошибкой,
  если гайдлайна нет.

## §3. Перевод читателей на таблицу

**1. `src/audit/diagnosis/clinic_recs.py`** (матчинг МКБ→file_id + возраст)
- `_load_manifest()` / `_find_matching_rows()` / `_find_matching_rows_by_prefix()` —
  удаляются; заменяются делегированием в `GuidelinesStorage.find_by_code` /
  `find_by_prefix` (матчинг переезжает в SQL: `ANY(mkb)` / `unnest+split_part`).
- `_is_age_eligible(guideline, age)` — работает по массиву `age_category`
  (регистронезависимо: `{a.lower() for a in age_category}`). Семантика прежняя:
  только `Дети` → ребёнок; только `Взрослые` → взрослый; оба/пусто → пропускаем.
- `ClinicRecs` становится async-зависимым от `GuidelinesStorage`; конструктор больше
  не принимает `manifest_path`.
- `decider.py` / `icd_prefix_picker.py` получают `Guideline`-объекты/дикты из нового
  источника — правок логики не требуют (меняется тип строки-кандидата).

**2. `src/audit/icd_check/validator.py`** (таблица манифеста для агента)
- Вызов `clinic_recs._load_manifest()` в `pipeline.py` заменяется на
  `GuidelinesStorage.all()`; возрастной фильтр — через `_is_age_eligible`.
- `_render_manifest_table` рендерит `mkb`/`age_category` из массивов через
  `", ".join(...)` — таблица для агента остаётся визуально идентичной.

**3. `src/reporting/result_parser.py`** (отчёт: name / date / age)
- `load_manifest_meta()` (чтение CSV) удаляется. Источник meta —
  `GuidelinesStorage.all()` → тот же словарь `{file_id: {name, published_at, age}}`.
- Тонкость async/sync: `load_manifest_meta()` сейчас синхронный, зовётся из
  sync-`ExcelFormatter._write_rows` и async-`api_formatter`. Решение: оба
  async-контекста (`ExcelFormatter`, `api_formatter`) подгружают meta через
  `GuidelinesStorage` в `__aenter__` и передают готовый dict в sync-парсер.
  Сигнатура `parse_diagnosis(data, manifest_meta)` сохраняется — меняется только
  источник meta.

**4. `src/storage/models/doc.py` + `src/storage/docs_storage.py`** (JOIN)
- `DocsStorage.get()` / `get_many()` добавляют
  `LEFT JOIN guidelines g ON g.file_id = docs.file_id`, выбирая
  `g.name, g.mkb, g.age_category`.
- `Doc` получает новые поля `name`, `mkb`, `age_category`; `_row_to_doc` кладёт
  значения из JOIN в них, а не в `metadata`.
- `Doc._format_chunk` читает `self.name/self.mkb/self.age_category` вместо
  `self.metadata[...]`; массивы рендерятся через `", ".join`. Шапка чанка для агента
  остаётся байт-в-байт прежней.

## §4. Миграция, seed, бэкфилл

`migrate.sh` применяет **все** `[0-9]*.sql` при каждом прогоне (нет версионирования),
с `ON_ERROR_STOP=1`. Значит все миграции обязаны быть идемпотентны.

**Порядок:**

1. `migrations/019_guidelines.sql` — `CREATE TABLE IF NOT EXISTS guidelines` + GIN.
2. **Seed:** `python scripts/seed-guidelines.py` — заливает `manifest.csv`
   (`upsert_many`, идемпотентно по PK).
3. `migrations/020_docs_metadata_cleanup.sql` — удаляет манифест-ключи из
   существующих чанков, оставляя chunk-intrinsic:
   ```sql
   UPDATE docs SET metadata = metadata
       - 'ID' - 'Наименование' - 'МКБ-10' - 'Возрастная категория'
       - 'Разработчик' - 'Статус одобрения НПС'
       - 'Дата размещения' - 'Статус применения'
   WHERE metadata ?| array['ID','Наименование','МКБ-10','Возрастная категория',
                           'Разработчик','Статус одобрения НПС',
                           'Дата размещения','Статус применения'];
   ```
   (`WHERE` делает миграцию фактически идемпотентной — повторный прогон трогает 0 строк.)
4. `migrations/021_docs_guideline_fk.sql` — FK `docs.file_id → guidelines.file_id`,
   в guard «добавить, если constraint не существует», с предварительной проверкой сирот.

Порядок применения `migrate.sh` (по именам файлов): `019` → `020` (cleanup) → `021` (FK).
Seed выполняется оператором между прогонами `migrate.sh`.

**Защита порядка через саму миграцию (важно).** FK-миграция проверяет сирот:
`SELECT DISTINCT file_id FROM docs WHERE file_id NOT IN (SELECT file_id FROM guidelines)`.
Если seed не выполнен, все `docs.file_id` — сироты, миграция **явно падает** с
сообщением, и `migrate.sh` (`ON_ERROR_STOP=1`) останавливается на этом шаге.
Это штатное, желаемое поведение: отдельный runbook-костыль для порядка не нужен —
оператор видит ошибку, выполняет `seed-guidelines.py` и повторяет `migrate.sh`.
После seed повторный прогон FK-миграции проходит (guard пропускает уже существующий
constraint, сирот нет).

## §5. TODO — Python-переборы под будущий перевод в SQL

Перевести в SQL в рамках этой задачи (обязательно, часть §3):
- `clinic_recs._find_matching_rows` / `_find_matching_rows_by_prefix` — линейный
  проход по всем строкам со split → `find_by_code` / `find_by_prefix`.

Оставить как есть, пометить `TODO(guidelines-sql)` в коде (справочник мал ~707 строк,
производительность не болит; меньше риска в аудит-пайплайне):
- `pipeline.py` — `[r for r in all_manifest_rows if _is_age_eligible(r, age)]`:
  грузит весь справочник и фильтрует по возрасту в Python. Свернуть в
  `GuidelinesStorage.all_age_eligible(age)` с фильтром в SQL.
- `result_parser.load_manifest_meta` / `api_formatter` — грузят весь справочник в
  dict и джойнят по строкам отчёта в Python. Свернуть в
  `WHERE file_id = ANY(:ids)` (загрузка только нужных file_id).

`icd_check._render_manifest_table` — это форматирование, не поиск; SQL не применим,
TODO не нужен.

Формулировка маркера в коде:
```python
# TODO(guidelines-sql): фильтрацию по возрасту вынести в SQL —
# GuidelinesStorage.all_age_eligible(age) вместо загрузки всего
# справочника и фильтра в Python. См. spec §5.
```

## §6. Тестирование

TDD, pytest (`pythonpath = src`, `asyncio_mode=auto`):
- `Guideline.from_manifest_row` — юнит: парсинг МКБ/возраста в массивы, strip/upper,
  пустые ячейки.
- `_is_age_eligible` по массиву — таблица кейсов (ребёнок/взрослый ×
  `Дети`/`Взрослые`/`Взрослые,дети`/пусто), регистронезависимость.
- `GuidelinesStorage.find_by_code` / `find_by_prefix` — против тестовой БД
  (как существующие storage-тесты).
- `Doc._format_chunk` — шапка чанка байт-в-байт совпадает с прежней при заполненных
  `name/mkb/age_category`.
- Регрессия существующих тестов манифеста: часть перевести на новый источник, часть
  (парсинг CSV в `Guideline`) сохранить.

**Verification-разрыв.** На машине разработки нет доступа к БД и нет возможности
`pip install`. Миграции, seed, бэкфилл и storage-тесты против БД прогоняются
**только на стенде**. Локально прогоняемы юнит-тесты чистых функций
(`from_manifest_row`, `_is_age_eligible`, `_format_chunk`) — при уже собранном venv.

## Область изменений (файлы)

Новые:
- `migrations/019_guidelines.sql`
- `migrations/020_docs_metadata_cleanup.sql`
- `migrations/021_docs_guideline_fk.sql`
- `src/storage/models/guideline.py`
- `src/storage/guidelines_storage.py`
- `scripts/seed-guidelines.py`

Изменяемые:
- `src/RAG/ingestion/data_loader.py` — убрать splat манифеста в metadata
- `src/audit/diagnosis/clinic_recs.py` — матчинг/возраст через GuidelinesStorage
- `src/audit/pipeline.py` — источник manifest_rows + TODO(guidelines-sql)
- `src/audit/icd_check/validator.py` — рендер из массивов
- `src/reporting/result_parser.py` — meta из GuidelinesStorage + TODO
- `src/reporting/api_formatter.py` — прокинуть meta, TODO
- `src/audit/excel_formatter.py` — прокинуть meta из GuidelinesStorage
- `src/storage/docs_storage.py` — JOIN guidelines в get/get_many
- `src/storage/models/doc.py` — поля name/mkb/age_category, чтение из них

Остаётся как seed-данные (не читается рантаймом):
- `resources/manifest.csv`
