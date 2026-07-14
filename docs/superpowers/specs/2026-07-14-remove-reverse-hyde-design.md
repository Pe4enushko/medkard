# Убрать reverse HyDE → контекстные эмбеддинги чанков

**Дата:** 2026-07-14 · **Статус:** утверждён · **Ветка:** `remove-reverse-hyde` (от `dev`)
**Связано:** `2026-07-09-reingest-with-resume-design.md` (reingest-инструментарий, которым
раскатываем re-embed), таблица `guidelines` (метаданные манифеста, JOIN по `file_id`).

## Зачем

Eval показал, что reverse HyDE реализован неправильно. Разбор кода это подтверждает:

1. **2/3 эмбеддингов — балласт.** На каждый чанк при ingest генерились и эмбеддились три
   гипотетических вопроса (`fact_q` / `procedure_q` / `constraint_q`). Но retrieval **всегда**
   бьёт только в `fact_q_embedding`: `hybrid_search` вызывается единственный раз
   (`rag_agent.py`) с жёстким `query_type="fact"`, а `_vector_search_filtered` по умолчанию
   тоже использует `fact_q_embedding`. Колонки `procedure_q_embedding` / `constraint_q_embedding`
   пишутся и **никогда не читаются** — впустую тратятся вызовы LLM и эмбеддера.
2. **Рассогласование представлений.** Единственная используемая колонка эмбеддит *гипотетический
   вопрос*, а реальный поисковый запрос эмбеддится нормально → матчинг «вопрос против вопроса»
   вместо «запрос против содержимого».

Решение: убрать HyDE целиком (ingestion + retrieval) и перейти на **обычные эмбеддинги** —
эмбеддить контекстный текст самого чанка. Гибридный поиск (BM25 + вектор через RRF) остаётся:
он ортогонален HyDE и работает по тексту чанка.

## Что эмбеддим

**Эмбед-текст = раздел (если есть) + текст чанка.** Ничего больше.

```
[3.1.1 Диагностика заболевания]
<текст чанка>
```

Обоснование состава:
- `section` (номерной заголовок раздела) различает чанки **внутри** документа — полезный сигнал.
- Наименование / МКБ / возрастная категория — document-level: одинаковы для всех чанков файла,
  а поиск скоупится по `file_id` (см. `searches.py`), поэтому внутри скоупа они не дают
  различающего сигнала. Не эмбеддим.
- Порядковый номер чанка (`фрагмент N`) — чистый ординал, для ретривала шум. Не эмбеддим.

`section` уже лежит в `chunk["metadata"]`, поэтому идентичность гайдлайна в пайплайн **тред не
нужен** — `process_chunk(chunk, file_id)` сохраняет сигнатуру.

Отдельный билдер (эмбед-текст ≠ `Doc._format_chunk()`, т.к. `_format_chunk` для показа LLM
по-прежнему включает Наименование/МКБ/фрагмент):

```python
# RAG/ingestion/pipeline.py
def embed_text(chunk: dict) -> str:
    section = (chunk.get("metadata") or {}).get("section")
    body = chunk_text(chunk)          # существующий: str, либо json.dumps(rows) для таблиц
    return f"[{section}]\n{body}" if section else body
```

## Миграция `024_docs_single_embedding.sql`

Идемпотентная, forward-only (как весь `migrate.sh`). Размерность вектора без изменений — 1024
(модель Qwen3-Embedding-0.6B не меняем).

```sql
DROP INDEX IF EXISTS docs_fact_q_embedding_idx;
DROP INDEX IF EXISTS docs_procedure_q_embedding_idx;
DROP INDEX IF EXISTS docs_constraint_q_embedding_idx;

ALTER TABLE docs
    DROP COLUMN IF EXISTS fact_q,
    DROP COLUMN IF EXISTS procedure_q,
    DROP COLUMN IF EXISTS constraint_q,
    DROP COLUMN IF EXISTS fact_q_embedding,
    DROP COLUMN IF EXISTS procedure_q_embedding,
    DROP COLUMN IF EXISTS constraint_q_embedding,
    ADD COLUMN IF NOT EXISTS embedding VECTOR(1024);

CREATE INDEX IF NOT EXISTS docs_embedding_idx
    ON docs USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

Существующие строки получают `embedding = NULL` до принудительного re-embed (см. «Раскатка»).

## Изменения кода

### Ingestion — удалить / переписать
- **Удалить файлы:** `src/LLM/query_generator.py`, `src/LLM/embed_queries.py`,
  `src/LLM/prompts/chunk_query_generator.txt`.
- **Переписать** `src/RAG/ingestion/pipeline.py`: `process_chunk` больше не зовёт
  `generate_queries` / `embed_queries`. Строит `embed_text(chunk)`, `embed(text)` → `Doc(embedding=…)`.
  Одна точка отказа (эмбеддер) вместо двух (LLM + эмбеддер). `chunk_text` остаётся.

### Retrieval — схлопнуть
- `src/RAG/retrieval/vector_store.py`: убрать `QueryType`, `_EMBEDDING_COL`,
  `search_fact/search_procedure/search_constraint`. `_vector_search` и `_vector_search_filtered`
  бьют в колонку `embedding` (параметр `col` уходит / фиксируется). `hybrid_search` теряет
  параметр `query_type`; BM25 + RRF без изменений. `_SELECT_COLS` — без `*_q`.
- `src/RAG/retrieval/searches.py`: секционные фильтры (`анамнез/исслед/лечен` по
  `metadata->>'section'`) остаются; меняется только колонка эмбеддинга. `get_section_chunks`
  SELECT — без `*_q`.
- `src/LLM/rag_agent.py`: `retrieve()` вызывает `hybrid_search` без `query_type`.

### Модель / хранилище — чистка
- `src/storage/models/doc.py`: убрать `fact_q/procedure_q/constraint_q` и три `*_q_embedding`;
  добавить `embedding: list[float] | None`. `_format_chunk()` **не меняется** (показ для LLM).
- `src/storage/docs_storage.py`: `_INSERT_DOC_SQL` / `_doc_params` / `_row_to_doc` под колонку
  `embedding`; все `SELECT` — без `*_q`.
- `src/LLM/tools.py`: убрать `fact_q=…/procedure_q=…/constraint_q=…` из конструирования `Doc`
  в `_format_results` и `ReadGuidelineSectionTool`.

## Раскатка re-embed

Представление сменилось для **всех** чанков, но PDF-хэши те же → штатный reingest их заскипает
(`done` + хеш совпал). Нужен разовый принудительный прогон.

Добавляем в `scripts/reingest-pdfs.py` флаг **`--force-all`**: классифицировать каждый файл
манифеста как `full` мимо hash/status-логики (аналог существующего `--file-id`, но на всю базу).
Раскатка: `bash migrations/migrate.sh` (024) → `python scripts/reingest-pdfs.py --force-all`.

## Тесты

**Без БД (dev-машина):**
- `pipeline.embed_text`: с `section` → `[section]\n body`; без section → `body`; для таблицы —
  `body` = `json.dumps(rows)`.
- `pipeline.process_chunk`: мок `embed` → `Doc` с одним `embedding`, без query-полей; эмбеддится
  результат `embed_text(chunk)`.
- `Doc`: поля `fact_q/…` отсутствуют, есть `embedding`; `_format_chunk()` регресс не затронут.
- `docs_storage`: `_doc_params(doc)` возвращает ключи `{file_id, chunk, metadata, embedding}`;
  `_INSERT_DOC_SQL` их упоминает и не упоминает `*_q`.
- Импорт-регресс: `vector_store` / `searches` / `rag_agent` собираются без `query_type`.

**На стенде (с БД):**
- Миграция 024 идемпотентна (двойной прогон без ошибок); колонки `*_q` исчезли, есть `embedding`
  + `docs_embedding_idx`.
- insert `Doc(embedding=…)` → вектор-поиск по `embedding` возвращает ближайший чанк.
- `hybrid_search` (без `query_type`) и секционные `search_anamnesis/inspection/treatment`
  возвращают результаты; `retrieve()` end-to-end.
- `reingest-pdfs.py --force-all` перегоняет все файлы в `full` и заполняет `embedding`.

## Область охвата

**Входит:** удаление HyDE из ingestion + retrieval; миграция 024; единый `embedding`; контекстный
эмбед-текст (`section` + чанк); флаг `--force-all` для раскатки; чистка модели/хранилища/тулзов.

**Не входит:** смена модели эмбеддинга или размерности вектора; изменение чанкинга; удаление
natasha / rank_bm25 / RRF (гибрид остаётся); down-миграция (весь `migrate.sh` forward-only).
