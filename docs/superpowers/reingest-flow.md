# Reingest-with-resume — процесс (от папки до завершения)

Сквозная схема работы `scripts/reingest-pdfs.py`: как файлы из `pdfs/` и строки
`resources/manifest.csv` попадают в БД (`docs` + `guidelines`) с устойчивостью к прерываниям.
См. спеку [`2026-07-09-reingest-with-resume-design.md`](specs/2026-07-09-reingest-with-resume-design.md)
и план [`2026-07-14-reingest-with-resume.md`](plans/2026-07-14-reingest-with-resume.md).

## Общий поток

```
┌──────────────────────────── ВХОД (оператор) ────────────────────────────┐
│  pdfs/<file_id>.pdf        — новые / изменённые / откатанные PDF          │
│  resources/manifest.csv    — строки: ID + метаданные (Наим., МКБ-10, ...) │
└───────────────────────────────────┬──────────────────────────────────────┘
                                     │
              (первый прогон / после смены схемы БД — один раз)
                                     │
        bash migrations/migrate.sh   → docs, guidelines, ingest_runs (019–023)
        python scripts/seed-guidelines.py → guidelines := снимок манифеста «как есть»
                                     │
                                     ▼
        python scripts/reingest-pdfs.py  [--data-dir DIR] [--only-failed] [--file-id ID] [--dry-run]
                                     │
      ┌──────────────────────── СБОР СОСТОЯНИЯ ────────────────────────┐
      │  manifest_rows ← manifest.csv                                   │
      │  runs          ← ingest_runs   (status, content_hash)           │
      │  guidelines    ← таблица guidelines  («старый» манифест в БД)    │
      │  current_hash  ← sha256(pdfs/<id>.pdf)   для каждого id          │
      └──────────────────────────────┬─────────────────────────────────┘
                                      ▼
                 build_worklist → classify(file_id)  на КАЖДЫЙ файл
                                      │
        ┌─────────────────────────────┼──────────────────────────────┐
        ▼                             ▼                               ▼
    ┌───────┐                 ┌───────────────┐                  ┌──────┐
    │ FULL  │                 │ METADATA_ONLY │                  │ SKIP │
    └───┬───┘                 └───────┬───────┘                  └──────┘
 нет записи в runs /       done + хеш тот же, НО             done + хеш совпал
 status != done /          строка манифеста != guidelines    + манифест совпал
 хеш PDF изменился         (изменились только колонки)        → ничего не делаем
 (в т.ч. откат файла)                │
        │                            ▼
        │                  guidelines.upsert_many([Guideline])   ← дёшево, без LLM
        │                  (docs не трогаем, ingest_runs не трогаем)
        ▼
```

## Ветка `FULL` (на один файл, per-file try/except)

```
   upsert ingest_runs(id, status='pending')          ← content_hash СОХРАНЯЕМ
        │
   load_documents(only={id}) → iter_chunks()          ← fitz: секции 3-ур. regex + таблицы
        │
   process_batch → LLM hypothetical queries + embeddings   ← самая медленная/хрупкая часть
        │
   replace_by_file_id(id, docs)                       ← атомарно: DELETE+INSERT в 1 транзакции
   guidelines.upsert_many([Guideline.from_manifest_row])
   mark_done(id, content_hash = current_hash)         ← ПОСЛЕДНИМ (точка «файл готов»)
        │
   ── Exception? ──► mark_failed(id, error);  логируем;  идём к следующему файлу
```

## Завершение / прерывание / повтор

```
   Ctrl+C или падение МЕЖДУ файлами   → текущий файл уже 'done' либо остался 'pending'/'failed'
   Ctrl+C ПОСРЕДИ файла               → docs-транзакция атомарна (нет полу-записи),
                                         но mark_done не дошёл → status='pending'
        │
   Повторный запуск  →  build_worklist заново:
        • 'pending'/'failed'/нет записи        → переделать (FULL)
        • 'done' + хеш совпал + манифест совпал → SKIP
        │
   Итог:  SELECT status, count(*) FROM ingest_runs GROUP BY status;
          SELECT file_id, error  FROM ingest_runs WHERE status='failed';
```

## Три инварианта корректности

1. **Хеш решает re-chunk.** Дорогая LLM-генерация запускается только если `sha256(PDF)` разошёлся с
   последней `done`-записью; смена одних метаданных → дешёвый upsert `guidelines`.
2. **`mark_done` последним.** Крах в любой момент оставляет файл не-`done` → следующий прогон его
   переделает (идемпотентно, без ручного вмешательства).
3. **`content_hash` пишется только на `done`.** `pending`/`failed` его сохраняют, поэтому сравнение
   идёт всегда с последним успешным состоянием (откат файла ловится как «изменился»).
