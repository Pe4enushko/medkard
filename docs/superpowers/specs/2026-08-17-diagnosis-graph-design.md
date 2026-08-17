# Аудит диагноза по клинрекам: кастомный граф вместо ReAct-агентов, реранкер, источники

Дата: 2026-08-17. Ветка спек: `specs-2026-08-17` (от `origin/release`).
Реализация — отдельной веткой от ветки спек. Зависит от спеки
`2026-08-17-grls-registry-design.md` (узел `lookup_drugs` читает `GrlsStorage`);
до её выкатки узел работает через `DrugsStorage`-совместимый адаптер — см. §5.3.

## 1. Зачем

Сейчас `DiagnosisValidator.validate_diagnosis()` (`src/audit/diagnosis/validator.py`)
запускает три `create_react_agent` (anamnesis / inspection / treatment) с
file-scoped тулами (`search_<aspect>`, `search_guideline`, `search_medicine`).
Проблемы:

1. **Устойчивость.** ReAct-цикл — главный источник битых карт: рекурсия,
   раздувание истории до лимита контекста, повторные вызовы одного тула, ретрай
   всей цепочки заново (`broken_cards_report.md`; для диагноз-чекеров — invalid
   JSON, отсутствие `parsed`, нарушение схемы).
2. **Источники пишет модель.** `CheckerSource(doc_title, section, cite)` LLM
   заполняет по промпту; программной связи issue ↔ чанк нет — цитату нельзя
   проверить, а «общий список источников аудита» собрать не из чего.
3. **Ретрив узкий и без реранка.** `TARGETED_TOP_K=4` из 24 кандидатов,
   секционный фильтр (`%лечен%`) отрезает релевантное вне подраздела; реранкер
   реализован (`rerank_results`), но выключен (`RERANK_BASE_URL` пуст).

Цель: детерминированный граф «вопросы → большой ретрив + реранк → судьи без
тулов → источники из метаданных чанков», где отказ любого шага деградирует, а не
роняет карту.

## 2. Границы

**Не меняется:** контракт `DiagnosisValidator.validate_diagnosis(diagnosis) ->
(DiagnosisAuditResult, tokens)` и его вызовы из `AuditPipeline` (включая повторный
аудит предложенного ICD-кода); `ClinicRecs.pick_recs` (подбор КР по МКБ);
ICD-чекер, его тулы (`get_guideline_structure`, `read_guideline_section`),
`rag_agent.create_checker_agent`, `LLMClient.call_agent` — остаются для ICD
(отдельная спека потом); формальная проверка; API/экспорт (получают новые поля
внутри `diag_result` JSON).

**Не входит:** ICD-чекер на граф; статус КР в результате; изменение
`pick_recs`; ретрив по нескольким КР на один диагноз; UI.

## 3. Расположение кода (архитектурное правило)

Графы LangGraph живут в `src/LLM/graphs/`; `audit/` — только высокоуровневые
пайплайны, которые вызывают модули других слоёв (парсеры, сборщики контекста,
LLM, storage). Это нужно, чтобы архитектуру медчека можно было переносить в
другие проекты.

```
src/LLM/graphs/
├── __init__.py
├── diagnosis.py        # build_diagnosis_graph(): StateGraph, рёбра, компиляция
├── diagnosis_state.py  # TypedDict/dataclass стейта, Chunk, Question, JudgeOutput…
└── diagnosis_nodes.py  # чистые async-функции узлов (тестируются без графа)
src/LLM/prompts/
├── diagnosis_questions.txt   # generate_questions
├── diagnosis_drugs.txt       # extract_drugs
├── anamnesis_checker.txt     # переписаны: без тулов, с нумерованными чанками
├── inspection_checker.txt
└── treatment_checker.txt
src/RAG/retrieval/searches.py # search_in_guideline() вместо search_anamnesis/inspection/treatment
src/audit/diagnosis/validator.py # оркестрация: pick_recs → граф → DiagnosisAuditResult
```

Пины в `requirements.txt`: `langgraph`, `langchain-core` (и `langchain*`)
запиниваются на версии, с которыми ветка проверена (сейчас без версий).

## 4. Граф

```
START ─┬─ generate_questions ─────────────┐
       └─ extract_drugs ── lookup_drugs ──┤
                                          ▼
                                     retrieve
                    ┌─────────────────────┼─────────────────────┐
             judge_anamnesis       judge_inspection       judge_treatment
                    └─────────────────────┼─────────────────────┘
                                   collect_sources ── END
```

`generate_questions` и `extract_drugs → lookup_drugs` — параллельные ветви от
START (fan-out/fan-in LangGraph); три судьи — параллельные ветви после
`retrieve`. Циклов и условных рёбер нет; `recursion_limit` не нужен.

### 4.1 Стейт (`diagnosis_state.py`)

```python
class Question(TypedDict):     aspect: Literal["anamnesis","inspection","treatment"]; text: str
class DrugMention(TypedDict):  as_written: str; normalized: str
class Chunk(TypedDict):
    ref: int                    # номер в контексте судьи (1..n), уникален внутри аспекта
    id: str                     # docs.id (uuid)
    file_id: str
    doc_title: str              # guidelines.name
    section: str | None
    chunk_index: int | None
    content_type: str           # text | table
    page: int | None            # только у table
    table_index: int | None
    text: str
    rrf_score: float
    rerank_score: float | None
    questions: list[str]        # какие вопросы его вытащили
class JudgeIssue(BaseModel):   issue: str; chunk_refs: list[int]
class JudgeOutput(BaseModel):  issues: list[JudgeIssue]
class DiagnosisState(TypedDict):
    # вход
    visit_context: str; patient_block: str; diagnosis_block: str
    file_id: str; doc_title: str; toc: list[str]
    # промежуточное
    questions: list[Question]; drug_mentions: list[DrugMention]; drug_context: str
    pools: dict[str, list[Chunk]]            # aspect -> нумерованный пул
    # выход
    issues: dict[str, list[ResolvedIssue]]   # aspect -> issues с IssueSource
    sources: list[GuidelineSource]
    errors: list[str]                        # человекочитаемые факты деградации
    tokens: int
```

Слияние параллельных ветвей — редьюсеры LangGraph (`operator.add` для
`errors`/`tokens`, merge dict для `pools`/`issues`).

### 4.2 `generate_questions`

Один structured-вызов `LLMClient.call(..., response_model=QuestionSet)`, где
`QuestionSet{anamnesis: list[str], inspection: list[str], treatment: list[str]}`,
на каждый аспект `1..QUESTIONS_PER_ASPECT_MAX` (по умолчанию 4). Вход: пациент,
диагноз (код, наименование, детализация, впервые), контекст записи
(`_format_visit_context` — как сейчас), **оглавление КР** (`get_sections_for_file`,
уже есть — даёт модели структуру документа и терминологию разделов). Промпт
требует вопросы в форме реального клинического запроса к КР («какие жалобы и
анамнестические данные обязательны при …», «какие исследования при первичной
диагностике …», «какая стартовая терапия / критерии смены терапии …»), без
пересказа записи.

Провал после ретраев `LLMClient` → **статические вопросы-шаблоны** по аспекту
(константа в `diagnosis_nodes.py`, по 2–3 на аспект с подстановкой названия
диагноза) + запись в `errors`: `"generate_questions: fallback templates
(<причина>)"`. Граф продолжает.

### 4.3 `extract_drugs` → `lookup_drugs`

`extract_drugs` — structured-вызов `DrugList{items: list[DrugMention]}` по
полям «Назначения», «Рекомендации», «Услуги» и назначенческим фрагментам
«ДанныеОсмотра». Пусто → узел возвращает `[]` (это не ошибка). Провал →
`drug_mentions=[]` + `errors`.

`lookup_drugs` — без LLM: для каждого `normalized` — `GrlsStorage.search_by_inn`
→ `search_by_trade_name` → `DietarySupplementsStorage.search`; результат через
общий `format_medicine_lookup(...)` (спека ГРЛС §6) в один блок `drug_context`:

```
## Справка по препаратам (ГРЛС, реестр от 2026-08-17)
- Амоксиклав → МНН амоксициллин + клавулановая кислота; РУ Действующий (до 2027-03-01); ЖНВЛП
- Ксизал → МНН левоцетиризин; РУ Истёкший (истекло 2025-12-31)
- Бак-Сет → БАД, свидетельство RU.77.99…
- Флюдитек → не найден в реестрах
```

Ошибка БД → `drug_context="справка недоступна"` + `errors`. Блок идёт **только**
treatment-судье.

### 4.4 `retrieve`

Для каждого вопроса (конкурентно, `asyncio.Semaphore(RETRIEVE_CONCURRENCY)`):

1. `embed(question)`;
2. `_vector_search_filtered(embedding, file_id, limit=CANDIDATES_PER_QUESTION)`
   — **без** `section_filter`: вопросы уже специфичны, а фильтр `%лечен%` /
   `%жалоб%` резал релевантные подразделы; исключение «Список литературы»
   остаётся;
3. BM25 по кандидатам + RRF (как в `_hybrid_filtered`);
4. `rerank_results(question, ranked, top_k=TOP_K_PER_QUESTION)` —
   `RERANK_CANDIDATE_LIMIT` поднимается до `CANDIDATES_PER_QUESTION`;
5. результат — `list[Chunk]` без `ref`.

Это оформляется как `searches.search_in_guideline(question, file_id, *,
candidates, top_k) -> list[dict]` (заменяет `search_anamnesis/inspection/
treatment`; `_hybrid_filtered` остаётся для ICD-путей, если они его используют).

Пул аспекта: объединение результатов вопросов этого аспекта по `id`; при
коллизии — max `rerank_score` (или `rrf_score`), `questions` сливаются; кап
`ASPECT_POOL_MAX_CHUNKS` (по умолчанию 20) по убыванию скора; затем сортировка
`(section, chunk_index)` для читаемости и нумерация `ref=1..n`. `doc_title` — из
`guidelines` по `file_id` (один запрос на диагноз, до графа).

Ошибки: недоступен реранкер → RRF-порядок (fail-open, телеметрия, как сейчас);
ошибка embed/БД на одном вопросе → вопрос пропущен + `errors`; все вопросы
аспекта упали → пул пуст, судья аспекта **не вызывается**, `issues[aspect]=[]`,
`errors`. Полный отказ БД → исключение наружу (карта broken, как сейчас — это не
деградация, а инфраструктурная авария).

### 4.5 `judge_<aspect>` ×3

Один structured-вызов `LLMClient.call(..., response_model=JudgeOutput)`:

- system — переписанный `<aspect>_checker.txt`: роль и правила осторожности
  как сейчас; **нет** раздела «Инструменты»; формат ответа —
  `{"issues":[{"issue": "...", "chunk_refs": [3, 7]}]}`; правило: замечание
  без хотя бы одного `chunk_ref` из показанного пула не допускается;
- user — `## Пациент`, `## Диагноз`, `## Клинический контекст записи` (как
  сейчас), для treatment — `## Справка по препаратам`, затем
  `## Фрагменты клинических рекомендаций «{doc_title}»` — пронумерованные
  `[ref] раздел | фрагмент chunk_index (стр. page для таблиц)` + текст (рендер
  как `Doc._format_chunk`, но с `[ref]` в заголовке).

Пост-обработка (код):

1. `ChineseDetector.repair_issue` для `issue` — как сейчас;
2. резолв `chunk_refs` по пулу аспекта → `IssueSource(doc_title, section,
   cite, chunk_id, chunk_index)`, где `cite` = первые `CITE_MAX_CHARS` (300)
   символов текста чанка (для таблиц — первая строка JSON); невалидный ref
   (нет в пуле) — отбрасывается с warning; issue без валидных ref —
   сохраняется с пустыми `sources` и warning (не теряем находку, но она видна
   как «без опоры»);
3. ретраи — уже в `LLMClient.call` (2 попытки, temp bump); финальный провал →
   `issues[aspect]=[]` + `errors: "judge_<aspect>: <причина>"`. Карта не
   ломается.

### 4.6 `collect_sources`

Из **всех** чанков всех пулов (показанных судьям), независимо от цитирования:

```python
class GuidelineSourceSection(TypedDict): section: str | None; chunk_indices: list[int]; cited: bool
class GuidelineSource(TypedDict):        file_id: str; doc_title: str; sections: list[GuidelineSourceSection]
```

Сейчас КР на диагноз одна — список из одного элемента; форма — список, чтобы не
менять контракт, когда `pick_recs` начнёт возвращать несколько. `cited=True`,
если хотя бы один чанк раздела попал в `chunk_refs` любого issue.

## 5. Результат и вывод

### 5.1 Модели

- `DiagnosisAuditResult` (`audit/models.py`): + `sources: list[GuidelineSource]`,
  + `errors: list[str]`. `to_dict()` включает оба.
- `IssueSource` (`storage/models/result.py`): + `chunk_id: str | None`,
  `chunk_index: int | None`; `pretty_format` — без изменений видимого формата.
- `DiagnosisResult` (то, что уходит в `done_cards.diag_result`): +
  `sources`, `errors`; `pretty_format` печатает блок
  `[ИСТОЧНИКИ]: <doc_title> — разделы: 3.1 Лечение (цит.), 2.2 Диагностика`
  и, если есть, `[ДЕГРАДАЦИЯ]: …`.
- Запись `diag_result` JSON:
  ```json
  {"icd_code": "J06.9", "guideline_file_id": "…",
   "issues": [{"issue": "…", "sources": [{"doc_title": "…", "section": "3.1 …", "cite": "…", "chunk_id": "uuid", "chunk_index": 42}]}],
   "sources": [{"file_id": "…", "doc_title": "…", "sections": [{"section": "3.1 …", "chunk_indices": [41, 42], "cited": true}]}],
   "errors": []}
  ```
  `reporting/result_parser.parse_diagnosis` терпит старые записи без
  `sources`/`errors`/`chunk_id` (обратная совместимость — тестом).

### 5.2 Excel и API

- `parsers/excel.py`: колонка **«Источники КР»** после «Проверка по
  клин.рекоммендациям» (`_HEADERS`, `_COLUMN_WIDTHS`), содержимое — по диагнозу:
  `[J06.9] <doc_title>: 3.1 Лечение (цит.); 2.2 Диагностика`. Legacy-раскладка
  не меняется. Тесты на заголовки — обновить.
- `pretty_format` диагноза (колонка «Проверка по клин.рекоммендациям») —
  добавляет блок источников/деградации, как в §5.1.
- API `export`/`check_updates`/`pull` — без правок кода: JSON и xlsx получают
  новые поля автоматически. `docs/visits-api.md` — описать новые поля.

### 5.3 Совместимость с ГРЛС-веткой

Если ветка графа приезжает раньше ГРЛС: `lookup_drugs` использует протокол
`MedicineLookup` с двумя реализациями — `GrlsStorage` (целевая) и адаптер над
`DrugsStorage`(старая) с тем же `format_medicine_lookup`. Выбор — по наличию
таблицы `grls_registry` (`to_regclass`). Адаптер удаляется вместе с `drugs`.

## 6. Реранкер

- Модель: `Qwen/Qwen3-Reranker-0.6B`, отдельный vLLM pooling-процесс (`:8011`)
  рядом с генеративным (`:8010`, на нём `/rerank` → 404). Команда и проверка —
  в `docs/vllm-configuration.md` (раздел «Отдельный rerank server» переписать
  под Qwen3: `--hf-overrides` на `Qwen3ForSequenceClassification`,
  `classifier_from_token ["no","yes"]`, `is_original_qwen3_reranker true`;
  smoke `curl :8011/rerank`).
- Qwen3-Reranker ожидает инструкцию в тексте запроса: `rerank_results` получает
  шаблоны `RERANK_QUERY_TEMPLATE` / `RERANK_DOC_TEMPLATE` (env; дефолты под
  Qwen3: `<Instruct>: {instruction}\n<Query>: {query}` / `<Document>: {doc}`,
  инструкция — «Оцени, отвечает ли фрагмент клинических рекомендаций на
  клинический вопрос»); для bge-подобных моделей шаблоны пустые.
- Конфиг (`.env.example`, `docs/rag.md`):
  `RERANK_BASE_URL`, `RERANK_MODEL`, `RERANK_CANDIDATE_LIMIT=40`,
  `RERANK_TIMEOUT_SECONDS=10` (есть) + `RERANK_QUERY_TEMPLATE`,
  `RERANK_DOC_TEMPLATE`, `RERANK_INSTRUCTION`.
- Fail-open сохраняется; телеметрия `retrieval_rerank(_error)` — есть.

## 7. Конфигурация графа (env, все с дефолтами)

| Переменная | Дефолт | Смысл |
|---|---|---|
| `DIAG_QUESTIONS_PER_ASPECT_MAX` | 4 | верхняя граница вопросов на аспект |
| `DIAG_CANDIDATES_PER_QUESTION` | 40 | кандидатов из HNSW на вопрос |
| `DIAG_TOP_K_PER_QUESTION` | 5 | после реранка |
| `DIAG_ASPECT_POOL_MAX_CHUNKS` | 20 | кап пула аспекта |
| `DIAG_RETRIEVE_CONCURRENCY` | 8 | семафор embed/rerank |
| `DIAG_CITE_MAX_CHARS` | 300 | длина `cite` |

Промпты — `.txt` в `src/LLM/prompts/`, как принято.

## 8. Удаляется

`SearchAnamnesisTool`, `SearchInspectionTool`, `SearchTreatmentTool`,
`SearchGuidelineTool`, `SearchMedicineTool` и `get_tools_for`/
`get_treatment_tools_for` (`src/LLM/tools.py`; ICD-тулы остаются);
`searches.search_anamnesis/inspection/treatment/search_by_file_id` (заменены
`search_in_guideline`); `audit/diagnosis/schemas.py` (`CheckerOutput` →
`JudgeOutput` в стейте графа); в `validator.py` — сборка агентов и
`_parse_issues`/`_load_checker_json` (fallback-парсинг JSON из фенсов не нужен —
схему держит `response_format`). `docs/diagnosis_validator.md`, `docs/rag.md`
(«LangChain tools») — переписать.

## 9. Тесты

Без БД/LLM (обязательные), `tests/test_diagnosis_graph_*.py`:

- узлы на фейках: `generate_questions` (парсинг, кап на аспект, фолбэк-шаблоны
  при исключении, `errors`); `extract_drugs`/`lookup_drugs` (форматирование
  `drug_context`, недоступная БД); `retrieve` — сборка пула из фейкового
  `search_in_guideline`: дедуп по `id`, max скор, кап, сортировка,
  нумерация, слияние `questions`, пропуск упавшего вопроса; `judge_*` —
  резолв `chunk_refs`, отбрасывание невалидных, issue без ref, `cite`
  обрезка/таблица, деградация при исключении; `collect_sources` — разделы,
  `cited`;
- сборка графа: топология (нет циклов), fan-out/fan-in редьюсеры, полный
  прогон на фейках даёт `DiagnosisAuditResult` с 3 списками + `sources` +
  `errors`;
- `parse_diagnosis` — старая и новая запись; `DiagnosisResult.pretty_format`
  с источниками; Excel — заголовки/ширины с новой колонкой;
- `rerank_results` — шаблоны Qwen3 применяются к query/documents (мок httpx),
  fail-open;
- `search_in_guideline` — сигнатура и передача `candidates`/`top_k` (мок
  `_vector_search_filtered`).

Стенд: `scripts/audit-file.py` на 10 кешированных визитах (в т.ч. из
`broken_cards.csv` с диагноз-ошибками) до/после — число issues по аспектам,
доля issues с валидными `sources`, `errors`, время и токены; ни одной broken
по диагноз-контуру.

## 10. Документация и журнал

`docs/diagnosis_validator.md` (схема графа, стейт, деградации),
`docs/rag.md` (ретрив по вопросам, реранк, шаблоны), `docs/vllm-configuration.md`
(Qwen3-Reranker), `docs/llm_calls.md` (новые вызовы: questions, drugs, 3 judges),
`docs/visits-api.md` (поля `sources`/`errors`), `.env.example`, `CLAUDE.md`
(схема «Diagnosis checker»). `docs/revision-log.md` — не трогает (нормативка/
реестры не меняются).

## 11. Definition of Done

1. Граф собран в `LLM/graphs/`, `validate_diagnosis` возвращает результат с
   `sources`/`errors`; ReAct-агенты диагноз-контура удалены; тесты §9 зелёные.
2. Реранкер поднят на стенде, `retrieval_rerank` в телеметрии, при остановке
   реранкера аудит идёт по RRF.
3. Стендовый прогон §9: 0 broken по диагноз-контуру на выборке; доля issues
   с непустыми `sources` ≥ 90 %.
4. Excel с колонкой «Источники КР», `export` отдаёт `sources`; доки обновлены.
