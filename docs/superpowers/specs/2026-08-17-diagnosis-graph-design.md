# Аудит диагноза по клинрекам: кастомный граф вместо ReAct-агентов, реранкер, источники

Дата: 2026-08-17. Ветка спек: `specs-2026-08-17` (от `origin/release`).
Реализация — отдельной веткой от ветки спек. Зависит от спеки
`2026-08-17-grls-registry-design.md` (узел `lookup_drugs` читает `GrlsStorage`);
до её выкатки узел работает через `DrugsStorage`-совместимый адаптер — см. §5.3.
Включает переаудит накопленных broken-карт (§9) — тема влита из ветки
`fix-broken`, отдельно её реализовывать не нужно.

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
       ├─ extract_drugs ── lookup_drugs ──┤
       │                                  ▼
       │                             retrieve
       │            ┌─────────────────────┼─────────────────────┐
       │     judge_anamnesis       judge_inspection       judge_treatment
       │            └─────────────────────┼─────────────────────┘
       └─ retrieve_criteria ── judge_criteria ──┤
                                                ▼
                                         collect_sources ── END
```

`generate_questions`, `extract_drugs → lookup_drugs` и `retrieve_criteria →
judge_criteria` — параллельные ветви от START (fan-out/fan-in LangGraph); три
судьи по вопросам — параллельные ветви после `retrieve`; всё сходится в
`collect_sources`. Циклов и условных рёбер нет; `recursion_limit` не нужен.

Четыре аспекта: `anamnesis`, `inspection`, `treatment` (по сгенерированным
вопросам) и `criteria` (по детерминированно вытащенному разделу КР «Критерии
оценки качества медицинской помощи», §4.5a). Отдельного судьи «по лекарствам»
нет: справка ГРЛС — часть контекста `treatment`, потому что назначения и
лекарства в записи неразделимы (решение 2026-08-17).

### 4.1 Стейт (`diagnosis_state.py`)

```python
Aspect = Literal["anamnesis","inspection","treatment","criteria"]
class Question(TypedDict):     aspect: Literal["anamnesis","inspection","treatment"]; text: str   # criteria — без вопросов
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
    visit_date: date | None                  # для status_at в lookup_drugs
    file_id: str; doc_title: str; toc: list[str]
    # промежуточное
    questions: list[Question]; drug_mentions: list[DrugMention]; drug_context: str
    pools: dict[Aspect, list[Chunk]]         # aspect -> нумерованный пул (criteria — из retrieve_criteria)
    # выход
    issues: dict[Aspect, list[ResolvedIssue]] # aspect -> issues с IssueSource
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
→ `search_by_trade_name` (запрос нормализуется `normalize_query`, субстанции
исключены — спека ГРЛС §5) → `DietarySupplementsStorage.search`; результат через
общий `format_medicine_lookup(..., on=visit_date, registry_date=…)` (спека ГРЛС
§6, статус относительно даты визита по `status_at` §5.1) в один блок
`drug_context`:

```
## Справка по препаратам (ГРЛС, реестр от 2026-08-17; визит 2025-03-10)
- Амоксиклав → МНН амоксициллин + клавулановая кислота; РУ Действующий (до 2027-03-01); формы: таблетки, порошок для суспензии; отпуск: по рецепту; ЖНВЛП
- Ксизал → МНН левоцетиризин; РУ Истёкший (истекло 2025-12-31; на дату визита действовало)
- Гепарин-Рус → МНН гепарин натрия; РУ Действующий, приостановлено применение (предупреждение)
- Бак-Сет → БАД, свидетельство RU.77.99…
- Флюдитек → не найден в реестрах
```

Ошибка БД → `drug_context="справка недоступна"` + `errors`. Блок идёт **только**
treatment-судье; правила трактовки статусов (истекло до визита — замечание;
действовало на дату визита — нет; приостановлено/на подтверждении/иностранная
упаковка — предупреждение, не брак) — в `treatment_checker.txt` (спека ГРЛС §6).

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

### 4.5 `judge_<aspect>` ×3 (anamnesis / inspection / treatment)

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

### 4.5a `retrieve_criteria` → `judge_criteria`

Каждая КР содержит стандартную таблицу «Критерии оценки качества медицинской
помощи» (№ | критерий | да/нет) — чек-лист, по которому качество помощи
оценивают страховые и Росздравнадзор. Это готовый источник замечаний, который
вопросный ретрив вытаскивает случайно; поэтому — отдельная детерминированная
ветка, без генерации вопросов и без реранка.

`retrieve_criteria` (без LLM): чанки КР `file_id`, у которых
`metadata->>'section' ILIKE CRITERIA_SECTION_PATTERN` (по умолчанию
`%критерии оценки качества%`), порядок `chunk_index`, кап
`CRITERIA_MAX_CHUNKS` (по умолчанию 8; таблица обычно 1–3 чанка) → пул
`pools["criteria"]` с `ref=1..n`. Реализация — `searches.get_section_chunks(
file_id, pattern, limit)` (рядом с `search_in_guideline`). Пусто (в КР нет
раздела или ингест назвал его иначе) → `pools["criteria"]=[]`, судья не
вызывается, `issues["criteria"]=[]`, `errors: "retrieve_criteria: section not
found"`. Ошибка БД → то же + `errors`.

`judge_criteria` — тот же контракт, что у судей §4.5 (structured `JudgeOutput`,
`chunk_refs` обязательны, пост-обработка та же), system-промпт
`criteria_checker.txt`: пройти по каждому критерию таблицы и отметить только те,
что **проверяемо не выполнены по записи визита** (одно замечание — один
критерий, с номером/формулировкой критерия); критерии, относящиеся к этапу,
которого в записи нет (стационар, реабилитация, повторный визит), — не
замечание; отсутствие данных ≠ невыполнение, если критерий про действие, а не
про фиксацию в записи. Контекст — как у остальных: пациент, диагноз, запись,
затем `## Критерии оценки качества «{doc_title}»` с `[ref]`.

Пересечение с формальной проверкой (`rules.json`, ярлык `203n`): там —
нозологические критерии приказа, здесь — критерии конкретной КР; они
пересекаются, но КР полнее и свежее. Дедупликацию между контурами не делаем;
если на стенде дубли окажутся заметными — решать отдельно.

Проверить в плане реализации: (а) как ингест режет таблицу критериев — если она
разваливается на текстовые чанки без `section` или заголовок раздела в
`metadata.section` не содержит «критерии оценки качества», паттерн/ингест
подстроить (`get_sections_for_file` по нескольким КР покажет реальные
заголовки); (б) укладывается ли таблица в `CRITERIA_MAX_CHUNKS`.

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

- `DiagnosisAuditResult` (`audit/models.py`): + `criteria_issues:
  list[DiagnosisIssue]` (входит в `all_issues` и в плоский `issues` JSON —
  контракт не меняется), + `sources: list[GuidelineSource]`, + `errors:
  list[str]`. `to_dict()` включает всё.
- `IssueSource` (`storage/models/result.py`): + `chunk_id: str | None`,
  `chunk_index: int | None`; `pretty_format` — без изменений видимого формата.
- `DiagnosisResult` (то, что уходит в `done_cards.diag_result`): +
  `sources`, `errors`; `pretty_format` печатает блок
  `[ИСТОЧНИКИ]: <doc_title> — разделы: 3.1 Лечение (цит.), 2.2 Диагностика`
  и, если есть, `[ДЕГРАДАЦИЯ]: …`; замечания `criteria` печатаются в
  существующем формате диагноз-замечаний с префиксом «Критерий качества:»
  (в тексте `issue`, задаётся промптом/пост-обработкой — на реализации).
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
| `DIAG_CRITERIA_SECTION_PATTERN` | `%критерии оценки качества%` | ILIKE-паттерн раздела КР для `retrieve_criteria` |
| `DIAG_CRITERIA_MAX_CHUNKS` | 8 | кап пула `criteria` |

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

## 9. Переаудит накопленных broken-карт (`scripts/fix-broken.py`)

Замещает ветку `fix-broken` и её спеку `docs/superpowers/specs/2026-08-07-fix-broken-design.md`
(коммит `db3bb0a`, влит в `dev` как `d6afb14` — **только документ, кода нет**).
Отдельно ту ветку реализовывать не нужно: скрипт въезжает сюда, потому что без
него DoD §12.3 проверяется лишь на новых картах, а старые broken остаются
мёртвыми навсегда.

### 9.1 Зачем скрипт нужен именно этой ветке

Граф убирает ReAct-цикл — источник падений диагноз-контура (§1.1) — и
деградирует вместо исключения (§4.2–4.5a). Но **уже накопленные** broken-карты
он не расчинит: они заморожены, и ночной прогон их не подберёт, сколько бы раз
они ни попали в выборку из 1С.

Заморозка складывается из двух шагов (проверено по коду на базе ветки,
`origin/release` @ `2e61516`):

1. `AuditPipeline.run_batched` грузит `get_done_guids()`
   (`src/audit/pipeline.py:97`), а тот отдаёт все GUID со `status = 'done'` —
   **включая broken** (`done_cards_storage.py:364`, фильтр только по
   `card_guid IS NOT NULL AND status='done' AND organization_id IS NOT DISTINCT FROM …`);
2. `CardFilter.filter` отсеивает всё, что попало в `done_guids`
   (`src/audit/filters.py:159-165`, дедуп по lower-cased GUID).

Упавшая карта — строка `done_cards` с `broken = TRUE`, `status = 'done'`,
сохранённым `card_data`, стектрейсом и NULL во всех колонках результатов
(`upsert_broken`, `done_cards_storage.py:199`). `broken` — булева колонка
(миграция `012_done_cards_broken.sql`), а не значение `status`; `status`
принимает `pending`/`done`, и у упавшей карты он `done`, потому что падение —
терминальный исход прогона.

Поэтому пункт «0 broken по диагноз-контуру» без скрипта означал бы «0 новых
broken», а старые — по-прежнему в отчётах.

### 9.2 Решение

Читаем broken-строки из БД и отдаём их `card_data` в пайплайн с
**`done_guids=set()`** — пустой набор отключает дедуп, и карта доходит до
аудита. Правок пайплайна не требуется: `run_batched` уже принимает `done_guids`
параметром, а `AppointmentParser.split` (`parsers/json_parser.py:51`) принимает
голый список визит-словарей — ровно то, что лежит в `card_data`.

При успешном переаудите `upsert()` ставит `broken = FALSE`
(`done_cards_storage.py:117`) — флаг снимается сам, отдельного UPDATE не нужно.

Источник данных — `card_data` из БД, без обращения к 1С. Скрипт работает офлайн
и чинит падения из-за LLM, сети или парсера. Падение из-за битых данных в самой
карте он не чинит — это осознанная граница.

**Скрипт не привязан к диагноз-контуру.** Он переаудирует карту целиком
(формальная проверка, диагноз, ICD), поэтому чинит и падения ICD-чекера, который
эта ветка не трогает (§2) и который даёт основную массу broken
(`broken_cards_report.md`: 81/96). Прогон на этой ветке починит карты,
упавшие по диагноз-контуру; карты с падением в ICD-ReAct упадут снова и
останутся broken — это ожидаемо и должно быть видно в сводке (§9.5), иначе
результат прогона нельзя будет прочитать как проверку DoD.

### 9.3 Интерфейс

```
python scripts/fix-broken.py ORG [-y] [--dry-run] [--num-batches N]
python scripts/fix-broken.py --all [-y] [--dry-run] [--num-batches N]
```

| Аргумент | Смысл |
|---|---|
| `ORG` | Организация: `Alenka` или `MDS`. Взаимоисключающ с `--all` |
| `--all` | Все организации разом |
| `-y` | Пропустить подтверждение |
| `--dry-run` | Показать, что будет переаудировано, и выйти без записи |
| `--num-batches N` | Параллельных карт за раз (по умолчанию 5) |

Ровно одно из `ORG` / `--all` обязательно; ни одного или оба — ошибка аргументов.

Excel-экспорта и выгрузки по FTP нет. Это ремонтный скрипт: починенные карты
подхватит обычный периодный отчёт, который и так строится из БД за период.

Подтверждение — по образцу `_confirm_period` из `scripts/audit-one-c-period.py`:
до запуска печатается режим, число broken-карт в разбивке по организациям и
применяемые фильтры, затем `Proceed? [y/N]`. `-y` пропускает вопрос,
`--dry-run` печатает то же самое и выходит с кодом 0, ничего не записав. Прогон
стоит денег на LLM — страховка окупается.

Логирование по образцу существующих скриптов: `logs/fix-broken_<timestamp>.log`,
уровень DEBUG в файл и stdout одновременно, `close_pool()` в `finally`.

### 9.4 Группировка по организациям

`AuditPipeline` принимает **один** `org_id` и **один** `card_filter` на экземпляр
(`src/audit/pipeline.py:54-56`). Оба влияют на запись: `org_id` проставляется в
каждую строку через `upsert`/`upsert_ignored`/`upsert_broken`, а `card_filter`
приходит из `load_card_filter(org)` (`parsers/filter_config.py:38`) и у
организаций разный (`filterconfig.json`: у `Alenka` активен `IcdFilter`, у `MDS`
— `KDLFilter` и `AnalysisFilter`).

Поэтому в `--all` буквально один прогон недопустим — он проштамповал бы карты
всех клиник одним `org_id` и прогнал через чужой фильтр, испортив атрибуцию.
Дедуп в `get_done_guids` тоже пер-организационный, что ту же границу подтверждает.

Вместо этого: одна команда, одно подтверждение, одна сводка, а внутри
broken-строки группируются по своему `organization_id`, и на каждую группу
создаётся свой экземпляр пайплайна с её `org_id` и её фильтром. Строки с
`organization_id IS NULL` образуют отдельную группу с `org_id=None` и пустым
`CardFilter([])`. При явном `ORG` группа ровно одна — тот же код, частный случай.

### 9.5 Сводка и повторные сбои

Карта, упавшая снова, остаётся broken: `upsert_broken` перезапишет строку свежим
стектрейсом. Ничего не теряется.

Итог считается **сверкой множества broken-GUID до и после прогона**, а не по
возврату пайплайна. `run_batched` возвращает только успешные пары — упавшие
карты в нём не видны (`pipeline.py:145`, `p is not None`), поэтому единственный
надёжный источник итога — состояние БД.

Печатается в stdout и в лог:

- починено — было broken, стало не broken и не ignored;
- ушло в ignored — см. ловушку §9.6;
- осталось broken — с GUID и первой строкой свежего стектрейса по каждой.

Последняя группа — рабочий материал для DoD §12.3: по первой строке стектрейса
видно, диагноз-контур это или ICD. Отдельного разбора по контурам скрипт не
делает — глазами по сводке.

### 9.6 Ловушка: фильтр применяется и при переаудите

`CardFilter` отрабатывает на переаудите так же, как на обычном прогоне. Если
карта упала, а позже в `filterconfig.json` появилась стратегия, под которую она
подходит, — карта не починится, а уйдёт в `ignored` через `upsert_ignored`
(`src/audit/pipeline.py:104`). Для `MDS` это не гипотетика: `KDLFilter` и
`AnalysisFilter` там активны сейчас, а часть broken-карт накопилась раньше.

Это корректное поведение: фильтр — источник истины о том, какие карты вообще
подлежат аудиту. Но в сводке такие карты выводятся **отдельной строкой**, чтобы
«пропала из broken» не читалось как «починилась».

Констрейнт миграции `014_done_cards_no_ignored_broken_overlap.sql` —
`CHECK (NOT (ignored AND broken))` — при этом не нарушается: `upsert_ignored`
ставит `broken = FALSE` в том же UPDATE (`done_cards_storage.py:185`).

### 9.7 Новый метод хранилища

В `DoneCardsStorage`, по образцу `get_pending` (`done_cards_storage.py:351`):

```python
async def get_broken(self, organization_id: str | None = None) -> list[dict]:
    """Return card_guid + card_data + organization_id for broken rows."""
```

- `WHERE broken = TRUE AND card_data IS NOT NULL` — без `card_data` чинить
  нечего, такие строки не выбираем;
- `organization_id` фильтруется, только когда он передан; иначе берём все
  организации и группируем на стороне скрипта (`IS NOT DISTINCT FROM` не годится:
  нужно различать «все» и «строки с NULL»);
- `organization_id` возвращается в выборке — он нужен для группировки в `--all`.

### 9.8 Порядок применения

Скрипт — часть этой ветки, но полезен и до неё, и после. На приёмке порядок
такой: выкатить граф → прогнать `--dry-run` (снять число broken до) → прогнать
`--all` → сверить сводку. Починенные картами диагноз-контура закрывают DoD
§12.3; остаток по ICD-контуру фиксируется числом и остаётся заделом для
отдельной ICD-спеки (§2).

Прогон меняет `updated_at` починенных строк, и `check_updates` отдаст их
«Искре» как свежие — это ровно то, что описано в §13.3 (карты, вечно висевшие
в `pending`, начнут приходить как `done`). Прогонять до, а не после того, как
аналитик снимет разрез «до/после».

## 10. Тесты

Без БД/LLM (обязательные), `tests/test_diagnosis_graph_*.py`:

- узлы на фейках: `generate_questions` (парсинг, кап на аспект, фолбэк-шаблоны
  при исключении, `errors`); `extract_drugs`/`lookup_drugs` (форматирование
  `drug_context`, недоступная БД); `retrieve` — сборка пула из фейкового
  `search_in_guideline`: дедуп по `id`, max скор, кап, сортировка,
  нумерация, слияние `questions`, пропуск упавшего вопроса; `judge_*` —
  резолв `chunk_refs`, отбрасывание невалидных, issue без ref, `cite`
  обрезка/таблица, деградация при исключении; `retrieve_criteria` — пул из
  фейкового `get_section_chunks` (порядок, кап, пустой раздел → судья не
  вызван + `errors`); `judge_criteria` — тот же набор, что у судей;
  `lookup_drugs` — `visit_date` доходит до `format_medicine_lookup`;
  `collect_sources` — разделы, `cited` (включая пул `criteria`);
- сборка графа: топология (нет циклов), fan-out/fan-in редьюсеры, полный
  прогон на фейках даёт `DiagnosisAuditResult` с 4 списками + `sources` +
  `errors`;
- `parse_diagnosis` — старая и новая запись; `DiagnosisResult.pretty_format`
  с источниками; Excel — заголовки/ширины с новой колонкой;
- `rerank_results` — шаблоны Qwen3 применяются к query/documents (мок httpx),
  fail-open;
- `search_in_guideline` — сигнатура и передача `candidates`/`top_k` (мок
  `_vector_search_filtered`).

Скрипт §9 — против реальной БД, как остальные тесты хранилища (гонять на
стенде):

1. **Починка** — broken-карта проходит переаудит, `broken` снимается, результаты
   заполнены;
2. **Повторный сбой** — карта, падающая снова, остаётся broken со свежим
   стектрейсом; старый перезаписан;
3. **`--all` с двумя организациями** — каждая карта сохраняет свой
   `organization_id`, ни одна не получает чужой, к каждой применён её фильтр;
4. **`--dry-run`** — в БД не меняется ничего;
5. **`get_broken`** — не возвращает строки с `card_data IS NULL`; фильтрует по
   организации, когда та передана, и отдаёт все (включая `organization_id IS
   NULL`), когда не передана;
6. **Сводка** — карта, ушедшая в `ignored` по фильтру, попадает в свою строку
   сводки, а не в «починено».

Стенд: `scripts/audit-file.py` на 10 кешированных визитах (в т.ч. из
`broken_cards.csv` с диагноз-ошибками) до/после — число issues по аспектам,
доля issues с валидными `sources`, `errors`, время и токены; ни одной broken
по диагноз-контуру.

## 11. Документация и журнал

`docs/diagnosis_validator.md` (схема графа, стейт, деградации),
`docs/rag.md` (ретрив по вопросам, реранк, шаблоны), `docs/vllm-configuration.md`
(Qwen3-Reranker), `docs/llm_calls.md` (новые вызовы: questions, drugs, 4 judges),
`docs/visits-api.md` (поля `sources`/`errors`), `docs/storage.md` (`get_broken`
и оговорка, что `get_done_guids` включает broken), `.env.example`, `CLAUDE.md`
(схема «Diagnosis checker»). `docs/revision-log.md` — не трогает (нормативка/
реестры не меняются).

## 12. Definition of Done

1. Граф собран в `LLM/graphs/`, `validate_diagnosis` возвращает результат с
   `sources`/`errors`; ReAct-агенты диагноз-контура удалены; тесты §10 зелёные.
2. Реранкер поднят на стенде, `retrieval_rerank` в телеметрии, при остановке
   реранкера аудит идёт по RRF.
3. Стендовый прогон §10: 0 broken по диагноз-контуру на выборке; доля issues
   с непустыми `sources` ≥ 90 %.
4. Excel с колонкой «Источники КР», `export` отдаёт `sources`; доки обновлены.
5. `scripts/fix-broken.py` (§9) есть, его тесты зелёные; прогон `--all` на
   стенде отработал, сводка сведена: накопленные broken диагноз-контура
   починены, остаток по ICD-контуру зафиксирован числом.

## 13. Влияние на интеграцию с «Искрой» (engine)

Проверено по коду engine (ветка `dev`, 2026-08-18). **Кода интеграции править не
нужно; обязательные правки — два промпта аналитика.**

### 13.1 Что проходит без правок

- `integrations/medcheck/exporter.py` — passthrough: `_INSERT`/`_UPSERT` кладут
  JSON как есть в `jsonb`-колонки; от строки API требуются только ключи
  `card_guid, card_data, formal_result, diag_result, icd_check_result, status,
  updated_at`. Новые поля внутри `diag_result` доезжают сами.
- `integrations/medcheck/replica/schema.sql` — колонки, `medcheck_visit_date`,
  индекс и вьюха `done_cards_metrics` от внутренностей `diag_result` не зависят:
  DDL реплики не меняется.
- `integrations/medcheck/client.py` — только query-параметры `export`/
  `check_updates`/`pull`; контракт API спека не трогает (§5.2).
- xlsx-отчёты: engine отдаёт байты как есть (`exporter.run_report`), файл не
  индексируется в RAG — новая колонка «Источники КР» доедет без правок.
- e2e вебклиента (`e2e/helpers/medcheck-db.ts`) считает эталоны живыми запросами
  по двум уровням `diag_result` → `d->'issues'`. Форма сохраняется, тесты не
  переписываются (изменятся только числа, а они вычисляются, не захардкожены).
- Второй контур (`integrations/medcheck_alenka/`) — та же passthrough-механика,
  выводы те же.

### 13.2 Что править обязательно — промпты аналитика

`src/engine/prompts/data_processor_modes/medcheck.md` и `medcheck_alenka.md`,
раздел «The reference: the three `*_result` fields». Сейчас там: элемент
`diag_result[]` = `{icd_code, guideline_file_id}`, внутри `issues[]` → `issue` +
`sources[]` (`cite`, `section`, `doc_title`). После графа элемент получает ещё
`sources[]` (агрегат по разделам КР, §4.6) и `errors[]`, а `issues[].sources[]` —
`chunk_id`/`chunk_index`.

1. **Коллизия имени `sources` на двух уровнях.** У агрегата другая форма
   (`file_id`, `sections[]{section, chunk_indices, cited}`), чем у источника
   замечания. Запрос `jsonb_array_elements(d->'sources')` раньше давал ноль
   строк, теперь вернёт разделы — и «количество замечаний» тихо превратится в
   «количество разделов». Варианты: (а) переименовать агрегат в `diag_result` в
   `guideline_sources` (правка medkard, §4.6/§5.1) — тогда в промпт достаточно
   добавить строку; (б) оставить `sources` и переписать абзац промпта, явно
   разделив уровни. **Рекомендация — (а)**: убирает грабли не только у аналитика.
2. **Аспект `criteria` неразличим машинно.** Его замечания льются в тот же
   плоский `issues[]` с текстовым префиксом «Критерий качества:», а промпт
   аналитика прямо запрещает группировать по тексту («нет тега → группируй по
   `icd_code`»). **Рекомендация:** добавить в JSON замечания поле `aspect`
   (`anamnesis|inspection|treatment|criteria`) — по строке в
   `DiagnosisIssue.to_dict` и `parse_diagnosis` — и разрез по аспектам
   становится обычным SQL. Иначе новый аспект в аналитике не отделить.
3. **`errors[]` — признак частичного аудита.** Карта остаётся `status='done'`,
   хотя пул аспекта был пуст или судья упал. Без строки в промпте аналитик
   посчитает её как «замечаний нет». Добавить: непустой `diag_result[].errors[]`
   = аудит деградировал, это не «нет замечаний».

### 13.3 Данные идут ровно, но непрерывность рвётся

- `diag_result` растёт: агрегат источников, `cite` до 300 символов на замечание,
  четыре аспекта вместо трёх. Полный full-replace (`importer.run_replica`) тянет
  `MEDCHECK_EXPORT_PAGE_SIZE=5000` строк одним запросом с таймаутом 60 с
  (`client.py` `_TIMEOUT`). Если страница начнёт упираться в таймаут — снизить
  env; кода это не касается.
- Broken-карты диагноз-контура исчезают (DoD §12.3), а `check_updates` их не
  отдаёт (`broken = FALSE`, `api_formatter.py:157`). Карты, вечно висевшие у
  «Искры» в `pending`, начнут приходить как `done` — знаменатели в отчётах
  аналитика поедут вверх. Это лечение, а не регресс, но в разрезы «до/после» оно
  попадёт.
- Число замечаний на карту вырастет за счёт аспекта `criteria`: сравнения
  «месяц к месяцу» через дату выкатки некорректны. Дату выкатки записать в
  `docs/revision-log.md`, чтобы разрыв был объясним.
