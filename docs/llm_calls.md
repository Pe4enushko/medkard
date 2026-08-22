# LLM Calls

Overview of every place the codebase makes an outbound LLM request, what it calls, and what it returns.

---

## validations.py — `validate_rule`

**File:** `LLM/validations.py`  
**Called from:** `FormalValidator.validate()`  
**Client:** `LLMClient` with JSON Schema structured output
**Call pattern:** one concurrent request per deterministically selected rule.

```python
raw, tokens = await client.call(
    response_model=_Findings,
    messages=[common_system_prompt, complete_visit_json, one_rule],
    temperature=0.0,
)
```

The first two messages are byte-identical for all rules of one card, so the
provider can reuse their prompt prefix. The varying rule is always last.

**Output:** established findings array: `[]` or
`[{"flag": "...", "issue": "...", "comment": "..."}]`.
The prompt tells the model to return an empty array unless the single supplied
rule is breached. Python attaches the trusted `flag_code` for that rule instead
of trusting a generated flag value. Tokens from all atomic calls are summed.
`validate_visit` remains only as a compatibility helper for direct callers/tests;
`FormalValidator` does not use the monolithic path.

---

## visit_classifier.py — `VisitClassifier.classify`

**File:** `LLM/visit_classifier.py`  
**Called from:** `FormalValidator.validate()` (via `get_visit_types` if NMU/keyword detection is inconclusive — currently always called)  
**Client:** raw `AsyncOpenAI`  
**Call:**

```python
resp = await client.chat.completions.create(
    model=MODEL,
    messages=[system_prompt, visit_json],
    temperature=0.7,
)
```

**Input:** system prompt from `visit_type_classifier.txt` + visit JSON.  
**Output:** one of `"primary"`, `"repeat"`, `"prophylactic"`, or `None` if label not recognised.  
**Token access:** `resp.usage.prompt_tokens`, `resp.usage.completion_tokens`.

---

## decider.py — `decide_file_id`

**File:** `LLM/decider.py`  
**Called from:** `ClinicRecs.pick_recs()` when multiple manifest rows match and BM25 scores are all zero.  
**Client:** raw `AsyncOpenAI`  
**Call:**

```python
resp = await client.chat.completions.create(
    model=MODEL,
    messages=[system, user],
    temperature=0.4,
)
```

**Input:** patient JSON + diagnosis JSON + list of candidate manifest rows.  
**Output:** a single `ID` string from the candidates list, or `None` if the model returns an unrecognised value.  
**Token access:** `resp.usage.prompt_tokens`, `resp.usage.completion_tokens`.

---

## icd_prefix_picker.py — `IcdPrefixPicker.pick`

**File:** `LLM/icd_prefix_picker.py`  
**Called from:** `ClinicRecs.pick_recs()` when exact match fails but prefix match returns candidates.  
**Client:** raw `AsyncOpenAI`  
**Call:**

```python
resp = await client.chat.completions.create(
    model=MODEL,
    messages=[system_prompt, user],
    temperature=0.4,
)
```

**Input:** system prompt from `icd_prefix_picker.txt` + patient JSON + diagnosis JSON + prefix-matched candidates.  
**Output:** a single `ID` string, or `None` if the model returns `"none"` or an unrecognised value.  
**Token access:** `resp.usage.prompt_tokens`, `resp.usage.completion_tokens`.

---

## query_generator.py — `generate_queries`

**File:** `LLM/query_generator.py`  
**Called from:** PDF ingestion pipeline (`scripts/ingest-pdfs.py`), not the audit pipeline.  
**Client:** `instructor.AsyncInstructor`  
**Call:**

```python
queries, completion = await client.chat.completions.create_with_completion(
    model=MODEL,
    response_model=HypotheticalQueries,
    messages=[{"role": "user", "content": rendered_prompt}],
    extra_body={"guided_json": _JSON_SCHEMA},
)
```

**Input:** chunk content (text or serialised table JSON) injected into `chunk_query_generator.txt`.  
**Output:** `HypotheticalQueries(fact_query, procedural_query, constraint_query)` — three hypothetical questions a user might ask to retrieve this chunk (reverse HyDE).  
**Token access:** `completion.usage.prompt_tokens`, `completion.usage.completion_tokens`.

---

## Граф аудита диагноза — structured calls

`LLM/graphs/diagnosis_nodes.py` вызывает `LLMClient.call` без tools:

- `generate_questions` → `QuestionSet`;
- `extract_drugs` → `DrugList`;
- `judge_anamnesis`, `judge_inspection`, `judge_treatment`, `judge_criteria`
  → `JudgeOutput` с числовыми `chunk_refs`.

Все ответы дополнительно проходят `model_validate_json` в узле. Обрезанный или
несоответствующий схеме ответ деградирует только соответствующую ветвь и
попадает в `errors`. Исключение — `generate_questions`: его статический fallback
требует непустой блок диагноза; без него ошибка выходит из графа, и карта
сохраняется как `broken`. Токены не теряются даже при ошибке парсинга.

## icd_check/validator.py — проверка кодирования МКБ-10

**File:** `audit/icd_check/validator.py`

Два вызова-этапа с раздельным контекстом, ReAct-цикла нет.

**Этап 1 — отбор гипотез** (`_pick_candidates`, промпт `icd_candidate_picker.txt`):
один вызов видит приём и перечень клинических рекомендаций целиком
(отфильтрованный по возрасту) и называет не больше трёх рекомендаций, которые
стоит прочитать. Перечень уходит модели ровно один раз: в ReAct он лежал в
человеческом сообщении и переотправлялся с историей на каждом шаге — 558 строк
≈ 18 тыс. токенов, помноженные на число шагов, и есть те 182 532 токена
ICD-агента на карте `8b809667` в прогоне 21.08.

Отбирать просят по клинической картине, а не по коду врача: чекер ищет код,
которого в карте нет, и более подходящий часто лежит в другом разделе МКБ
(`R51` головная боль → `G43.0` мигрень). `file_id` вне перечня отбрасывается.

**Этап 2 — проверка гипотез** (`_judge_candidate`, промпт `icd_code_judge.txt`):
разделы названных рекомендаций читаются кодом (`_pick_sections` — нумерация
«1.x» шаблона Минздрава плюс ключевые слова: определение, кодирование по
МКБ-10, классификация, критерии установления диагноза), каждая гипотеза судится
отдельным вызовом и не видит остальных. Число вызовов известно до начала
работы, зацикливаться нечему.

`suggested_code` обязан быть одним из кодов той рекомендации, по тексту которой
судили: код вне её перечня проверить нашим же контуром нечем.

Вывод — рекомендация другого кода с обоснованием и цитатой, а не вердикт об
ошибке: чекер видит карту и справочник, но не пациента. Порог `confidence ≥ 8`.
По рекомендованному коду граф аудита диагноза не запускается — это гипотеза, а
не диагноз визита.

Отказ чекера не ломает карту: формальный и диагнозный контуры доезжают,
`icd_check_result` остаётся NULL, и «мнения нет» отличимо от «замечаний нет»
одним SQL-запросом. `None` возвращается, если первый этап не ответил по
контракту или если ни одна гипотеза не дошла до суждения.

`rag_agent.py`, `LLM/tools.py` и `LLMClient.call_agent` после этого не имеют
вызывающих — см. `docs/tech-debt.md`.

---

## Model configuration

All LLM calls resolve the model from `LLM_MODEL` env var (default varies by module). The OpenAI base URL can be overridden via `OPENAI_BASE_URL` for self-hosted inference (vLLM, LM Studio, etc.).
