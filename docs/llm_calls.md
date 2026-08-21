# LLM Calls

Overview of every place the codebase makes an outbound LLM request, what it calls, and what it returns.

---

## validations.py — `validate_visit`

**File:** `LLM/validations.py`  
**Called from:** `FormalValidator.validate()`  
**Client:** `instructor.AsyncInstructor` (wraps `AsyncOpenAI`)  
**Call:**

```python
result, completion = await client.chat.completions.create_with_completion(
    model=MODEL,
    response_model=_Findings,
    messages=[system, user],
    temperature=0.7,
)
```

**Input:** system prompt (rules injected) + full visit JSON as user message.  
**Output:** `list[{"flag": str, "issue": str}]` — structured findings parsed by instructor into `_Findings`.  
**Token access:** `completion.usage.prompt_tokens`, `completion.usage.completion_tokens`.

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
попадает в `errors`. Токены не теряются даже при ошибке парсинга.

`rag_agent.py` и `LLMClient.call_agent` остаются для ICD-чекера, который не
входит в граф аудита диагноза.

---

## Model configuration

All LLM calls resolve the model from `LLM_MODEL` env var (default varies by module). The OpenAI base URL can be overridden via `OPENAI_BASE_URL` for self-hosted inference (vLLM, LM Studio, etc.).
