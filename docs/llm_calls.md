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

## rag_agent.py — checker agents (`agent.ainvoke`)

**File:** `LLM/rag_agent.py`, called from `audit/diagnosis/validator.py`  
**Called from:** `_run_checker()` — three times per diagnosis (anamnesis, inspection, treatment) in parallel.  
**Client:** LangChain `ChatOpenAI` via `create_react_agent` (LangGraph ReAct agent)
**Call:**

```python
result = await agent.ainvoke(
    {"messages": [("user", human_message)]},
    config={"recursion_limit": AGENT_MAX_STEPS},
)
```

**Input:** system prompt from one of `anamnesis_checker.txt`, `inspection_checker.txt`, `treatment_checker.txt` + a combined user message containing patient info, diagnosis, and examination data.  
**Output:** `result["structured_response"]` when native JSON schema succeeds; raw content is only a fallback.
**Structured output:** `create_react_agent(response_format=...)` uses native provider JSON schema. For the current Qwen/vLLM endpoint, `chat_template_kwargs.enable_thinking=false` is also sent; legacy `guided_json` is not the primary mode.
**Safeguards:** `ToolCallGuard` limits duplicate calls, total tool calls, and tool-result size. A single `GraphRecursionError` retry switches to compact limits.
**Telemetry:** events are written to `logs/llm_observability.jsonl`; see [docs/llm-observability.md](llm-observability.md) and [docs/vllm-configuration.md](vllm-configuration.md).

---

## Model configuration

All LLM calls resolve the model from `LLM_MODEL` env var (default varies by module). The OpenAI base URL can be overridden via `OPENAI_BASE_URL` for self-hosted inference (vLLM, LM Studio, etc.).
