# RAG Retrieval

Overview of the retrieval-augmented generation stack used by the checker agents to look up clinical guidelines.

---

## Embeddings — `RAG/retrieval/embeddings.py`

### Provider selection

The embedding backend is chosen at startup via `EMBEDDING_PROVIDER` env var:

| Value | Class | Backend |
|---|---|---|
| `openai` (default) | `OpenAIEmbeddingAdapter` | OpenAI-compatible REST API |
| `st` | `SentenceTransformersAdapter` | Local `sentence-transformers` |
| `fastembed` | `FastEmbedAdapter` | Local ONNX via `fastembed` |

The singleton adapter is returned by `get_adapter()` (cached with `@lru_cache`).

### `embed(text) -> list[float]`

Top-level function. Delegates to the configured adapter. Returns a normalised float vector of length `EMBEDDING_DIM` (default `1024`, from `EMBEDDING_MODEL` / `EMBEDDING_DIM` env vars).

### `OpenAIEmbeddingAdapter.embed`

```python
response = await client.embeddings.create(
    model=EMBEDDING_MODEL,
    input=text,
    dimensions=EMBEDDING_DIM,
)
return response.data[0].embedding
```

### Local adapters (`st`, `fastembed`)

Both run `encode` / `embed` in a thread executor to avoid blocking the event loop. No network call.

---

## Vector store — `RAG/retrieval/vector_store.py`

Postgres + pgvector backend. Uses `asyncpg` with a lazy-initialised connection pool.

### Schema

The `docs` table stores:
- `chunk` — raw text or serialised table rows
- `fact_q_embedding`, `procedure_q_embedding`, `constraint_q_embedding` — HNSW-indexed vectors of the three hypothetical queries generated during ingestion (reverse HyDE)
- `metadata` — JSONB with section, page, file_id, etc.
- `fact_q`, `procedure_q`, `constraint_q` — the query strings themselves

### `hybrid_search(query_text, embedding, top_k) -> list[dict]`

General-purpose hybrid retrieval across the entire docs table:

1. Fetches `top_k × CANDIDATES_FACTOR` (default ×6) rows by cosine distance on the column matching `query_type` (`fact_q_embedding` / `procedure_q_embedding` / `constraint_q_embedding`).
2. Re-ranks the same candidate set with `BM25Okapi` (tokenised via Natasha for Russian text).
3. Merges vector rank and BM25 rank with **Reciprocal Rank Fusion** (`RRF_K=50`).
4. If `RERANK_BASE_URL` and `RERANK_MODEL` are configured, sends a bounded
   candidate set to a separate vLLM `/rerank` endpoint and returns its top `top_k`.
5. Returns the top `top_k` results sorted by reranker score, or by `rrf_score`
   when reranking is disabled/unavailable.

Result dict shape:
```python
{
    "id":           str,
    "chunk":        str,
    "metadata":     dict,
    "fact_q":       str | None,
    "procedure_q":  str | None,
    "constraint_q": str | None,
    "rrf_score":    float,
    "rerank_score": float,  # present when optional reranking is enabled
}
```

### `_vector_search_filtered(embedding, file_id, limit, section_filter)`

Internal — same cosine search but adds `file_id = $2` and optional `section` LIKE filter. Used by `searches.py`.

---

## Targeted searches — `RAG/retrieval/searches.py`

`search_in_guideline(query, file_id, candidates=…, top_k=…)` используется
графом аудита диагноза. Он выполняет поиск по всему документу без фильтра по
названию раздела, объединяет vector/BM25 через RRF и применяет реранкер.
Результаты нескольких вопросов дедуплицируются и ограничиваются уже в узле
графа. `get_section_chunks_by_pattern` отдельно читает все части таблицы
критериев качества в порядке страницы, таблицы и чанка. Узел графа
реконструирует из JSON-батчей одну Markdown-таблицу без ограничения количества
чанков.

Старые секционные функции `search_anamnesis` / `search_inspection` /
`search_treatment` удалены вместе с ReAct-инструментами диагноз-контура.

---

## LangChain tools — `LLM/tools.py`

Диагноз-контур больше не использует LangChain tools и ReAct. В `LLM/tools.py`
остаются только инструменты ICD-чекера для чтения структуры и разделов КР.
`GetGuidelineStructureTool` и `ReadGuidelineSectionTool` по-прежнему нужны
ICD-ReAct-чекеру; его перенос на граф в эту работу не входит.

---

## Ingestion — reverse HyDE

During ingestion (`LLM/query_generator.py`), each chunk from `PDFContentReader.iter_chunks()` is passed to `generate_queries()`, which asks the LLM to generate three hypothetical questions (`fact_query`, `procedural_query`, `constraint_query`) that a user might type to retrieve that chunk. These queries are embedded and stored alongside the chunk. At retrieval time, the incoming query is embedded and matched against these stored question embeddings — improving recall for domain-specific Russian medical terminology where direct chunk similarity is weak.
