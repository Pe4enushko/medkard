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

### `hybrid_search(query_text, embedding, query_type, top_k) -> list[dict]`

General-purpose hybrid retrieval across the entire docs table:

1. Fetches `top_k × CANDIDATES_FACTOR` (default ×6) rows by cosine distance on the column matching `query_type` (`fact_q_embedding` / `procedure_q_embedding` / `constraint_q_embedding`).
2. Re-ranks the same candidate set with `BM25Okapi` (tokenised via Natasha for Russian text).
3. Merges vector rank and BM25 rank with **Reciprocal Rank Fusion** (`RRF_K=50`).
4. Returns the top `top_k` results sorted by `rrf_score` descending.

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
}
```

### `_vector_search_filtered(embedding, file_id, limit, section_filter, col)`

Internal — same cosine search but adds `file_id = $2` and optional `section` LIKE filter. Used by `searches.py`.

---

## Targeted searches — `RAG/retrieval/searches.py`

All four functions wrap `_hybrid_filtered`, which calls `_vector_search_filtered` (scoped to a single document) and then applies BM25 + RRF. Returns `TARGETED_TOP_K=4` results.

| Function | Section filter | Used by |
|---|---|---|
| `search_by_file_id(file_id, query)` | none | `SearchGuidelineTool` |
| `search_anamnesis(file_id, query)` | section LIKE `%жалоб%` | `SearchAnamnesisTool` |
| `search_inspection(file_id, query)` | section LIKE `%исследов%` | `SearchInspectionTool` |
| `search_treatment(file_id, query)` | section LIKE `%лечен%` | `SearchTreatmentTool` |

---

## LangChain tools — `LLM/tools.py`

Wraps the search functions above as `BaseTool` subclasses with `file_id` baked in at construction time. Each tool's `_arun(query)` calls the corresponding search function and formats results with `_format_results`.

| Tool class | Calls |
|---|---|
| `SearchGuidelineTool` | `search_by_file_id` |
| `SearchAnamnesisTool` | `search_anamnesis` |
| `SearchInspectionTool` | `search_inspection` |
| `SearchTreatmentTool` | `search_treatment` |

### Factory functions

```python
get_anamnesis_tools_for(file_id)   # [SearchAnamnesisTool, SearchGuidelineTool]
get_inspection_tools_for(file_id)  # [SearchInspectionTool, SearchGuidelineTool]
get_treatment_tools_for(file_id)   # [SearchTreatmentTool, SearchGuidelineTool]
```

Each checker agent receives its domain-specific tool plus the general `SearchGuidelineTool` as a fallback.

---

## Ingestion — reverse HyDE

During ingestion (`LLM/query_generator.py`), each chunk from `PDFContentReader.iter_chunks()` is passed to `generate_queries()`, which asks the LLM to generate three hypothetical questions (`fact_query`, `procedural_query`, `constraint_query`) that a user might type to retrieve that chunk. These queries are embedded and stored alongside the chunk. At retrieval time, the incoming query is embedded and matched against these stored question embeddings — improving recall for domain-specific Russian medical terminology where direct chunk similarity is weak.
