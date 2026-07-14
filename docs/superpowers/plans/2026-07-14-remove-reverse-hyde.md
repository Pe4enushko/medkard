# Remove Reverse HyDE → Contextual Chunk Embeddings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace reverse-HyDE (3 hypothetical-query embeddings per chunk) with a single embedding of the chunk's contextual text (`section` + body), across ingestion, retrieval, storage, and schema.

**Architecture:** One `docs.embedding` column (VECTOR(1024)) replaces six HyDE columns. Ingestion embeds `[section]\n<chunk>` instead of generating/embedding LLM queries. Retrieval collapses the `fact/procedure/constraint` `query_type` dimension (only `fact` was ever used) into a single vector column; BM25+RRF hybrid stays. A `--force-all` reingest flag re-embeds the whole corpus.

**Tech Stack:** Python 3, pytest (pythonpath=src, asyncio_mode=auto), psycopg3 + pgvector (storage), asyncpg + pgvector (retrieval), fastembed (embeddings), Postgres/pgvector, migrate.sh (idempotent forward-only SQL).

## Global Constraints

- Embedding model and vector dimension UNCHANGED: Qwen3-Embedding-0.6B, `VECTOR(1024)`.
- Migrations are idempotent and forward-only (`IF EXISTS` / `IF NOT EXISTS`); no down-migration.
- Embedded text = `section` (when present) + chunk body ONLY. No Наименование / МКБ / возраст / `фрагмент N`.
- `Doc._format_chunk()` (LLM display formatting) MUST NOT change — it still shows name/МКБ/section/fragment.
- Keep the BM25 + vector RRF hybrid (natasha / rank_bm25 / RRF stay).
- Use `python3` (not `python`); run tests with `pytest` from the worktree root.
- Comments terse, English, only where non-obvious.

---

### Task 1: Migration 024 — single embedding column

**Files:**
- Create: `migrations/024_docs_single_embedding.sql`
- Test: `tests/test_migration_024.py`

**Interfaces:**
- Produces: table `docs` with columns dropped (`fact_q`, `procedure_q`, `constraint_q`, `fact_q_embedding`, `procedure_q_embedding`, `constraint_q_embedding`) and added `embedding VECTOR(1024)`; index `docs_embedding_idx`.

- [ ] **Step 1: Write the failing test** (text-level guard; DB idempotence is stand-only)

Create `tests/test_migration_024.py`:

```python
from pathlib import Path

_SQL = (Path(__file__).resolve().parent.parent
        / "migrations" / "024_docs_single_embedding.sql").read_text(encoding="utf-8")


def test_drops_all_six_hyde_columns():
    for col in ("fact_q", "procedure_q", "constraint_q",
                "fact_q_embedding", "procedure_q_embedding", "constraint_q_embedding"):
        assert f"DROP COLUMN IF EXISTS {col}" in _SQL


def test_adds_single_embedding_column_and_index():
    assert "ADD COLUMN IF NOT EXISTS embedding VECTOR(1024)" in _SQL
    assert "docs_embedding_idx" in _SQL
    assert "hnsw (embedding vector_cosine_ops)" in _SQL


def test_drops_old_indexes():
    for idx in ("docs_fact_q_embedding_idx",
                "docs_procedure_q_embedding_idx",
                "docs_constraint_q_embedding_idx"):
        assert f"DROP INDEX IF EXISTS {idx}" in _SQL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_migration_024.py -v`
Expected: FAIL (FileNotFoundError — migration does not exist yet).

- [ ] **Step 3: Create the migration**

Create `migrations/024_docs_single_embedding.sql`:

```sql
-- 024_docs_single_embedding.sql
-- Remove reverse-HyDE columns; store one embedding of the chunk's contextual text.
-- VECTOR dim stays 1024 (Qwen/Qwen3-Embedding-0.6B). Forward-only, idempotent.

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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_migration_024.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add migrations/024_docs_single_embedding.sql tests/test_migration_024.py
git commit -m "feat(db): migration 024 — single docs.embedding, drop HyDE columns"
```

**Stand-only verification (record, run on stand):** `bash migrations/migrate.sh` twice → no error (idempotent); `\d docs` shows `embedding` present and no `*_q*` columns.

---

### Task 2: Doc model + DocsStorage — single embedding

**Files:**
- Modify: `src/storage/models/doc.py:20-28` (query fields → `embedding`)
- Modify: `src/storage/docs_storage.py` (`_INSERT_DOC_SQL`, `_doc_params`, `_row_to_doc`, SELECTs in `get`/`get_many`)
- Test: `tests/test_docs_storage_params.py` (new, no-DB), `tests/test_doc_format_chunk.py` (unchanged, must stay green)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `Doc.embedding: list[float] | None` (replaces `fact_q`/`procedure_q`/`constraint_q` and their `*_q_embedding` fields).
  - `docs_storage._doc_params(doc) -> dict` with keys exactly `{file_id, chunk, metadata, embedding}`.
  - `docs_storage._row_to_doc(row) -> Doc` reading `row["embedding"]` (via `.get`), no `*_q`.
  - `_INSERT_DOC_SQL` inserts `(file_id, chunk, metadata, embedding)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_docs_storage_params.py`:

```python
from storage import docs_storage
from storage.models import Doc


def test_doc_params_keys_are_minimal():
    doc = Doc(file_id="F1", chunk="txt", metadata={"section": "1.1"}, embedding=[0.1, 0.2])
    params = docs_storage._doc_params(doc)
    assert set(params) == {"file_id", "chunk", "metadata", "embedding"}
    assert params["embedding"] == [0.1, 0.2]


def test_insert_sql_mentions_embedding_not_hyde():
    sql = docs_storage._INSERT_DOC_SQL
    assert "embedding" in sql
    for gone in ("fact_q", "procedure_q", "constraint_q"):
        assert gone not in sql


def test_row_to_doc_reads_embedding():
    row = {"id": "u1", "file_id": "F1", "chunk": "c", "metadata": {},
           "embedding": [0.3], "g_name": "N", "g_mkb": ["I63"], "g_age_category": []}
    doc = docs_storage._row_to_doc(row)
    assert doc.embedding == [0.3]
    assert doc.name == "N" and doc.mkb == ["I63"]
    assert not hasattr(doc, "fact_q")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_docs_storage_params.py -v`
Expected: FAIL (`Doc.__init__` rejects `embedding=` / still has `fact_q`).

- [ ] **Step 3: Edit `src/storage/models/doc.py`**

Replace the HyDE field block (lines ~20-28):

```python
    # Hypothetical queries (reverse HyDE)
    fact_q: str | None = None
    procedure_q: str | None = None
    constraint_q: str | None = None

    # Embedding vectors — populated for writes, not fetched on reads.
    fact_q_embedding: list[float] | None = None
    procedure_q_embedding: list[float] | None = None
    constraint_q_embedding: list[float] | None = None
```

with:

```python
    # Embedding of the chunk's contextual text (section + body).
    # Populated for writes, not fetched on reads.
    embedding: list[float] | None = None
```

Leave `_format_chunk()` untouched.

- [ ] **Step 4: Edit `src/storage/docs_storage.py`**

Replace `_INSERT_DOC_SQL`:

```python
_INSERT_DOC_SQL = """
    INSERT INTO docs (file_id, chunk, metadata, embedding)
    VALUES (%(file_id)s, %(chunk)s, %(metadata)s, %(embedding)s)
    RETURNING id::text
"""
```

Replace `_doc_params`:

```python
def _doc_params(doc: Doc) -> dict:
    return {
        "file_id": doc.file_id,
        "chunk": doc.chunk,
        "metadata": json.dumps(doc.metadata),
        "embedding": doc.embedding,
    }
```

Replace `_row_to_doc` (drop `*_q`):

```python
def _row_to_doc(row: dict) -> Doc:
    return Doc(
        id=row["id"],
        file_id=row["file_id"],
        chunk=row["chunk"],
        metadata=row["metadata"],
        name=row.get("g_name"),
        mkb=list(row.get("g_mkb") or []),
        age_category=list(row.get("g_age_category") or []),
    )
```

In `get()` and `get_many()` SELECTs, delete the line `docs.fact_q, docs.procedure_q, docs.constraint_q,` (both queries). The `embedding` column is intentionally NOT selected on reads (large, unused by consumers).

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_docs_storage_params.py tests/test_doc_format_chunk.py -v`
Expected: PASS (new file 3 tests; format-chunk tests still green).

- [ ] **Step 6: Commit**

```bash
git add src/storage/models/doc.py src/storage/docs_storage.py tests/test_docs_storage_params.py
git commit -m "refactor(storage): Doc.embedding single column, drop HyDE fields"
```

---

### Task 3: Ingestion pipeline — embed contextual text

**Files:**
- Modify: `src/RAG/ingestion/pipeline.py` (imports, add `embed_text`, rewrite `process_chunk`)
- Test: `tests/test_ingest_pipeline.py` (rewrite HyDE-based tests)

**Interfaces:**
- Consumes: `Doc.embedding` (Task 2); `RAG.retrieval.embeddings.embed(text) -> list[float]`.
- Produces:
  - `pipeline.embed_text(chunk: dict) -> str` — `f"[{section}]\n{body}"` when `metadata.section` truthy, else `body`.
  - `pipeline.process_chunk(chunk, file_id) -> Doc | None` — embeds `embed_text(chunk)`, returns `Doc(file_id, chunk=body, metadata, embedding=vec)`; `None` on embed error.
  - `pipeline.process_batch(chunks, file_id)` — unchanged signature.

- [ ] **Step 1: Rewrite the test file**

Replace the entire contents of `tests/test_ingest_pipeline.py`:

```python
from RAG.ingestion import pipeline


def test_chunk_text_passthrough_str():
    assert pipeline.chunk_text({"content": "hello"}) == "hello"


def test_chunk_text_serializes_list_as_json():
    assert pipeline.chunk_text({"content": [{"a": "1"}]}) == '[{"a": "1"}]'


def test_embed_text_prepends_section():
    chunk = {"content": "body", "metadata": {"section": "3.1.1 Диагностика"}}
    assert pipeline.embed_text(chunk) == "[3.1.1 Диагностика]\nbody"


def test_embed_text_without_section_is_body_only():
    assert pipeline.embed_text({"content": "body", "metadata": {}}) == "body"


def test_embed_text_table_serialized():
    chunk = {"content": [{"a": "1"}], "metadata": {"section": "S"}}
    assert pipeline.embed_text(chunk) == '[S]\n[{"a": "1"}]'


async def test_process_chunk_builds_doc_with_embedding(monkeypatch):
    captured = {}

    async def fake_embed(text):
        captured["text"] = text
        return [0.5, 0.6]

    monkeypatch.setattr(pipeline, "embed", fake_embed)

    chunk = {"content": "txt", "metadata": {"section": "1.1", "content_type": "text", "chunk_index": 0}}
    doc = await pipeline.process_chunk(chunk, "F1")

    assert doc.file_id == "F1"
    assert doc.chunk == "txt"                       # stored body is raw chunk text
    assert doc.embedding == [0.5, 0.6]
    assert captured["text"] == "[1.1]\ntxt"          # embedded text is contextual
    assert doc.metadata == {"section": "1.1", "content_type": "text", "chunk_index": 0}


async def test_process_chunk_returns_none_on_embed_error(monkeypatch):
    async def boom(text):
        raise RuntimeError("embed down")

    monkeypatch.setattr(pipeline, "embed", boom)
    doc = await pipeline.process_chunk({"content": "t", "metadata": {"section": "1.1"}}, "F1")
    assert doc is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingest_pipeline.py -v`
Expected: FAIL (`pipeline.embed_text` undefined; `pipeline.embed` attribute missing).

- [ ] **Step 3: Rewrite `src/RAG/ingestion/pipeline.py`**

Replace the whole file with:

```python
"""pipeline.py — shared per-chunk ingest pipeline: chunk → contextual embedding → Doc.

Shared by scripts/ingest-pdfs.py and scripts/reingest-pdfs.py. Embeds the chunk's
contextual text ("[section]\\n<body>") — NOT hypothetical queries (reverse HyDE removed).
"""
import asyncio
import json
import logging

from RAG.retrieval.embeddings import embed
from storage.models import Doc

log = logging.getLogger(__name__)


def chunk_text(chunk: dict) -> str:
    content = chunk["content"]
    if isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    return content


def embed_text(chunk: dict) -> str:
    """Contextual text to embed: section header (if any) + chunk body."""
    section = (chunk.get("metadata") or {}).get("section")
    body = chunk_text(chunk)
    return f"[{section}]\n{body}" if section else body


async def process_chunk(chunk: dict, file_id: str) -> Doc | None:
    """Embed the chunk's contextual text; return a ready-to-insert Doc (None on embed error)."""
    body = chunk_text(chunk)
    try:
        vector = await embed(embed_text(chunk))
    except Exception as exc:
        meta = chunk.get("metadata", {})
        log.error(
            "Embedding failed for %s [%s #%s section=%r]: %s",
            file_id, meta.get("content_type"), meta.get("chunk_index"),
            meta.get("section"), exc,
        )
        return None

    return Doc(
        file_id=file_id,
        chunk=body,
        metadata=chunk["metadata"],
        embedding=vector,
    )


async def process_batch(chunks: list[dict], file_id: str) -> list[Doc | None]:
    """Process a batch of chunks concurrently."""
    return list(await asyncio.gather(*[process_chunk(c, file_id) for c in chunks]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingest_pipeline.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/RAG/ingestion/pipeline.py tests/test_ingest_pipeline.py
git commit -m "refactor(ingest): embed contextual chunk text, drop HyDE query generation"
```

---

### Task 4: Retrieval — collapse query_type into single embedding column

**Files:**
- Modify: `src/RAG/retrieval/vector_store.py` (remove `QueryType`, `_EMBEDDING_COL`, `search_fact/search_procedure/search_constraint`; column → `embedding`; `hybrid_search` drops `query_type`; `_SELECT_COLS` drops `*_q`)
- Test: `tests/test_vector_store_api.py` (new, no-DB: import + signature/introspection guards)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `vector_store.hybrid_search(query_text: str, embedding: list[float], top_k: int = 10) -> list[dict]` (NO `query_type`).
  - `vector_store._vector_search(embedding, limit) -> list[dict]` (column fixed to `embedding`).
  - `vector_store._vector_search_filtered(embedding, file_id, limit, section_filter=None) -> list[dict]` (column fixed to `embedding`; `col` param removed).
  - `search_fact` / `search_procedure` / `search_constraint` / `QueryType` / `_EMBEDDING_COL` REMOVED.

- [ ] **Step 1: Write the failing test**

Create `tests/test_vector_store_api.py`:

```python
import inspect

from RAG.retrieval import vector_store


def test_hybrid_search_has_no_query_type():
    params = inspect.signature(vector_store.hybrid_search).parameters
    assert "query_type" not in params
    assert set(params) >= {"query_text", "embedding", "top_k"}


def test_query_type_machinery_removed():
    assert not hasattr(vector_store, "search_fact")
    assert not hasattr(vector_store, "search_procedure")
    assert not hasattr(vector_store, "search_constraint")
    assert not hasattr(vector_store, "_EMBEDDING_COL")


def test_select_cols_have_no_hyde():
    for gone in ("fact_q", "procedure_q", "constraint_q"):
        assert gone not in vector_store._SELECT_COLS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vector_store_api.py -v`
Expected: FAIL (`hybrid_search` still has `query_type`; `search_fact` exists).

- [ ] **Step 3: Edit `src/RAG/retrieval/vector_store.py`**

3a. Delete `QueryType` (line ~46) and the `_EMBEDDING_COL` dict (lines ~48-52).

3b. Replace `_SELECT_COLS` (lines ~54-61) with:

```python
_SELECT_COLS = """
    id::text,
    chunk,
    metadata
"""
```

3c. Replace `_vector_search` — drop the `col` param, fix column to `embedding`:

```python
async def _vector_search(embedding: list[float], limit: int) -> list[dict]:
    """Fetch rows closest to *embedding* in the embedding column (cosine distance)."""
    pool = await _get_pool()
    vec = np.array(embedding, dtype=np.float32)
    where_sql = " AND ".join(["embedding IS NOT NULL", *_chunk_text_exclusion_clauses()])
    rows = await pool.fetch(
        f"""
        SELECT {_SELECT_COLS},
            embedding <=> $1 AS distance
        FROM docs
        WHERE {where_sql}
        ORDER BY distance ASC
        LIMIT $2
        """,
        vec,
        limit,
    )
    return [dict(r) for r in rows]
```

3d. Replace `_vector_search_filtered` — drop `col` param, fix column to `embedding`:

```python
async def _vector_search_filtered(
    embedding: list[float],
    file_id: str,
    limit: int,
    section_filter: str | None = None,
) -> list[dict]:
    """Fetch rows by cosine distance with file_id, optional section, and text filters."""
    pool = await _get_pool()
    vec = np.array(embedding, dtype=np.float32)

    where_clauses = [
        "embedding IS NOT NULL",
        *_chunk_text_exclusion_clauses(),
        "file_id = $2",
    ]
    params: list = [vec, file_id]

    if section_filter:
        params.append(f"%{section_filter}%")
        where_clauses.append(f"lower(metadata->>'section') LIKE ${len(params)}")

    where_sql = " AND ".join(where_clauses)

    rows = await pool.fetch(
        f"""
        SELECT {_SELECT_COLS},
               embedding <=> $1 AS distance
        FROM docs
        WHERE {where_sql}
        ORDER BY distance ASC
        LIMIT ${len(params) + 1}
        """,
        *params,
        limit,
    )
    return [dict(r) for r in rows]
```

3e. Delete `search_fact`, `search_procedure`, `search_constraint` (lines ~181-202) entirely.

3f. Replace `hybrid_search` signature and body head — remove `query_type`:

```python
async def hybrid_search(
    query_text: str,
    embedding: list[float],
    top_k: int = 10,
) -> list[dict]:
    """Hybrid retrieval: HNSW vector search → BM25 rerank → RRF fusion.

    Args:
        query_text:  Raw query string used for BM25 lexical scoring.
        embedding:   Query embedding vector (must match EMBEDDING_DIM).
        top_k:       Number of results to return.

    Returns:
        List of dicts with keys: id, chunk, metadata, rrf_score. Sorted by rrf_score desc.
    """
    n_candidates = top_k * CANDIDATES_FACTOR

    candidates = await _vector_search(embedding, n_candidates)
    if not candidates:
        logger.info(
            "🔎 [retrieval] hybrid_search found no chunks top_k=%d query=%r",
            top_k, query_text,
        )
        return []
```

Keep the rest of `hybrid_search` (vector_ranking / bm25 / rrf / assembly), but update the
`_log_hybrid_chunks(...)` call to drop the `query_type=` argument, and update
`_log_hybrid_chunks` to remove its `query_type` parameter and the `f"query_type: {query_type}"`
line. (The `ValueError` guard on `query_type` is deleted with the signature change.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vector_store_api.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/RAG/retrieval/vector_store.py tests/test_vector_store_api.py
git commit -m "refactor(retrieval): single embedding column, drop query_type dimension"
```

---

### Task 5: Retrieval consumers — searches, rag_agent, tools

**Files:**
- Modify: `src/RAG/retrieval/searches.py` (`get_section_chunks` SELECT drops `*_q`; module docstring)
- Modify: `src/LLM/rag_agent.py:53-58` (`hybrid_search` call drops `query_type="fact"`)
- Modify: `src/LLM/tools.py` (`_format_results` and `ReadGuidelineSectionTool._arun` drop `fact_q=/procedure_q=/constraint_q=` from `Doc(...)`)
- Test: `tests/test_retrieval_consumers_import.py` (new, no-DB import smoke)

**Interfaces:**
- Consumes: `hybrid_search(query_text, embedding, top_k)` (Task 4); `Doc` without `*_q` (Task 2).
- Produces: no new public API; call sites made consistent.

- [ ] **Step 1: Write the failing test**

Create `tests/test_retrieval_consumers_import.py`:

```python
import inspect


def test_rag_agent_retrieve_has_no_query_type_literal():
    from LLM import rag_agent
    src = inspect.getsource(rag_agent.retrieve.func)  # @tool wraps; .func is the coroutine
    assert "query_type" not in src


def test_tools_module_constructs_doc_without_hyde():
    from LLM import tools
    src = inspect.getsource(tools)
    for gone in ("fact_q=", "procedure_q=", "constraint_q="):
        assert gone not in src


def test_get_section_chunks_select_has_no_hyde():
    from RAG.retrieval import searches
    src = inspect.getsource(searches.get_section_chunks)
    for gone in ("fact_q", "procedure_q", "constraint_q"):
        assert gone not in src
```

Note: if `rag_agent.retrieve.func` is not the right attribute for the `@tool`-decorated coroutine, use `inspect.getsource(rag_agent)` and assert the module source contains no `query_type` after edits. Prefer the whole-module form if unsure:

```python
def test_rag_agent_retrieve_has_no_query_type_literal():
    from LLM import rag_agent
    assert "query_type" not in inspect.getsource(rag_agent)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retrieval_consumers_import.py -v`
Expected: FAIL (`query_type="fact"` still in rag_agent; `fact_q=` still in tools; `*_q` still in `get_section_chunks`).

- [ ] **Step 3a: Edit `src/LLM/rag_agent.py`**

Replace the `hybrid_search` call (lines ~53-58):

```python
    results = await hybrid_search(
        query_text=query,
        embedding=embedding,
        top_k=RAG_TOP_K,
    )
```

- [ ] **Step 3b: Edit `src/RAG/retrieval/searches.py`**

In `get_section_chunks`, change the SELECT list from
`SELECT id::text, chunk, metadata, fact_q, procedure_q, constraint_q`
to:

```python
        SELECT id::text, chunk, metadata
```

Update the module docstring line `same shape as hybrid_search` context is fine; no `*_q` references remain to change beyond the SELECT.

- [ ] **Step 3c: Edit `src/LLM/tools.py`**

In `_format_results`, the `Doc(...)` construction — remove the three kwargs:

```python
        doc = Doc(
            chunk=raw.get("chunk", ""),
            file_id=raw.get("file_id", ""),
            metadata=meta,
            id=raw.get("id"),
        )
```

In `ReadGuidelineSectionTool._arun`, the `Doc(...)` construction — remove the three kwargs:

```python
            doc = Doc(
                chunk=raw.get("chunk", ""),
                file_id=raw.get("file_id", file_id),
                metadata=meta,
                id=raw.get("id"),
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retrieval_consumers_import.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/LLM/rag_agent.py src/RAG/retrieval/searches.py src/LLM/tools.py tests/test_retrieval_consumers_import.py
git commit -m "refactor(retrieval): update consumers for single-embedding API"
```

---

### Task 6: reingest `--force-all` flag (re-embed rollout)

**Files:**
- Modify: `scripts/reingest-pdfs.py` (add `_forced_full_worklist` helper, `--force-all` arg, wire into `main`)
- Test: `tests/test_reingest_cli.py` (add cases)

**Interfaces:**
- Consumes: nothing new.
- Produces: `reingest._forced_full_worklist(manifest_rows: dict) -> list[tuple[str, str]]` returning `[(file_id, "full"), ...]` for every manifest id; `--force-all` argparse flag selecting it (bypasses hash/status).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reingest_cli.py`:

```python
def test_forced_full_worklist_marks_all_full():
    manifest_rows = {"A": {"ID": "A"}, "B": {"ID": "B"}}
    wl = reingest._forced_full_worklist(manifest_rows)
    assert wl == [("A", "full"), ("B", "full")]


def test_force_all_flag_parses():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-all", action="store_true")
    assert parser.parse_args(["--force-all"]).force_all is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reingest_cli.py -v`
Expected: FAIL (`reingest._forced_full_worklist` undefined).

- [ ] **Step 3: Edit `scripts/reingest-pdfs.py`**

3a. Add the helper near `_summarize`:

```python
def _forced_full_worklist(manifest_rows: dict) -> list[tuple[str, str]]:
    """Every manifested file classified as 'full' — bypasses hash/status (re-embed rollout)."""
    return [(file_id, "full") for file_id in manifest_rows]
```

3b. Add the argparse flag (next to `--file-id`):

```python
    parser.add_argument("--force-all", action="store_true",
                        help="reingest every manifested file (bypass hash/status) — e.g. after "
                             "an embedding representation change")
```

3c. Wire into `main`'s worklist selection — insert an `elif` between the `--file-id` and `else` branches:

```python
        if args.file_id:
            worklist = [(args.file_id, "full")]
        elif args.force_all:
            worklist = _forced_full_worklist(manifest_rows)
        else:
            worklist = build_worklist(manifest_rows, runs, guidelines_by_id,
                                      lambda fid: _current_hash(fid, pdfs_dir))
            if args.only_failed:
                worklist = [(fid, "full") for fid, _ in worklist
                            if runs.get(fid, (None, None))[0] == "failed"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reingest_cli.py -v`
Expected: PASS (existing 5 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add scripts/reingest-pdfs.py tests/test_reingest_cli.py
git commit -m "feat(reingest): --force-all to re-embed the whole corpus"
```

---

### Task 7: Delete dead HyDE code + fix docs

**Files:**
- Delete: `src/LLM/query_generator.py`, `src/LLM/embed_queries.py`, `src/LLM/prompts/chunk_query_generator.txt`
- Modify: `CLAUDE.md` (Project Overview, Key Design Notes "Reverse HyDE", architecture tree comments), `src/LLM/chinese_detector.py:72` (docstring mention), `src/RAG/retrieval/vector_store.py` module docstring (result-shape block), `src/RAG/retrieval/searches.py` module docstring
- Test: `tests/test_no_hyde_residue.py` (new)

**Interfaces:**
- Consumes: nothing (pipeline no longer imports the deleted modules — Task 3).
- Produces: no HyDE modules; no source imports of them.

- [ ] **Step 1: Write the failing test**

Create `tests/test_no_hyde_residue.py`:

```python
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"


def test_hyde_modules_deleted():
    assert not (_SRC / "LLM" / "query_generator.py").exists()
    assert not (_SRC / "LLM" / "embed_queries.py").exists()
    assert not (_SRC / "LLM" / "prompts" / "chunk_query_generator.txt").exists()


def test_no_source_imports_hyde_modules():
    offenders = []
    for py in _SRC.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if "query_generator" in text or "embed_queries" in text or "HypotheticalQueries" in text:
            offenders.append(py.name)
    assert offenders == [], f"HyDE references remain in: {offenders}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_no_hyde_residue.py -v`
Expected: FAIL (modules still present; `chinese_detector.py` mentions `HypotheticalQueries`).

- [ ] **Step 3: Delete modules and scrub references**

```bash
git rm src/LLM/query_generator.py src/LLM/embed_queries.py src/LLM/prompts/chunk_query_generator.txt
```

Edit `src/LLM/chinese_detector.py:72` — replace the docstring sentence mentioning
`HypotheticalQueries` with a neutral description, e.g. `Accepts a pydantic model instance or a plain dict.`

Edit `src/RAG/retrieval/vector_store.py` module docstring: remove the `fact_q / procedure_q / constraint_q` lines from the documented hybrid-result shape (now `id, chunk, metadata, rrf_score`).

Edit `src/RAG/retrieval/searches.py` module docstring: remove any `fact_q/procedure_q/constraint_q` from the documented result shape.

Edit `CLAUDE.md`:
- Project Overview: replace the sentence describing reverse HyDE with: `Core technology: LLM-based analysis (OpenAI-compatible API) combined with RAG — chunk contextual text (section + body) is embedded at ingest and matched against the query embedding, with a BM25 + vector RRF hybrid.`
- Architecture tree: change `embeddings.py # embed()` comment as-is (fine); change `searches.py # search_fact / search_procedural / search_constraint` comment to `searches.py # file/section-scoped hybrid search`.
- Key Design Notes: delete the entire `**Reverse HyDE**:` bullet; replace with `**Contextual embeddings**: each chunk is embedded from its section header + body; at retrieval the query embedding is matched against those chunk embeddings (BM25+RRF hybrid).`
- Key Data Flows → Ingestion: replace `generate_queries() ... embed_queries() → insert into docs with three embedding columns` with `embed contextual text (section + body) → insert into docs with one embedding column`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_no_hyde_residue.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Full suite regression (no-DB)**

Run: `pytest -q -k "not replace and not runs_storage" 2>&1 | tail -15`
Expected: PASS for all collectable non-DB tests (stand-only DB tests `test_docs_replace`, `test_ingest_runs_storage` are excluded by `-k`; if they error on collection due to imports, that's a bug to fix, not a skip).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: delete HyDE modules, scrub docs/comments"
```

---

## Stand Rollout (record — run on stand, not dev machine)

After merge to `dev` and deploy to stand:

```bash
bash migrations/migrate.sh                    # applies 024 (idempotent)
python3 scripts/reingest-pdfs.py --force-all  # re-embed the whole corpus
```

Then run the stand-only DB tests: `pytest tests/test_docs_replace.py tests/test_ingest_runs_storage.py`
and spot-check retrieval: `retrieve()` / `search_anamnesis` return non-empty for a known guideline.

## Self-Review

**Spec coverage:**
- Migration 024 → Task 1. ✓
- Contextual embed text (section + body, no name/МКБ/age/fragment) → Task 3 (`embed_text`). ✓
- Delete query_generator/embed_queries/prompt → Task 7. ✓
- Rewrite pipeline → Task 3. ✓
- Retrieval collapse (query_type, search_fact/proc/constr, hybrid_search) → Task 4; consumers → Task 5. ✓
- Doc model + docs_storage cleanup → Task 2. ✓
- tools.py `fact_q=` cleanup → Task 5. ✓
- `--force-all` rollout → Task 6. ✓
- Tests (no-DB + stand) → each task + Stand Rollout section. ✓

**Type consistency:** `Doc.embedding: list[float] | None` used identically in Tasks 2/3. `hybrid_search(query_text, embedding, top_k)` defined in Task 4, called in Task 5. `_vector_search_filtered(embedding, file_id, limit, section_filter=None)` — Task 4 signature matches existing `searches.py` positional call. `_forced_full_worklist(manifest_rows)` defined and tested in Task 6.

**Placeholder scan:** none — all steps carry full code.
