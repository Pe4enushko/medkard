# Reingest-with-resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `scripts/reingest-pdfs.py` that re-syncs the `docs` (PDF chunks) and `guidelines` (manifest metadata) tables to the current manifest + PDFs, resumably, re-chunking only when a PDF's hash changed.

**Architecture:** A pure classifier (`reingest_planner.classify`) decides per file between `full` re-chunk, `metadata_only` guidelines upsert, or `skip`, driven by a resume table `ingest_runs` (status + last-done PDF hash) and a manifest-vs-`guidelines` diff. The existing per-chunk ingest pipeline is extracted from `scripts/ingest-pdfs.py` into `src/RAG/ingestion/pipeline.py` so both scripts share it. Full reingest replaces a file's `docs` rows atomically (`DocsStorage.replace_by_file_id`), upserts its `guidelines` row, then marks `ingest_runs` done last so a crash mid-file safely re-runs.

**Tech Stack:** Python 3.10, psycopg3 (async, `BaseStorage` shared pool), pgvector, pymupdf/`fitz` + tabula (chunking), OpenAI-compatible LLM (hypothetical-query gen), pytest (`asyncio_mode=auto`, `pythonpath=src`).

## Global Constraints

- `pythonpath = src` (pytest.ini) — imports are `from RAG.ingestion...`, `from storage...`, no `src.` prefix.
- `asyncio_mode=auto` — async test functions need no `@pytest.mark.asyncio`.
- Migrations must be **idempotent** (`migrate.sh` re-applies every `[0-9]*.sql` each run, no applied-tracking). Next free migration number is **023** (019/020/021 = guidelines, 022 = export).
- PDF path for a file: `PDFS_DIR / (file_id + PDF_EXTENSION)` where `PDFS_DIR`, `PDF_EXTENSION`, `MANIFEST_PATH` live in `src/RAG/ingestion/data_loader.py`.
- Two storage backends coexist (asyncpg for pgvector reads, psycopg3 for `storage/`); this feature uses psycopg3 `storage/` classes only.
- Comment style: terse, English, only where non-obvious.
- **Infra note per task:** tests are tagged `[dev-runnable]` (pure/mocked — run in this env after `pip install -r requirements.txt`, or at least psycopg+langchain-core) or `[stand-only]` (need live Postgres and/or `fitz`/tabula, which are absent on the dev machine — write test-first, verify on the stand). Do not treat a `[stand-only]` collection error (`ModuleNotFoundError: fitz` / `KeyError: POSTGRES_HOST`) as a code failure.

---

### Task 1: Extract shared ingest pipeline

Move the per-chunk processing out of the `ingest-pdfs.py` script into an importable module so `reingest-pdfs.py` reuses it. Behavior of `ingest-pdfs.py` must not change.

**Files:**
- Create: `src/RAG/ingestion/pipeline.py`
- Modify: `scripts/ingest-pdfs.py` (remove `_chunk_text`/`_process_chunk`/`_process_batch`, import from pipeline)
- Test: `tests/test_ingest_pipeline.py`

**Interfaces:**
- Produces:
  - `pipeline.chunk_text(chunk: dict) -> str`
  - `pipeline.process_chunk(chunk: dict, file_id: str) -> Doc | None`
  - `pipeline.process_batch(chunks: list[dict], file_id: str) -> list[Doc | None]`
  - Module-level names `generate_queries`, `embed_queries` (imported into pipeline's namespace, monkeypatchable).

- [ ] **Step 1: Write the failing tests** `[dev-runnable]`

```python
# tests/test_ingest_pipeline.py
from RAG.ingestion import pipeline


class _Q:
    fact_query = "f"; procedural_query = "p"; constraint_query = "c"


class _E:
    fact_embedding = [0.1]; procedural_embedding = [0.2]; constraint_embedding = [0.3]


def test_chunk_text_passthrough_str():
    assert pipeline.chunk_text({"content": "hello"}) == "hello"


def test_chunk_text_serializes_list_as_json():
    assert pipeline.chunk_text({"content": [{"a": "1"}]}) == '[{"a": "1"}]'


async def test_process_chunk_builds_doc(monkeypatch):
    async def fake_gen(chunk):
        return (None, _Q())

    async def fake_embed(queries):
        return _E()

    monkeypatch.setattr(pipeline, "generate_queries", fake_gen)
    monkeypatch.setattr(pipeline, "embed_queries", fake_embed)

    chunk = {"content": "txt", "metadata": {"section": "1.1", "content_type": "text", "chunk_index": 0}}
    doc = await pipeline.process_chunk(chunk, "F1")

    assert doc.file_id == "F1"
    assert doc.chunk == "txt"
    assert (doc.fact_q, doc.procedure_q, doc.constraint_q) == ("f", "p", "c")
    assert doc.fact_q_embedding == [0.1]
    assert doc.metadata == {"section": "1.1", "content_type": "text", "chunk_index": 0}


async def test_process_chunk_returns_none_on_llm_error(monkeypatch):
    async def boom(chunk):
        raise RuntimeError("llm down")

    monkeypatch.setattr(pipeline, "generate_queries", boom)
    doc = await pipeline.process_chunk({"content": "t", "metadata": {"page": 3}}, "F1")
    assert doc is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingest_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'RAG.ingestion.pipeline'`.

- [ ] **Step 3: Create `src/RAG/ingestion/pipeline.py`**

```python
"""pipeline.py — shared per-chunk ingest pipeline: chunk → LLM queries → embeddings → Doc.

Extracted from scripts/ingest-pdfs.py so ingest-pdfs.py and reingest-pdfs.py share it.
"""
import asyncio
import json
import logging

from LLM.embed_queries import embed_queries
from LLM.query_generator import generate_queries
from storage.models import Doc

log = logging.getLogger(__name__)


def chunk_text(chunk: dict) -> str:
    content = chunk["content"]
    if isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    return content


async def process_chunk(chunk: dict, file_id: str) -> Doc | None:
    """Generate queries + embeddings for one chunk; return a ready-to-insert Doc (None on LLM error)."""
    text = chunk_text(chunk)
    try:
        _, queries = await generate_queries(chunk)
        embeddings = await embed_queries(queries)
    except Exception as exc:
        log.error(
            "Query/embedding generation failed for %s page %s: %s",
            file_id, chunk["metadata"].get("page"), exc,
        )
        return None

    return Doc(
        file_id=file_id,
        chunk=text,
        metadata=chunk["metadata"],
        fact_q=queries.fact_query,
        procedure_q=queries.procedural_query,
        constraint_q=queries.constraint_query,
        fact_q_embedding=embeddings.fact_embedding,
        procedure_q_embedding=embeddings.procedural_embedding,
        constraint_q_embedding=embeddings.constraint_embedding,
    )


async def process_batch(chunks: list[dict], file_id: str) -> list[Doc | None]:
    """Process a batch of chunks concurrently."""
    return list(await asyncio.gather(*[process_chunk(c, file_id) for c in chunks]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingest_pipeline.py -v`
Expected: PASS (4 passed). If import of `LLM.query_generator` errors on a missing light dep (e.g. `langchain_core`, `openai`), `pip install` that dep — it is not `fitz`.

- [ ] **Step 5: Rewire `scripts/ingest-pdfs.py` to import from pipeline**

Delete the local `_chunk_text` (lines ~55-59), `_process_chunk` (~62-87), `_process_batch` (~90-92). Add to the import block (near `from RAG.ingestion.data_loader import load_documents`):

```python
from RAG.ingestion.pipeline import chunk_text, process_batch
```

Then replace usages in `main()`:
- `_chunk_text(chunk)` → `chunk_text(chunk)` (the `content_preview` line ~127).
- `docs = await _process_batch(batch, current_file_id)` → `docs = await process_batch(batch, current_file_id)` (~141).

Leave everything else (logging, batching loop, KeyboardInterrupt handling) unchanged.

- [ ] **Step 6: Verify `ingest-pdfs.py` still imports cleanly**

Run: `python -c "import ast; ast.parse(open('scripts/ingest-pdfs.py').read()); print('parse ok')"`
Expected: `parse ok`. (Full import pulls `fitz` via data_loader — `[stand-only]`; a byte-compile/parse check is enough here.)
Also run: `pytest tests/test_ingest_pipeline.py -q` → still PASS.

- [ ] **Step 7: Commit**

```bash
git add src/RAG/ingestion/pipeline.py scripts/ingest-pdfs.py tests/test_ingest_pipeline.py
git commit -m "refactor: extract shared ingest pipeline into RAG/ingestion/pipeline.py"
```

---

### Task 2: Reingest work-list classifier (pure core)

The decision logic and PDF hashing, with zero DB/`fitz` imports so it is fully unit-testable.

**Files:**
- Create: `src/RAG/ingestion/reingest_planner.py`
- Test: `tests/test_reingest_planner.py`

**Interfaces:**
- Produces:
  - `reingest_planner.sha256_file(path: pathlib.Path) -> str`
  - `reingest_planner.Decision = Literal["full", "metadata_only", "skip"]`
  - `reingest_planner.classify(*, status: str | None, stored_hash: str | None, current_hash: str, stored_guideline: Guideline | None, new_guideline: Guideline) -> Decision`
  - `reingest_planner.build_worklist(manifest_rows: dict[str, dict], runs: dict[str, tuple[str, str | None]], guidelines_by_id: dict[str, Guideline], hash_of) -> list[tuple[str, Decision]]` — pure; `hash_of` is a callable `file_id -> str | None` (None = PDF missing → file skipped).

- [ ] **Step 1: Write the failing tests** `[dev-runnable]`

```python
# tests/test_reingest_planner.py
from pathlib import Path

from RAG.ingestion.reingest_planner import classify, sha256_file
from storage.models.guideline import Guideline


def _g(file_id="F1", name="A", mkb=None):
    return Guideline(file_id=file_id, name=name, mkb=mkb or ["I10"])


# --- classify: full-reingest triggers ---
def test_no_row_is_full():
    assert classify(status=None, stored_hash=None, current_hash="h1",
                    stored_guideline=None, new_guideline=_g()) == "full"


def test_pending_is_full():
    assert classify(status="pending", stored_hash="h1", current_hash="h1",
                    stored_guideline=_g(), new_guideline=_g()) == "full"


def test_failed_is_full():
    assert classify(status="failed", stored_hash="h1", current_hash="h1",
                    stored_guideline=_g(), new_guideline=_g()) == "full"


def test_hash_changed_is_full_even_if_metadata_same():
    assert classify(status="done", stored_hash="old", current_hash="new",
                    stored_guideline=_g(), new_guideline=_g()) == "full"


def test_rollback_hash_differs_from_last_done_is_full():
    # PDF rolled back to an older version → current hash != last-done hash → full
    assert classify(status="done", stored_hash="hB", current_hash="hA",
                    stored_guideline=_g(), new_guideline=_g()) == "full"


# --- classify: metadata-only ---
def test_done_same_hash_metadata_diff_is_metadata_only():
    assert classify(status="done", stored_hash="h1", current_hash="h1",
                    stored_guideline=_g(name="Old"), new_guideline=_g(name="New")) == "metadata_only"


def test_done_same_hash_mkb_diff_is_metadata_only():
    assert classify(status="done", stored_hash="h1", current_hash="h1",
                    stored_guideline=_g(mkb=["I10"]), new_guideline=_g(mkb=["I11"])) == "metadata_only"


# --- classify: skip ---
def test_done_same_hash_same_metadata_is_skip():
    assert classify(status="done", stored_hash="h1", current_hash="h1",
                    stored_guideline=_g(), new_guideline=_g()) == "skip"


# --- sha256_file ---
def test_sha256_file_deterministic_and_sensitive(tmp_path: Path):
    p = tmp_path / "a.pdf"
    p.write_bytes(b"hello")
    first = sha256_file(p)
    assert first == sha256_file(p)  # deterministic
    p.write_bytes(b"hello!")
    assert sha256_file(p) != first  # sensitive to content


# --- build_worklist ---
# Stored guidelines are built via from_manifest_row so equality with the "new"
# guideline holds by construction — the skip/metadata_only split is unambiguous.
def test_build_worklist_mixed():
    from RAG.ingestion.reingest_planner import build_worklist

    rows = {
        "A": {"ID": "A", "Наименование": "A"},        # done, hash same, meta same -> skip
        "B": {"ID": "B", "Наименование": "B-new"},    # done, hash same, meta diff -> metadata_only
        "C": {"ID": "C", "Наименование": "C"},        # failed                     -> full
        "D": {"ID": "D", "Наименование": "D"},        # no ingest_runs row          -> full
    }
    runs = {"A": ("done", "hA"), "B": ("done", "hB"), "C": ("failed", "hC")}
    guidelines_by_id = {
        "A": Guideline.from_manifest_row(rows["A"]),                              # == new -> skip
        "B": Guideline.from_manifest_row({"ID": "B", "Наименование": "B-old"}),   # != new -> metadata_only
        "C": Guideline.from_manifest_row(rows["C"]),
    }
    hash_of = {"A": "hA", "B": "hB", "C": "hC", "D": "hD"}.get  # dict.get -> None if missing

    wl = dict(build_worklist(rows, runs, guidelines_by_id, hash_of))
    assert wl == {"A": "skip", "B": "metadata_only", "C": "full", "D": "full"}


def test_build_worklist_skips_missing_pdf():
    from RAG.ingestion.reingest_planner import build_worklist

    rows = {"A": {"ID": "A", "Наименование": "A"}}
    wl = build_worklist(rows, {}, {}, lambda fid: None)  # PDF missing on disk
    assert wl == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reingest_planner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'RAG.ingestion.reingest_planner'`.

- [ ] **Step 3: Create `src/RAG/ingestion/reingest_planner.py`**

```python
"""reingest_planner.py — pure work-list classification + PDF hashing for reingest.

No DB and no fitz imports — unit-testable in isolation. See
docs/superpowers/specs/2026-07-09-reingest-with-resume-design.md (work-list).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from storage.models.guideline import Guideline

Decision = Literal["full", "metadata_only", "skip"]


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file's bytes (hex)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def classify(
    *,
    status: str | None,
    stored_hash: str | None,
    current_hash: str,
    stored_guideline: Guideline | None,
    new_guideline: Guideline,
) -> Decision:
    """Decide reingest action for one manifest file.

    - full          — no ingest_runs row, or status != 'done', or the PDF hash
                      differs from the last successful ('done') hash.
    - metadata_only — file is done and PDF unchanged, but the manifest row
                      differs from the stored guideline (compared as normalized
                      Guideline dataclasses).
    - skip          — done, hash matches, metadata matches.
    """
    if status != "done" or stored_hash != current_hash:
        return "full"
    if new_guideline != stored_guideline:
        return "metadata_only"
    return "skip"


def build_worklist(manifest_rows, runs, guidelines_by_id, hash_of):
    """Pure work-list: -> list[(file_id, decision)] over manifest files present on disk.

    manifest_rows:     {file_id: raw csv row dict}
    runs:              {file_id: (status, content_hash)}   (ingest_runs snapshot)
    guidelines_by_id:  {file_id: Guideline}                (stored 'old' manifest snapshot)
    hash_of:           callable file_id -> current sha256 hex, or None if PDF missing
    """
    out: list[tuple[str, Decision]] = []
    for file_id, row in manifest_rows.items():
        current_hash = hash_of(file_id)
        if current_hash is None:
            continue  # PDF missing on disk; loader logs it too
        status, stored_hash = runs.get(file_id, (None, None))
        decision = classify(
            status=status,
            stored_hash=stored_hash,
            current_hash=current_hash,
            stored_guideline=guidelines_by_id.get(file_id),
            new_guideline=Guideline.from_manifest_row(row),
        )
        out.append((file_id, decision))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reingest_planner.py -v`
Expected: PASS (11 passed). If `from storage.models.guideline import Guideline` errors on missing `psycopg`, `pip install psycopg` (import chain via `storage/__init__.py`); no live DB needed.

- [ ] **Step 5: Commit**

```bash
git add src/RAG/ingestion/reingest_planner.py tests/test_reingest_planner.py
git commit -m "feat: reingest work-list classifier + PDF hashing (pure)"
```

---

### Task 3: `load_documents(only=)` selection

Let the loader yield only a chosen set of `file_id`s, so reingest can process one file at a time.

**Files:**
- Modify: `src/RAG/ingestion/data_loader.py` (function `load_documents`, ~283-313)
- Test: `tests/test_data_loader_only.py`

**Interfaces:**
- Produces: `load_documents(manifest_path=..., pdfs_dir=..., exceptions: set[str] | None = None, only: set[str] | None = None)` — when `only` is given, yield only rows whose `ID` is in `only`; `exceptions` still applies.

- [ ] **Step 1: Write the failing test** `[stand-only]` (imports `data_loader` → `fitz`)

```python
# tests/test_data_loader_only.py
import csv
from pathlib import Path

from RAG.ingestion.data_loader import load_documents


def _make_manifest(dir_: Path, ids: list[str]) -> Path:
    mpath = dir_ / "manifest.csv"
    with open(mpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ID", "Наименование"])
        w.writeheader()
        for i in ids:
            w.writerow({"ID": i, "Наименование": f"name-{i}"})
    return mpath


def test_only_yields_selected_ids(tmp_path: Path):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    for i in ["A", "B", "C"]:
        (pdfs / f"{i}.pdf").write_bytes(b"%PDF-1.4")
    manifest = _make_manifest(tmp_path, ["A", "B", "C"])

    got = [r.metadata["ID"] for r in load_documents(manifest_path=manifest, pdfs_dir=pdfs, only={"B"})]
    assert got == ["B"]


def test_only_and_exceptions_combine(tmp_path: Path):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    for i in ["A", "B"]:
        (pdfs / f"{i}.pdf").write_bytes(b"%PDF-1.4")
    manifest = _make_manifest(tmp_path, ["A", "B"])

    got = [r.metadata["ID"] for r in
           load_documents(manifest_path=manifest, pdfs_dir=pdfs, only={"A", "B"}, exceptions={"A"})]
    assert got == ["B"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_loader_only.py -v`
Expected on stand: FAIL — `TypeError: load_documents() got an unexpected keyword argument 'only'`.
Expected on dev machine: collection ERROR `ModuleNotFoundError: No module named 'fitz'` — acceptable; this task is verified on the stand.

- [ ] **Step 3: Add the `only` parameter**

In `src/RAG/ingestion/data_loader.py`, change the signature and the loop:

```python
def load_documents(
    manifest_path: Path = MANIFEST_PATH,
    pdfs_dir: Path = PDFS_DIR,
    exceptions: set[str] | None = None,
    only: set[str] | None = None,
) -> Generator[PDFContentReader, None, None]:
```

Extend the docstring `Args:` with:

```
        only:          Optional set of ID strings to yield exclusively
                       (everything else is skipped). Applied together with
                       `exceptions` (exceptions win).
```

Inside the `for row in csv.DictReader(fh):` loop, right after `file_id = row["ID"]`, add the `only` gate before the existing `exceptions` gate:

```python
            file_id = row["ID"]
            if only is not None and file_id not in only:
                continue
            if exceptions is not None and file_id in exceptions:
                continue
```

- [ ] **Step 4: Run test to verify it passes**

Run (on the stand): `pytest tests/test_data_loader_only.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/RAG/ingestion/data_loader.py tests/test_data_loader_only.py
git commit -m "feat: load_documents(only=) to select specific file_ids"
```

---

### Task 4: `DocsStorage.replace_by_file_id` (atomic delete+insert)

Replace all of a file's `docs` rows in a single transaction. Refactor the shared INSERT so `insert`, `insert_many`, and `replace_by_file_id` reuse one SQL statement.

**Files:**
- Modify: `src/storage/docs_storage.py` (writes section ~61-130)
- Test: `tests/test_docs_replace.py`

**Interfaces:**
- Consumes: `Doc` (existing model).
- Produces: `DocsStorage.replace_by_file_id(file_id: str, docs: list[Doc]) -> list[str]` — deletes existing rows for `file_id`, bulk-inserts `docs`, all in one transaction; returns new UUIDs and sets each `doc.id`.

- [ ] **Step 1: Write the failing test** `[stand-only]` (needs live Postgres)

```python
# tests/test_docs_replace.py
from storage import DocsStorage
from storage.models import Doc


import os

_DIM = int(os.environ["EMBEDDING_DIM"])  # docs.*_embedding columns are vector(EMBEDDING_DIM)


def _doc(file_id: str, chunk: str) -> Doc:
    return Doc(
        file_id=file_id, chunk=chunk, metadata={"section": "1.1", "content_type": "text", "chunk_index": 0},
        fact_q="f", procedure_q="p", constraint_q="c",
        fact_q_embedding=[0.0] * _DIM, procedure_q_embedding=[0.0] * _DIM, constraint_q_embedding=[0.0] * _DIM,
    )


async def test_replace_by_file_id_swaps_rows():
    async with DocsStorage() as s:
        await s.replace_by_file_id("RP1", [_doc("RP1", "old-a"), _doc("RP1", "old-b")])
        new_ids = await s.replace_by_file_id("RP1", [_doc("RP1", "new-only")])
        assert len(new_ids) == 1
        rows = await s.get_many(new_ids)
        assert [r.chunk for r in rows] == ["new-only"]
```

- [ ] **Step 2: Run test to verify it fails**

Run (stand): `pytest tests/test_docs_replace.py -v`
Expected: FAIL — `AttributeError: 'DocsStorage' object has no attribute 'replace_by_file_id'`.

- [ ] **Step 3: Extract shared INSERT and add `replace_by_file_id`**

In `src/storage/docs_storage.py`, add module-level helpers (after the imports / `_row_to_doc`):

```python
_INSERT_DOC_SQL = """
    INSERT INTO docs (
        file_id, chunk, metadata,
        fact_q, procedure_q, constraint_q,
        fact_q_embedding, procedure_q_embedding, constraint_q_embedding
    ) VALUES (
        %(file_id)s, %(chunk)s, %(metadata)s,
        %(fact_q)s, %(procedure_q)s, %(constraint_q)s,
        %(fact_q_embedding)s, %(procedure_q_embedding)s, %(constraint_q_embedding)s
    )
    RETURNING id::text
"""


def _doc_params(doc: "Doc") -> dict:
    return {
        "file_id": doc.file_id,
        "chunk": doc.chunk,
        "metadata": json.dumps(doc.metadata),
        "fact_q": doc.fact_q,
        "procedure_q": doc.procedure_q,
        "constraint_q": doc.constraint_q,
        "fact_q_embedding": doc.fact_q_embedding,
        "procedure_q_embedding": doc.procedure_q_embedding,
        "constraint_q_embedding": doc.constraint_q_embedding,
    }
```

Rewrite `insert` and `insert_many` to use them, and add `replace_by_file_id`:

```python
    async def insert(self, doc: Doc) -> str:
        """Insert a single Doc and return its UUID. Also sets doc.id."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(_INSERT_DOC_SQL, _doc_params(doc))
            row = await cur.fetchone()
        doc.id = row["id"]
        return row["id"]

    async def insert_many(self, docs: list[Doc]) -> list[str]:
        """Bulk-insert multiple Docs in one transaction; returns UUIDs, sets each doc.id."""
        ids: list[str] = []
        async with self._pool.connection() as conn:
            for doc in docs:
                cur = await conn.execute(_INSERT_DOC_SQL, _doc_params(doc))
                result = await cur.fetchone()
                doc.id = result["id"]
                ids.append(result["id"])
        return ids

    async def replace_by_file_id(self, file_id: str, docs: list[Doc]) -> list[str]:
        """Atomically delete all rows for file_id and bulk-insert `docs` (one transaction).

        Returns new UUIDs and sets each doc.id. `docs` may be empty (pure delete).
        """
        ids: list[str] = []
        async with self._pool.connection() as conn:
            await conn.execute("DELETE FROM docs WHERE file_id = %(file_id)s", {"file_id": file_id})
            for doc in docs:
                cur = await conn.execute(_INSERT_DOC_SQL, _doc_params(doc))
                result = await cur.fetchone()
                doc.id = result["id"]
                ids.append(result["id"])
        return ids
```

(`json` is already imported in this module.)

- [ ] **Step 4: Run test to verify it passes**

Run (stand): `pytest tests/test_docs_replace.py -v`
Expected: PASS. Also run existing docs tests to confirm the refactor didn't regress: `pytest tests/test_doc_format_chunk.py -q`.

- [ ] **Step 5: Commit**

```bash
git add src/storage/docs_storage.py tests/test_docs_replace.py
git commit -m "feat: DocsStorage.replace_by_file_id (atomic delete+insert); DRY insert SQL"
```

---

### Task 5: `ingest_runs` table + `IngestRunsStorage`

The resume-state table and its storage class. `content_hash` is written only on `mark_done`; `upsert_pending`/`mark_failed` preserve it.

**Files:**
- Create: `migrations/023_ingest_runs.sql`
- Create: `src/storage/ingest_runs_storage.py`
- Modify: `src/storage/__init__.py` (export)
- Test: `tests/test_ingest_runs_storage.py`

**Interfaces:**
- Produces:
  - `IngestRunsStorage.get_all() -> dict[str, tuple[str, str | None]]` — `file_id -> (status, content_hash)`
  - `IngestRunsStorage.upsert_pending(file_id: str) -> None`
  - `IngestRunsStorage.mark_done(file_id: str, content_hash: str) -> None`
  - `IngestRunsStorage.mark_failed(file_id: str, error: str) -> None`
  - `from storage import IngestRunsStorage`

- [ ] **Step 1: Create the migration**

`migrations/023_ingest_runs.sql`:

```sql
-- 023_ingest_runs.sql — resume-state for scripts/reingest-pdfs.py.
-- content_hash = sha256 of the PDF at the last successful ('done') reingest.
CREATE TABLE IF NOT EXISTS ingest_runs (
    file_id      TEXT PRIMARY KEY,
    status       TEXT NOT NULL DEFAULT 'pending',   -- 'pending' | 'done' | 'failed'
    content_hash TEXT,
    error        TEXT,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

- [ ] **Step 2: Write the failing tests** `[stand-only]` (needs live Postgres + migration 023 applied)

```python
# tests/test_ingest_runs_storage.py
from storage import IngestRunsStorage


async def test_pending_then_done_records_hash():
    async with IngestRunsStorage() as s:
        await s.upsert_pending("IR1")
        assert (await s.get_all())["IR1"] == ("pending", None)
        await s.mark_done("IR1", "hashA")
        assert (await s.get_all())["IR1"] == ("done", "hashA")


async def test_failed_preserves_last_done_hash():
    async with IngestRunsStorage() as s:
        await s.mark_done("IR2", "hashB")
        await s.upsert_pending("IR2")          # re-run starts
        assert (await s.get_all())["IR2"] == ("pending", "hashB")   # hash preserved
        await s.mark_failed("IR2", "boom")
        status, h = (await s.get_all())["IR2"]
        assert status == "failed" and h == "hashB"                  # hash still preserved
```

- [ ] **Step 3: Run tests to verify they fail**

Run (stand, after `bash migrations/migrate.sh`): `pytest tests/test_ingest_runs_storage.py -v`
Expected: FAIL — `ImportError: cannot import name 'IngestRunsStorage' from 'storage'`.

- [ ] **Step 4: Create `src/storage/ingest_runs_storage.py`**

```python
"""ingest_runs_storage.py — resume-state for reingest (table: 023_ingest_runs.sql).

Invariant: content_hash is written ONLY by mark_done; upsert_pending and
mark_failed preserve the existing hash, so it always reflects the last
successful ('done') reingest of the file.
"""
from __future__ import annotations

from .base import BaseStorage


class IngestRunsStorage(BaseStorage):
    async def get_all(self) -> dict[str, tuple[str, str | None]]:
        """file_id -> (status, content_hash) for every recorded file."""
        async with self._pool.connection() as conn:
            cur = await conn.execute("SELECT file_id, status, content_hash FROM ingest_runs")
            rows = await cur.fetchall()
        return {r["file_id"]: (r["status"], r["content_hash"]) for r in rows}

    async def upsert_pending(self, file_id: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO ingest_runs (file_id, status)
                VALUES (%(file_id)s, 'pending')
                ON CONFLICT (file_id) DO UPDATE SET
                    status = 'pending', updated_at = now()
                """,
                {"file_id": file_id},
            )

    async def mark_done(self, file_id: str, content_hash: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO ingest_runs (file_id, status, content_hash, error)
                VALUES (%(file_id)s, 'done', %(h)s, NULL)
                ON CONFLICT (file_id) DO UPDATE SET
                    status = 'done', content_hash = %(h)s, error = NULL, updated_at = now()
                """,
                {"file_id": file_id, "h": content_hash},
            )

    async def mark_failed(self, file_id: str, error: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO ingest_runs (file_id, status, error)
                VALUES (%(file_id)s, 'failed', %(e)s)
                ON CONFLICT (file_id) DO UPDATE SET
                    status = 'failed', error = %(e)s, updated_at = now()
                """,
                {"file_id": file_id, "e": error},
            )
```

- [ ] **Step 5: Export from `src/storage/__init__.py`**

```python
from .docs_storage import DocsStorage
from .done_cards_storage import DoneCardsStorage
from .ingest_runs_storage import IngestRunsStorage
from .organizations_storage import OrganizationsStorage
from .models import Doc, Result

__all__ = ["DocsStorage", "DoneCardsStorage", "IngestRunsStorage", "OrganizationsStorage", "Doc", "Result"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run (stand): `pytest tests/test_ingest_runs_storage.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add migrations/023_ingest_runs.sql src/storage/ingest_runs_storage.py src/storage/__init__.py tests/test_ingest_runs_storage.py
git commit -m "feat: ingest_runs table + IngestRunsStorage (resume state)"
```

---

### Task 6: `scripts/reingest-pdfs.py` orchestration

Tie it together: build the work-list, run per-file with per-file try/except and resume, support `--only-failed` and `--file-id`.

**Files:**
- Create: `scripts/reingest-pdfs.py`

(The work-list logic is already covered by `reingest_planner.build_worklist` tests in Task 2. This task adds only the orchestration script; its verification is the stand smoke in Step 2.)

**Interfaces:**
- Consumes: `reingest_planner.build_worklist`/`sha256_file`, `pipeline.process_batch`, `data_loader.load_documents(only=)` + constants `MANIFEST_PATH`/`PDFS_DIR`/`PDF_EXTENSION`, `DocsStorage.replace_by_file_id`, `GuidelinesStorage.upsert_many`/`.all`, `IngestRunsStorage.get_all`/`upsert_pending`/`mark_done`/`mark_failed`, `Guideline.from_manifest_row`.
- Produces: the CLI `python scripts/reingest-pdfs.py [--only-failed] [--file-id ID]`.

- [ ] **Step 1: Create `scripts/reingest-pdfs.py`**

```python
#!/usr/bin/env python3
"""reingest-pdfs.py — re-sync docs + guidelines to the current manifest & PDFs, resumably.

Unlike ingest-pdfs.py, does NOT skip already-ingested files. Per file it decides
(via reingest_planner.classify) between a full re-chunk, a cheap metadata-only
guidelines upsert, or skip — driven by the ingest_runs resume table (status +
last-done PDF hash) and a manifest-vs-guidelines diff.

    python scripts/reingest-pdfs.py [--only-failed] [--file-id ID]
"""
import argparse
import asyncio
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from RAG.ingestion.data_loader import MANIFEST_PATH, PDFS_DIR, PDF_EXTENSION, load_documents
from RAG.ingestion.pipeline import process_batch
from RAG.ingestion.reingest_planner import build_worklist, sha256_file
from storage import DocsStorage, IngestRunsStorage
from storage.guidelines_storage import GuidelinesStorage
from storage.models.guideline import Guideline

QUERY_GENERATION_BATCH_SIZE = 3

LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
log_filename = LOGS_DIR / f"reingest-pdfs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_filename, encoding="utf-8")],
)
log = logging.getLogger(__name__)


def _pdf_path(file_id: str) -> Path:
    return PDFS_DIR / (file_id + PDF_EXTENSION)


def _read_manifest_rows() -> dict:
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as fh:
        return {(r.get("ID") or "").strip(): r
                for r in csv.DictReader(fh) if (r.get("ID") or "").strip()}


async def _full_reingest(file_id, row, docs_storage, guidelines_storage, runs_storage):
    await runs_storage.upsert_pending(file_id)
    try:
        readers = list(load_documents(only={file_id}))
        if not readers:
            raise FileNotFoundError(f"no reader for {file_id} (missing PDF?)")
        chunks = list(readers[0].iter_chunks())

        docs = []
        for start in range(0, len(chunks), QUERY_GENERATION_BATCH_SIZE):
            batch = chunks[start:start + QUERY_GENERATION_BATCH_SIZE]
            docs.extend(d for d in await process_batch(batch, file_id) if d is not None)

        await docs_storage.replace_by_file_id(file_id, docs)
        await guidelines_storage.upsert_many([Guideline.from_manifest_row(row)])
        await runs_storage.mark_done(file_id, sha256_file(_pdf_path(file_id)))
        log.info("Reingested %s — %d chunk(s)", file_id, len(docs))
    except Exception as exc:  # per-file: never halt the whole run
        await runs_storage.mark_failed(file_id, str(exc))
        log.error("FAILED %s: %s", file_id, exc)


async def _metadata_only(file_id, row, guidelines_storage):
    await guidelines_storage.upsert_many([Guideline.from_manifest_row(row)])
    log.info("Metadata-only update for %s (PDF unchanged)", file_id)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Reingest PDFs / sync guidelines with resume.")
    parser.add_argument("--only-failed", action="store_true", help="only files with status='failed'")
    parser.add_argument("--file-id", help="force full reingest of one file_id (bypass diff logic)")
    args = parser.parse_args()

    manifest_rows = _read_manifest_rows()

    async with DocsStorage() as docs_storage, \
            GuidelinesStorage() as guidelines_storage, \
            IngestRunsStorage() as runs_storage:

        runs = await runs_storage.get_all()
        guidelines_by_id = {g.file_id: g for g in await guidelines_storage.all()}

        if args.file_id:
            worklist = [(args.file_id, "full")]
        else:
            worklist = build_worklist(manifest_rows, runs, guidelines_by_id, _current_hash)
            if args.only_failed:
                worklist = [(fid, "full") for fid, _ in worklist
                            if runs.get(fid, (None, None))[0] == "failed"]

        log.info("Work-list: %d file(s)", len(worklist))
        for file_id, decision in worklist:
            row = manifest_rows.get(file_id)
            if decision == "skip" or row is None:
                continue
            if decision == "metadata_only":
                await _metadata_only(file_id, row, guidelines_storage)
            else:
                await _full_reingest(file_id, row, docs_storage, guidelines_storage, runs_storage)

    log.info("Reingest complete. Log: %s", log_filename)


def _current_hash(file_id: str):
    p = _pdf_path(file_id)
    return sha256_file(p) if p.exists() else None


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: End-to-end smoke on the stand** `[stand-only]`

Run (stand, after `bash migrations/migrate.sh` and `python scripts/seed-guidelines.py`):
```bash
python scripts/reingest-pdfs.py --file-id <one-known-id>   # forces full reingest of one file
python scripts/reingest-pdfs.py                            # second run → that file now 'skip'
```
Expected: first run logs `Reingested <id> — N chunk(s)`; `SELECT status,count(*) FROM ingest_runs GROUP BY status` shows it `done`; second run logs it under work-list but skips (no re-chunk). Spot-check `SELECT count(*) FROM docs WHERE file_id='<id>'` is stable across the second run.

- [ ] **Step 3: Commit**

```bash
git add scripts/reingest-pdfs.py
git commit -m "feat: scripts/reingest-pdfs.py — resumable manifest/PDF re-sync"
```

---

## Notes for the executor

- **Migration/seed order on the stand:** `bash migrations/migrate.sh` (applies 019-023 idempotently) → `python scripts/seed-guidelines.py` (needed so `guidelines` holds the "old manifest" snapshot the diff compares against, and so FK 021 is satisfied) → then reingest.
- **Work-list lives in `reingest_planner.py`** (pure, no `fitz`), so it is unit-tested in Task 2 without the loader; the script only orchestrates.
- **Do not** add a FK from `ingest_runs` to `docs`/`guidelines` (a file is transiently absent from `docs` between delete and insert).
