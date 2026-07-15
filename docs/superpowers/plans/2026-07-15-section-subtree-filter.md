# Section Subtree Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the section keyword filter also match the numbered subtree of any keyword-matched section, so `search_treatment` etc. stop missing subchapters whose leaf title lacks the keyword (e.g. `3.1.2 Наружная терапия` under `3 Лечение`).

**Architecture:** Pure Python owns all section-number parsing (`_extract_section_number`, `_section_like_patterns`); SQL does zero number parsing — just `metadata->>'section' LIKE ANY($patterns)`. `_vector_search_filtered` resolves keyword-matched anchor sections via one small query, turns them into LIKE patterns in Python, and adds the predicate. The original keyword `LIKE` stays as a fallback.

**Tech Stack:** Python (asyncpg), PostgreSQL/pgvector, pytest (asyncio_mode=auto, pythonpath=src).

## Global Constraints

- Matching expands strictly **down** the subtree from a keyword-matched section; sibling sections at the same level are never pulled in.
- Subtree boundary is enforced by space/dot LIKE patterns: for section number `N`, the two patterns are `"N %"` (the section itself: number + space + title) and `"N.%"` (numbered descendants). Example: for `3.1`, `"3.10 …"` matches neither pattern.
- **Python is the sole number parser.** SQL contains no `regexp_match`/number logic — only `LIKE ANY`. Do not reintroduce number parsing in SQL.
- The keyword `LIKE '%<keyword>%'` clause is retained; when there are no numbered anchors the pattern array is empty and the predicate degrades to today's keyword-only behavior.
- `_vector_search_filtered`'s public signature is unchanged (`section_filter` stays a keyword substring). `searches.py` and the three `search_*` functions are NOT modified.
- No data changes: no migration, no backfill, no re-ingest. The section number already lives in `metadata.section`.
- No reachable medkard Postgres on this dev machine. The pure helpers and the query-wiring (via a fake pool) are unit-tested here; real end-to-end behavior against pgvector is a stand-only test authored in `tests/test_vector_store.py` following its existing DB-integration pattern (it will not run on this machine — that is expected, not a failure).

---

### Task 1: Pure section-number helpers

Add the two pure functions that parse a section number and build LIKE patterns. Fully unit-testable on the dev machine.

**Files:**
- Modify: `src/RAG/retrieval/vector_store.py` (add `import re`; add two functions near the other module-level helpers, e.g. just above `_vector_search_filtered`)
- Test: `tests/test_section_filter.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `_extract_section_number(section: str | None) -> str | None`
  - `_section_like_patterns(anchor_sections: list[str]) -> list[str]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_section_filter.py`:

```python
"""Unit tests for pure section-number helpers in vector_store (no DB)."""
import fnmatch

from RAG.retrieval.vector_store import (
    _extract_section_number,
    _section_like_patterns,
)


def _sql_like(text: str, pattern: str) -> bool:
    """Emulate SQL LIKE for our patterns (only '%' wildcard; no '_' or '[' used)."""
    return fnmatch.fnmatchcase(text, pattern.replace("%", "*"))


def test_extract_number_three_levels():
    assert _extract_section_number("3.1.2 Наружная терапия") == "3.1.2"


def test_extract_number_top_level():
    assert _extract_section_number("3 Лечение") == "3"


def test_extract_number_non_numbered_is_none():
    assert _extract_section_number("Приложение А") is None


def test_extract_number_empty_and_none():
    assert _extract_section_number("") is None
    assert _extract_section_number(None) is None


def test_patterns_top_level():
    assert _section_like_patterns(["3 Лечение"]) == ["3 %", "3.%"]


def test_patterns_subsection():
    assert _section_like_patterns(["2.1 Жалобы и анамнез"]) == ["2.1 %", "2.1.%"]


def test_patterns_dedup_by_number():
    # same number, different titles -> one pair
    assert _section_like_patterns(["3 Лечение", "3 Лечение (доп)"]) == ["3 %", "3.%"]


def test_patterns_skip_non_numbered():
    assert _section_like_patterns(["Приложение А", "2 Диагностика"]) == ["2 %", "2.%"]


def test_patterns_empty_input():
    assert _section_like_patterns([]) == []


def test_boundary_3_1_excludes_3_10_includes_children():
    pats = _section_like_patterns(["3.1 Медикаментозное лечение"])
    assert not any(_sql_like("3.10 Иное", p) for p in pats)      # sibling number, not a child
    assert any(_sql_like("3.1 Медикаментозное лечение", p) for p in pats)
    assert any(_sql_like("3.1.2 Наружная терапия", p) for p in pats)


def test_boundary_chapter_3_includes_3_10():
    pats = _section_like_patterns(["3 Лечение"])
    assert any(_sql_like("3.10 Иное", p) for p in pats)          # 3.10 is part of chapter 3
    assert not any(_sql_like("4 Реабилитация", p) for p in pats)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/savoy/projects/medkard-section-filter && python3 -m pytest tests/test_section_filter.py -v`
Expected: FAIL at import — `_extract_section_number` / `_section_like_patterns` do not exist yet (`ImportError`).

- [ ] **Step 3: Add `import re` and the two helpers**

In `src/RAG/retrieval/vector_store.py`, add `import re` to the stdlib import group at the top (alongside `import json`, `import logging`, `import os`). Then add these two functions just above `async def _vector_search_filtered(`:

```python
_SECTION_NUM_RE = re.compile(r"^\d+(?:\.\d+)*")


def _extract_section_number(section: str | None) -> str | None:
    """Leading dotted number of a section title, or None.

    '3.1.2 Наружная терапия' -> '3.1.2'; '3 Лечение' -> '3'; 'Приложение А' -> None.
    """
    m = _SECTION_NUM_RE.match(section or "")
    return m.group(0) if m else None


def _section_like_patterns(anchor_sections: list[str]) -> list[str]:
    """SQL-LIKE patterns covering each anchor section itself and its numbered descendants.

    '3 Лечение'  -> ['3 %', '3.%']      '2.1 Жалобы' -> ['2.1 %', '2.1.%']

    '<num> %' matches the section itself (number + space + title); '<num>.%' matches
    numbered descendants. The dot is a LIKE literal, so '3.1 %'/'3.1.%' do NOT match
    '3.10 …' (a '0', not a space/dot, follows '3.1'). Non-numbered anchors are skipped;
    patterns are de-duplicated by number, order preserved.
    """
    patterns: list[str] = []
    seen: set[str] = set()
    for section in anchor_sections:
        num = _extract_section_number(section)
        if num and num not in seen:
            seen.add(num)
            patterns.append(f"{num} %")
            patterns.append(f"{num}.%")
    return patterns
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/savoy/projects/medkard-section-filter && python3 -m pytest tests/test_section_filter.py -v`
Expected: PASS (11 passed).

- [ ] **Step 5: Commit**

```bash
git add src/RAG/retrieval/vector_store.py tests/test_section_filter.py
git commit -m "feat(retrieval): pure section-number helpers (extract + LIKE patterns)"
```

---

### Task 2: Wire subtree expansion into `_vector_search_filtered`

Add the anchor-resolution query and the `LIKE ANY` predicate. Cover the wiring with a fake-pool unit test (dev-runnable) and author the real behavior test in `tests/test_vector_store.py` (stand-only).

**Files:**
- Modify: `src/RAG/retrieval/vector_store.py` (add `_section_anchor_sections`; extend `_vector_search_filtered`)
- Test: `tests/test_section_filter.py` (add a fake-pool wiring test)
- Test: `tests/test_vector_store.py` (add a stand-only behavior test)

**Interfaces:**
- Consumes: `_extract_section_number`, `_section_like_patterns` (Task 1); `_get_pool()` (existing).
- Produces: `async _section_anchor_sections(pool, file_id: str, keyword_like: str) -> list[str]`.

- [ ] **Step 1: Write the failing fake-pool wiring test**

Append to `tests/test_section_filter.py`:

```python
import pytest

from RAG.retrieval import vector_store


class _FakePool:
    """Records fetch calls; returns anchor rows for the anchor query, [] otherwise."""

    def __init__(self, anchor_sections):
        self._anchor_sections = anchor_sections
        self.calls = []  # list of (sql, args)

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        if "SELECT DISTINCT metadata->>'section'" in sql:
            return [{"section": s} for s in self._anchor_sections]
        return []


@pytest.mark.asyncio
async def test_vector_search_filtered_wires_subtree_patterns(monkeypatch):
    pool = _FakePool(["3 Лечение"])

    async def fake_get_pool():
        return pool

    monkeypatch.setattr(vector_store, "_get_pool", fake_get_pool)

    await vector_store._vector_search_filtered(
        [0.1] * vector_store.EMBEDDING_DIM, "F1", 8, section_filter="лечен"
    )

    anchor_sql, anchor_args = pool.calls[0]
    assert "SELECT DISTINCT metadata->>'section'" in anchor_sql
    assert "F1" in anchor_args
    assert "%лечен%" in anchor_args

    # main_args holds a numpy embedding, so filter by type before membership checks
    # (numpy `==` in `in` would raise "truth value ambiguous").
    main_sql, main_args = pool.calls[1]
    str_args = [a for a in main_args if isinstance(a, str)]
    list_args = [a for a in main_args if isinstance(a, list)]
    assert "LIKE ANY(" in main_sql
    assert ["3 %", "3.%"] in list_args          # patterns reached the main query
    assert "%лечен%" in str_args                # keyword fallback retained


@pytest.mark.asyncio
async def test_vector_search_filtered_no_section_skips_anchor_query(monkeypatch):
    pool = _FakePool([])

    async def fake_get_pool():
        return pool

    monkeypatch.setattr(vector_store, "_get_pool", fake_get_pool)

    await vector_store._vector_search_filtered(
        [0.1] * vector_store.EMBEDDING_DIM, "F1", 8, section_filter=None
    )

    assert len(pool.calls) == 1  # only the main query, no anchor resolution
    assert "LIKE ANY(" not in pool.calls[0][0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/savoy/projects/medkard-section-filter && python3 -m pytest tests/test_section_filter.py -v -k "wires_subtree or no_section"`
Expected: FAIL — `_vector_search_filtered` currently emits `lower(metadata->>'section') LIKE $n` (no `LIKE ANY`, no anchor query); the wiring assertions fail.

- [ ] **Step 3: Add `_section_anchor_sections` and rewire the filter**

In `src/RAG/retrieval/vector_store.py`, add this function just above `_vector_search_filtered`:

```python
async def _section_anchor_sections(pool, file_id: str, keyword_like: str) -> list[str]:
    """Distinct numbered section titles in *file_id* whose title matches the keyword.

    *keyword_like* is the already-wrapped LIKE argument, e.g. '%лечен%'.
    """
    rows = await pool.fetch(
        """
        SELECT DISTINCT metadata->>'section' AS section
        FROM docs
        WHERE file_id = $1
          AND lower(metadata->>'section') LIKE $2
          AND metadata->>'section' ~ '^[0-9]'
        """,
        file_id,
        keyword_like,
    )
    return [r["section"] for r in rows if r["section"]]
```

Then, inside `_vector_search_filtered`, replace the current section block:

```python
    if section_filter:
        params.append(f"%{section_filter}%")
        where_clauses.append(f"lower(metadata->>'section') LIKE ${len(params)}")
```

with:

```python
    if section_filter:
        keyword_like = f"%{section_filter}%"
        anchors = await _section_anchor_sections(pool, file_id, keyword_like)
        patterns = _section_like_patterns(anchors)

        params.append(keyword_like)
        kw_idx = len(params)
        params.append(patterns)
        pat_idx = len(params)
        where_clauses.append(
            f"(lower(metadata->>'section') LIKE ${kw_idx} "
            f"OR metadata->>'section' LIKE ANY(${pat_idx}::text[]))"
        )
```

(The `::text[]` cast lets asyncpg type an empty `patterns` array. Everything else in the function — `where_sql` join, the main `pool.fetch`, the `LIMIT ${len(params) + 1}` and trailing `limit` arg — is unchanged.)

- [ ] **Step 4: Run the wiring tests to verify they pass**

Run: `cd /home/savoy/projects/medkard-section-filter && python3 -m pytest tests/test_section_filter.py -v`
Expected: PASS (all — 11 helper tests + 2 wiring tests = 13).

- [ ] **Step 5: Author the stand-only behavior test**

Add to `tests/test_vector_store.py` (follows the existing `seeded_docs`/`DocsStorage` DB pattern in that file; it requires Postgres and will not run on this dev machine — that is expected):

```python
@pytest.mark.asyncio
async def test_section_filter_pulls_numbered_subtree(seeded_docs):
    """search via _vector_search_filtered('лечен') returns the whole '3 Лечение' subtree,
    including differently-named children, but not sibling chapters."""
    from RAG.retrieval.vector_store import _vector_search_filtered, EMBEDDING_DIM

    # NOTE: seed docs in this file's fixture style with sections:
    #   "3 Лечение", "3.1.2 Наружная терапия", "3.10 Иное", "2.2 Диагностика prm"
    # then:
    results = await _vector_search_filtered([0.0] * EMBEDDING_DIM, TEST_FILE_ID, 50, "лечен")
    sections = {(_metadata(r).get("section")) for r in results}

    assert "3 Лечение" in sections
    assert "3.1.2 Наружная терапия" in sections   # child, title lacks 'лечен' — the fix
    assert "3.10 Иное" in sections                # 3.10 belongs to chapter 3
    assert "2.2 Диагностика prm" not in sections  # different chapter, not pulled
```

(If `tests/test_vector_store.py` has no `TEST_FILE_ID`/`_metadata` helper, seed one guideline + these four docs in the test itself following the file's existing fixture, and read `r["metadata"]` directly. Keep the seed/teardown symmetric with the file's other tests.)

- [ ] **Step 6: Commit**

```bash
git add src/RAG/retrieval/vector_store.py tests/test_section_filter.py tests/test_vector_store.py
git commit -m "feat(retrieval): expand section filter to keyword-matched numbered subtree"
```

---

## Notes for the executor

- **Baseline:** run new tests file-scoped (`python3 -m pytest tests/test_section_filter.py -v`). The broader suite has pre-existing failures where infra is absent (no reachable Postgres / OPENAI / embedding env) — those are NOT regressions from this work.
- **Stand-only checklist (not automated here, spec §Тесты):** `search_treatment` returns `3.1.2 Наружная терапия` under `3 Лечение`; anchor `2.1` does not pull `2.2`; `3.10` is excluded under anchor `3.1` but included under `3`; a file with no numbered keyword sections yields keyword-parity with today.
- **Do not** touch `searches.py`, the three `search_*` functions, `get_section_chunks`, or add any migration/backfill — all out of scope per the spec.
