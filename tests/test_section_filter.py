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
