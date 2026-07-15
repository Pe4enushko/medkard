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
