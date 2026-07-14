"""Tests for numbered-section splitting in RAG/ingestion/data_loader.py.

Covers the 3-level subsection support (e.g. "3.1.1 Title") added on top of
the original 2-level ("3.1 Title") regex, plus the guard against ToC lines
with runs of dots (e.g. "Список литературы....5").

Pure regex/string tests — no DB, no PDF parsing.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from RAG.ingestion.data_loader import _split_into_sections


class TestSplitIntoSectionsThreeLevel:
    def test_three_level_subsections_are_separate_sections(self):
        text = (
            "3.1 Консервативное лечение\n"
            "Общий текст про консервативное лечение.\n\n"
            "3.1.1 Медикаментозная терапия\n"
            "Текст про медикаментозную терапию.\n\n"
            "3.1.2 Физиотерапия\n"
            "Текст про физиотерапию.\n\n"
            "3.2 Хирургическое лечение\n"
            "Текст про хирургию.\n"
        )
        sections = _split_into_sections(text)
        titles = [t for t, _ in sections]

        assert "3.1 Консервативное лечение" in titles
        assert "3.1.1 Медикаментозная терапия" in titles
        assert "3.1.2 Физиотерапия" in titles
        assert "3.2 Хирургическое лечение" in titles

        # 3.1.1 must not swallow 3.1.2's content.
        body_311 = next(txt for t, txt in sections if t == "3.1.1 Медикаментозная терапия")
        assert "Физиотерапия" not in body_311

    def test_two_level_section_without_third_level_still_works(self):
        text = (
            "3.1 Диагностика\n"
            "Текст диагностики без подглав.\n\n"
            "3.2 Лечение\n"
            "Текст лечения.\n"
        )
        sections = _split_into_sections(text)
        titles = [t for t, _ in sections]
        assert titles == ["3.1 Диагностика", "3.2 Лечение"]

    def test_two_digit_second_level_not_confused_with_third_level(self):
        # "3.12" is a 2-level section with a two-digit second component,
        # not "3.1" + a stray ".2". Must not be misparsed as 3-level.
        text = (
            "3.12 Наблюдение\n"
            "Текст про наблюдение, раздел 3.12.\n\n"
            "3.13 Реабилитация\n"
            "Текст реабилитации.\n"
        )
        sections = _split_into_sections(text)
        titles = [t for t, _ in sections]
        assert titles == ["3.12 Наблюдение", "3.13 Реабилитация"]
        for t in titles:
            assert not t.startswith("3.1 ")

    def test_toc_lines_with_dot_leaders_are_not_treated_as_sections(self):
        text = (
            "Список литературы....................5\n"
            "3.1.1 Приложение А1..........12\n"
            "3.1 Настоящий раздел\n"
            "Реальный текст раздела 3.1, без точек.\n\n"
            "3.1.1 Настоящая подглава\n"
            "Реальный текст подглавы 3.1.1.\n"
        )
        sections = _split_into_sections(text)
        titles = [t for t, _ in sections]

        for t in titles:
            assert t is None or ".." not in t

        assert "3.1 Настоящий раздел" in titles
        assert "3.1.1 Настоящая подглава" in titles

    def test_body_dots_far_from_title_line_do_not_break_matching(self):
        # The dot-leader guard is line-scoped ([^\n]*), so a ".." that
        # appears in the section BODY (not the title line) must not
        # prevent the section from being matched.
        text = (
            "3.1.1 Подглава с точками в теле\n"
            "Первая строка ок.\n"
            "В теле текста многоточие.... — это нормально, не заголовок.\n\n"
            "3.1.2 Следующая подглава\n"
            "Текст.\n"
        )
        sections = _split_into_sections(text)
        titles = [t for t, _ in sections]
        assert "3.1.1 Подглава с точками в теле" in titles
        assert "3.1.2 Следующая подглава" in titles

    def test_no_numbered_sections_returns_single_none_section(self):
        text = "Просто текст без номеров секций.\nЕщё одна строка."
        sections = _split_into_sections(text)
        assert sections == [(None, text)]
