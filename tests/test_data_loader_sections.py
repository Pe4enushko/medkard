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

    def test_non_breaking_space_after_section_number_is_supported(self):
        text = "3.1\N{NO-BREAK SPACE}Диетотерапия\nТекст.\n"

        sections = _split_into_sections(text)

        assert [title for title, _ in sections] == ["3.1\N{NO-BREAK SPACE}Диетотерапия"]

    def test_section_number_and_title_may_be_split_across_lines(self):
        text = (
            "3.5.4 \n"
            "Ультразвуковой контроль\n"
            "Текст раздела.\n\n"
            "3.6 Следующий раздел\n"
            "Следующий текст.\n"
        )

        sections = _split_into_sections(text)

        assert [title for title, _ in sections] == [
            "3.5.4 \nУльтразвуковой контроль",
            "3.6 Следующий раздел",
        ]

    def test_bare_section_number_does_not_consume_next_numbered_title(self):
        text = (
            "1.6\n"
            "1.7 Клиническая картина заболевания\n"
            "Текст раздела.\n"
        )

        sections = _split_into_sections(text)

        assert [title for title, _ in sections] == [
            "1.7 Клиническая картина заболевания"
        ]

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

    def test_bibliography_stops_last_numbered_section(self):
        text = (
            "8.4 Критерии оценки качества\n"
            "1. Выполнена спирометрия — Да/Нет.\n\n"
            "Список литературы\n"
            "1. Первая публикация.\n"
            "2. Вторая публикация.\n"
        )

        sections = _split_into_sections(text)

        assert [title for title, _ in sections] == [
            "8.4 Критерии оценки качества",
            "Список литературы",
        ]
        assert "Первая публикация" not in sections[0][1]
        assert "Первая публикация" in sections[1][1]

    @pytest.mark.parametrize(
        "heading",
        [
            "СПИСОК ЛИТЕРАТУРЫ",
            "XIII. Список литературы",
            "9. Список литературы",
            "9.Список литературы",
            "   Список литературы",
            "Список литературы:",
            "Список литературы.",
            "Список использованной литературы",
            "Список использованных источников",
            "Библиографический список",
            "Литература",
        ],
    )
    def test_bibliography_heading_variants_are_boundaries(self, heading):
        text = f"8.4 Критерии\nТаблица.\n\n{heading}\nИсточник.\n"

        sections = _split_into_sections(text)

        assert [title for title, _ in sections] == ["8.4 Критерии", heading.strip()]

    def test_bibliography_toc_entry_is_not_a_boundary(self):
        text = (
            "8.4 Критерии оценки качества\n"
            "Список литературы....................98\n"
            "Текст критериев.\n"
        )

        sections = _split_into_sections(text)

        assert len(sections) == 1
        assert "Текст критериев" in sections[0][1]

    @pytest.mark.parametrize(
        "toc_line",
        [
            "Список литературы....................98",
            "Список литературы 98",
        ],
    )
    def test_bibliography_toc_variants_are_not_boundaries(self, toc_line):
        text = f"8.4 Критерии оценки качества\n{toc_line}\nТекст критериев.\n"

        sections = _split_into_sections(text)

        assert len(sections) == 1

    def test_bibliography_heading_without_numbered_sections_keeps_all_text(self):
        text = "Введение без номера.\n\nСписок литературы\n1. Источник.\n"

        sections = _split_into_sections(text)

        assert sections == [(None, text)]

    @pytest.mark.parametrize(
        "heading",
        [
            "Критерии оценки качества медицинской помощи",
            "8. Критерии оценки качества медицинской помощи",
            "XII. Критерии оценки качества медицинской помощи",
            "6- Критерии оценки качества медицинской помощи",
            "Таблица 1. Критерии оценки качества медицинской помощи",
            "Таблица 10.1 - Критерии оценки качества первичной помощи",
        ],
    )
    def test_quality_criteria_heading_variants_are_boundaries(self, heading):
        text = (
            "6.4 Организация оказания помощи\n"
            "Организационный текст.\n\n"
            f"{heading}\n"
            "№ п/п | Критерий | Да/Нет\n"
        )

        sections = _split_into_sections(text)

        assert [title for title, _ in sections] == [
            "6.4 Организация оказания помощи",
            heading,
        ]

    @pytest.mark.parametrize(
        "line",
        [
            "Критерии оценки качества медицинской помощи....................98",
            "Критерии оценки качества медицинской помощи приведены в таблице 12.",
            "Критерии оценки качества медицинской помощи: приказ Минздрава России.",
        ],
    )
    def test_quality_criteria_references_are_not_boundaries(self, line):
        text = f"6.4 Организация оказания помощи\n{line}\nПродолжение текста.\n"

        sections = _split_into_sections(text)

        assert len(sections) == 1
        assert sections[0][0] == "6.4 Организация оказания помощи"

    def test_multiline_criteria_toc_entry_is_not_a_boundary(self):
        text = (
            "3.3 Лечение\n"
            "Текст.\n"
            "Критерии оценки качества специализированной помощи взрослым\n"
            "при заболевании................................................137\n"
            "Продолжение оглавления.\n"
        )

        sections = _split_into_sections(text)

        assert len(sections) == 1
        assert sections[0][0] == "3.3 Лечение"

    @pytest.mark.parametrize(
        "heading",
        [
            "Приложение А1. Состав рабочей группы",
            "Приложение А3/1. Справочные материалы",
            "Приложение Б1. Алгоритм действий врача",
            "Приложение Г1 - ГN. Шкалы оценки состояния пациента",
        ],
    )
    def test_descriptive_appendix_headings_are_boundaries(self, heading):
        text = (
            "6.4 Организация оказания помощи\n"
            "Организационный текст.\n\n"
            "8. Критерии оценки качества медицинской помощи\n"
            "Таблица критериев.\n\n"
            f"{heading}\n"
            "Содержимое приложения.\n"
        )

        sections = _split_into_sections(text)

        assert [title for title, _ in sections] == [
            "6.4 Организация оказания помощи",
            "8. Критерии оценки качества медицинской помощи",
            heading,
        ]

    @pytest.mark.parametrize(
        "line",
        [
            "Приложение",
            "Приложение А3/5.",
            "Приложение А1. Состав рабочей группы....................186",
        ],
    )
    def test_appendix_references_and_toc_entries_are_not_boundaries(self, line):
        text = f"8. Критерии оценки качества\nТаблица.\n{line}\nПродолжение.\n"

        sections = _split_into_sections(text)

        assert len(sections) == 1
