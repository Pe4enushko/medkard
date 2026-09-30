"""Аббревиатуры исследований, принятые моделью за лекарственные препараты.

Карта ЦДЗ-00867066 за 24.09.2026: в «Заметках» стоит план обследования
«Рекомендовано: 1. Контроль ЭКГ 2. КАК 3. ОАМ», и правило 1094н выдало
«в назначении 'КАК' отсутствуют наименование препарата, дозировка…».
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.formal_structure.validator import _drop_study_abbreviation_findings
from parsers.study_abbreviations import is_study_abbreviation

_LIST = json.loads(
    (ROOT / "resources" / "study_abbreviations.json").read_text(encoding="utf-8")
)["abbreviations"]

_FLAG = "НЕПОЛНОЕ_НАЗНАЧЕНИЕ_ПРЕПАРАТА"

_VISIT = {
    "Пациент": {"AGE": "16"},
    "ДанныеОсмотра": [
        {"Параметр": "Заметки", "Значение": "Рекомендовано: 1. Контроль ЭКГ 2. КАК 3. ОАМ"},
        {"Параметр": "Рекомендации и назначения:", "Значение": "Дона 1500 мг 1 порошок в день"},
    ],
}


def _finding(issue: str, flag: str = _FLAG) -> dict[str, str]:
    return {"flag": flag, "issue": issue}


def test_every_listed_abbreviation_is_recognised():
    for abbreviation in _LIST:
        assert is_study_abbreviation(abbreviation), abbreviation


def test_separators_and_case_do_not_matter():
    """«кл.ан.кр.», «КЛ АН КР» и «ЭХО-КГ» — те же сокращения, что в списке."""
    assert is_study_abbreviation("кл.ан.кр.")
    assert is_study_abbreviation("КЛ АН КР")
    assert is_study_abbreviation("ЭХО-КГ")
    assert is_study_abbreviation(" оак ")


def test_drug_names_are_not_abbreviations():
    """Сокращения препаратов в список не входят — иначе фильтр снимет настоящее замечание."""
    for name in ("АСК", "Д3", "В12", "Mg B6", "Дона", "КАК и Дона 1500 мг", ""):
        assert not is_study_abbreviation(name), name


def test_finding_about_a_study_is_dropped():
    findings = [_finding("В назначении 'КАК' отсутствуют дозировка и продолжительность")]
    assert _drop_study_abbreviation_findings(findings, _VISIT) == []


def test_field_label_next_to_the_abbreviation_does_not_save_the_finding():
    """Подпись поля в кавычках — не предмет замечания, и снятию она не мешает."""
    findings = [_finding("В поле 'Заметки' в назначении 'КАК' нет дозировки")]
    assert _drop_study_abbreviation_findings(findings, _VISIT) == []


def test_finding_about_a_real_drug_survives():
    findings = [_finding("В назначении 'Дона' отсутствует продолжительность лечения")]
    assert _drop_study_abbreviation_findings(findings, _VISIT) == findings


def test_finding_naming_a_drug_and_a_study_survives():
    """Рядом с исследованием назван препарат — врачу есть что исправить."""
    findings = [_finding("В назначениях 'КАК' и 'Дона' нет дозировки")]
    assert _drop_study_abbreviation_findings(findings, _VISIT) == findings


def test_finding_without_a_quoted_subject_survives():
    """Предмет не процитирован — судить не о чем, замечание остаётся."""
    findings = [_finding("Назначение препарата приведено без дозировки")]
    assert _drop_study_abbreviation_findings(findings, _VISIT) == findings


def test_other_flags_are_never_touched():
    """Фильтр знает только правила о назначении препарата."""
    findings = [_finding("В поле 'Заметки' указано 'КАК'", flag="ОБНАРУЖЕНЫ_ЗАГЛУШКИ")]
    assert _drop_study_abbreviation_findings(findings, _VISIT) == findings
