"""Правила, проверяемые сравнением, без обращения к модели."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.deterministic import DeterministicValidator
from audit.deterministic.validator import _RULES_PATH

_DOC = json.loads(_RULES_PATH.read_text(encoding="utf-8"))


def _visit(diagnoses=None, services=None, inspection=None):
    return {
        "Прием": {"GUID": "test-guid"},
        "Диагнозы": diagnoses or [],
        "Услуги": services or [{"Код": "B01.047.001", "Наименование": "Прием терапевта первичный"}],
        "ДанныеОсмотра": inspection or [],
    }


_REPEAT = [{"Код": "B01.047.002", "Наименование": "Прием терапевта повторный"}]


async def _flags(visit) -> set[str]:
    return {f["flag"] for f in await DeterministicValidator().validate(visit)}


# ── схема файла правил ────────────────────────────────────────────────────────

def test_every_rule_carries_its_source_and_issue():
    """Без source_ref находку нечем защитить перед врачом, без issue — нечего показать."""
    for rule in _DOC["rules"]:
        assert rule.get("flag_code"), rule
        assert rule.get("issue"), rule["rule_id"]
        assert rule.get("source_ref"), rule["rule_id"]
        assert rule.get("verified_at"), rule["rule_id"]


def test_disabled_rule_says_why():
    for rule in _DOC["rules"]:
        if not rule.get("enabled", True):
            assert rule.get("disabled_reason"), rule["rule_id"]


def test_flag_codes_are_unique():
    codes = [r["flag_code"] for r in _DOC["rules"]]
    assert len(codes) == len(set(codes)), codes


def test_no_rule_reads_inspection_fields():
    """Проверки полей осмотра живут в formal_structure.required_fields.

    Там они привязаны к шаблону записи. 1С не присылает незаполненное поле
    вовсе, поэтому без шаблона «поля нет» неотличимо от «такого поля у клиники
    не бывает», и правило палило бы по всем картам чужого шаблона.
    """
    for rule in _DOC["rules"]:
        if rule.get("enabled", True):
            assert rule["check"]["kind"] != "inspection_field_nonempty", rule["rule_id"]


# ── диагноз с кодом МКБ на первичном приёме ───────────────────────────────────

async def test_primary_visit_without_any_diagnosis_is_flagged():
    assert "ПЕРВИЧНЫЙ_ПРИЁМ_БЕЗ_КОДА_МКБ" in await _flags(_visit())


async def test_primary_visit_with_coded_diagnosis_is_clean():
    got = await _flags(_visit(diagnoses=[{"КодМКБ": "J06.9"}]))
    assert "ПЕРВИЧНЫЙ_ПРИЁМ_БЕЗ_КОДА_МКБ" not in got


async def test_diagnosis_without_a_code_does_not_count():
    """Строка «Диагнозы» без КодМКБ — это не диагноз с кодом по МКБ."""
    got = await _flags(_visit(diagnoses=[{"НаименованиеМКБ": "ОРВИ", "КодМКБ": ""}]))
    assert "ПЕРВИЧНЫЙ_ПРИЁМ_БЕЗ_КОДА_МКБ" in got


async def test_repeat_visit_is_out_of_scope():
    """274н требует код МКБ в разделе первичного приёма.

    Раздел «Медицинское наблюдение в динамике» такой строки не содержит —
    расширять скоуп на повторные приёмы не на что.
    """
    got = await _flags(_visit(services=_REPEAT))
    assert "ПЕРВИЧНЫЙ_ПРИЁМ_БЕЗ_КОДА_МКБ" not in got


# ── внешняя причина при травме ────────────────────────────────────────────────

async def test_injury_without_external_cause_is_flagged():
    got = await _flags(_visit(diagnoses=[{"КодМКБ": "S52.5"}], services=_REPEAT))
    assert "ТРАВМА_БЕЗ_КОДА_ВНЕШНЕЙ_ПРИЧИНЫ" in got


async def test_poisoning_counts_as_injury():
    got = await _flags(_visit(diagnoses=[{"КодМКБ": "T36.0"}], services=_REPEAT))
    assert "ТРАВМА_БЕЗ_КОДА_ВНЕШНЕЙ_ПРИЧИНЫ" in got


async def test_injury_with_external_cause_is_clean():
    got = await _flags(
        _visit(diagnoses=[{"КодМКБ": "S52.5"}, {"КодМКБ": "W01.0"}], services=_REPEAT)
    )
    assert "ТРАВМА_БЕЗ_КОДА_ВНЕШНЕЙ_ПРИЧИНЫ" not in got


async def test_card_without_injury_is_not_asked_for_a_cause():
    got = await _flags(_visit(diagnoses=[{"КодМКБ": "J06.9"}], services=_REPEAT))
    assert "ТРАВМА_БЕЗ_КОДА_ВНЕШНЕЙ_ПРИЧИНЫ" not in got


async def test_external_cause_is_searched_across_all_diagnoses():
    """Проблемный лист приходит целиком: код причины может стоять не рядом."""
    got = await _flags(
        _visit(
            diagnoses=[{"КодМКБ": "I10"}, {"КодМКБ": "S52.5"}, {"КодМКБ": "Y04.0"}],
            services=_REPEAT,
        )
    )
    assert "ТРАВМА_БЕЗ_КОДА_ВНЕШНЕЙ_ПРИЧИНЫ" not in got


# ── скоуп и выключенные правила ───────────────────────────────────────────────

async def test_disabled_rule_never_fires():
    """protocol_card_number_present выключено: поля номера карты в выгрузке нет."""
    got = await _flags(
        _visit(services=[{"Код": "A04.10.002", "Наименование": "УЗИ сердца"}])
    )
    assert "ПРОТОКОЛ_БЕЗ_НОМЕРА_МЕДКАРТЫ" not in got


# ── контролируемые показатели 168н ────────────────────────────────────────────

import json as _json
import tempfile

import pytest


@pytest.fixture(scope="module")
def indicators_validator():
    """Правило заведено выключенным до эвала — включаем копию только для теста."""
    doc = _json.loads(_RULES_PATH.read_text(encoding="utf-8"))
    for rule in doc["rules"]:
        if rule["rule_id"] == "dispensary_controlled_indicators":
            rule["enabled"] = True
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        _json.dump(doc, f, ensure_ascii=False)
        path = f.name
    return DeterministicValidator(path)


_INDICATORS_FLAG = "НЕ_ОТРАЖЕНЫ_КОНТРОЛИРУЕМЫЕ_ПОКАЗАТЕЛИ"
_DISPENSARY = [{"Код": "B04.047.001", "Наименование": "Диспансерный приём терапевта"}]


def _card(age=54, icd="I10", services=None, text="Жалоб нет"):
    return {
        "Прием": {"GUID": "g"},
        "Пациент": {"AGE": age},
        "Диагнозы": [{"КодМКБ": icd}],
        "Услуги": services or _DISPENSARY,
        "ДанныеОсмотра": [{"Параметр": "Осмотр", "Значение": text}],
    }


async def _issue(v, card):
    for f in await v.validate(card):
        if f["flag"] == _INDICATORS_FLAG:
            return f["issue"]
    return None


async def test_missing_indicators_are_listed_in_one_finding(indicators_validator):
    """Медиана — шесть показателей на строку; шесть отдельных замечаний врач
    прочитал бы как шесть дефектов."""
    issue = await _issue(indicators_validator, _card())
    assert issue is not None
    assert issue.count(":") == 1
    assert "Артериальное давление" in issue and "ЧСС" not in issue.split(":")[0]


async def test_indicator_written_by_the_doctor_is_counted(indicators_validator):
    """Приказ пишет «АД, ЧСС», врач — «АД 130/80, пульс 72». Это те же показатели."""
    issue = await _issue(
        indicators_validator,
        _card(text="Вес 82 кг, ИМТ 27,1. АД 130/80, пульс 72. Окружность талии 94 см. Не курит."),
    )
    assert issue is not None
    for named in ("Артериальное давление", "Частота сердечных сокращений",
                  "Вес, индекс массы тела", "Окружность талии", "Статус курения"):
        assert named not in issue, named


async def test_rule_is_silent_on_a_diagnosis_outside_the_appendices(indicators_validator):
    assert await _issue(indicators_validator, _card(icd="J06.9")) is None


async def test_rule_is_silent_on_an_ordinary_visit(indicators_validator):
    """168н — про диспансерное наблюдение. На острой пневмонии J12 требовать
    вес и статус курения было бы неверно; тип приёма это отсекает."""
    ordinary = [{"Код": "B01.047.001", "Наименование": "Приём терапевта первичный"}]
    assert await _issue(indicators_validator, _card(services=ordinary)) is None
    assert await _issue(indicators_validator, _card(icd="J12.9", services=ordinary)) is None


async def test_rule_is_silent_for_minors(indicators_validator):
    """168н п. 1 — взрослые 18 лет и старше; несовершеннолетние по 192н,
    а в нём перечней с кодами МКБ нет."""
    assert await _issue(indicators_validator, _card(age=12)) is None


async def test_unknown_age_keeps_the_rule_silent(indicators_validator):
    card = _card()
    card["Пациент"] = {}
    assert await _issue(indicators_validator, card) is None


async def test_indicators_of_several_diagnoses_are_united(indicators_validator):
    """168н п. 9: перечень «должен включать все параметры, соответствующие
    каждому заболеванию»."""
    card = _card()
    card["Диагнозы"] = [{"КодМКБ": "I10"}, {"КодМКБ": "E11"}]
    both = await _issue(indicators_validator, card)
    card["Диагнозы"] = [{"КодМКБ": "I10"}]
    one = await _issue(indicators_validator, card)
    assert len(both) > len(one)
    assert "Гликированный гемоглобин" in both and "Гликированный гемоглобин" not in one


async def test_rule_stays_disabled_in_the_shipped_file():
    """Включать только после эвала: список недостающего может оказаться длинным
    почти на каждом диспансерном приёме."""
    doc = _json.loads(_RULES_PATH.read_text(encoding="utf-8"))
    rule = next(r for r in doc["rules"] if r["rule_id"] == "dispensary_controlled_indicators")
    assert rule["enabled"] is False
    assert rule["disabled_reason"]
