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


def test_rule_without_a_normative_source_says_so_out_loud():
    """Пустой source — это заявление «нормативного основания нет».

    Правило внутренней непротиворечивости имеет право существовать, но не имеет
    права выглядеть нормативным: сослаться на приказ, который такого не требует,
    хуже, чем не сослаться вовсе.
    """
    for rule in _DOC["rules"]:
        if rule.get("source"):
            continue
        assert "НОРМАТИВНОГО ОСНОВАНИЯ НЕТ" in rule["source_ref"], rule["rule_id"]


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
    got = await _flags(_visit(diagnoses=[{"КодМКБ": "S52.5"}]))
    assert "ТРАВМА_БЕЗ_КОДА_ВНЕШНЕЙ_ПРИЧИНЫ" in got


async def test_poisoning_counts_as_injury():
    got = await _flags(_visit(diagnoses=[{"КодМКБ": "T36.0"}]))
    assert "ТРАВМА_БЕЗ_КОДА_ВНЕШНЕЙ_ПРИЧИНЫ" in got


async def test_injury_with_external_cause_is_clean():
    got = await _flags(_visit(diagnoses=[{"КодМКБ": "S52.5"}, {"КодМКБ": "W01.0"}]))
    assert "ТРАВМА_БЕЗ_КОДА_ВНЕШНЕЙ_ПРИЧИНЫ" not in got


async def test_card_without_injury_is_not_asked_for_a_cause():
    got = await _flags(_visit(diagnoses=[{"КодМКБ": "J06.9"}]))
    assert "ТРАВМА_БЕЗ_КОДА_ВНЕШНЕЙ_ПРИЧИНЫ" not in got


async def test_external_cause_is_not_required_on_a_repeat_visit():
    """Строка «Внешняя причина при травмах» стоит в разделе первичного приёма.

    Раздел «Медицинское наблюдение в динамике» её не содержит: там дата,
    жалобы, данные наблюдения, назначения, препараты, лист нетрудоспособности
    и рецепты. Первая редакция правила требовала больше, чем форма.
    """
    got = await _flags(_visit(diagnoses=[{"КодМКБ": "S52.5"}], services=_REPEAT))
    assert "ТРАВМА_БЕЗ_КОДА_ВНЕШНЕЙ_ПРИЧИНЫ" not in got


async def test_external_cause_is_searched_across_all_diagnoses():
    """Проблемный лист приходит целиком: код причины может стоять не рядом."""
    got = await _flags(
        _visit(diagnoses=[{"КодМКБ": "I10"}, {"КодМКБ": "S52.5"}, {"КодМКБ": "Y04.0"}])
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


# ── услуга без кода номенклатуры ──────────────────────────────────────────────

_NO_CODE_FLAG = "УСЛУГА_БЕЗ_КОДА_НОМЕНКЛАТУРЫ"


def _with_services(services):
    return {
        "Прием": {"GUID": "g"},
        "Пациент": {"AGE": 40},
        "Диагнозы": [{"КодМКБ": "J06.9"}],
        "Услуги": services,
        "ДанныеОсмотра": [],
    }


async def test_service_without_any_code_is_flagged():
    """Техничка интеграции: «каждый элемент содержит КодЕГИСЗ», «все поля
    обязательны». Без кода услугу не сопоставить с 804н и не собрать СЭМД."""
    got = await _flags(_with_services(
        [{"КодЕГИСЗ": "", "Артикул": "", "Код": "00000003324",
          "Наименование": "Прием интегративный (осмотр, консультация) врача акушера-гинеколога"}]
    ))
    assert _NO_CODE_FLAG in got


async def test_code_in_artikul_is_enough():
    """На боевых картах КодЕГИСЗ пуст, а код лежит в Артикул.

    Такая услуга с 804н сопоставима — замечание врачу было бы шумом; это
    дефект интеграции, а не записи приёма.
    """
    got = await _flags(_with_services(
        [{"КодЕГИСЗ": "", "Артикул": "B01.023.001",
          "Наименование": "Прием (осмотр, консультация) врача невролога  Чепухина Л.А."}]
    ))
    assert _NO_CODE_FLAG not in got


async def test_finding_names_the_service_that_lacks_the_code():
    """Иначе врач не поймёт, к какой из услуг замечание."""
    issue = None
    for f in await DeterministicValidator().validate(_with_services([
        {"Артикул": "B01.023.001", "Наименование": "Приём невролога"},
        {"Артикул": "", "Наименование": "Массаж"},
    ])):
        if f["flag"] == _NO_CODE_FLAG:
            issue = f["issue"]
    assert issue is not None
    assert "Массаж" in issue and "невролога" not in issue


async def test_a_visit_without_services_at_all_is_not_flagged():
    """Пустой массив услуг — отдельный случай, здесь сообщать не о чем."""
    assert _NO_CODE_FLAG not in await _flags(_with_services([]))


# ── вид приёма против кода услуги ─────────────────────────────────────────────

_CONTRADICTION_FLAG = "ВИД_ПРИЁМА_НЕ_СООТВЕТСТВУЕТ_КОДУ_УСЛУГИ"


def _consultation(code, said, param="Консультация перв/повтор"):
    return {
        "Прием": {"GUID": "g"},
        "Пациент": {"AGE": 40},
        "Диагнозы": [{"КодМКБ": "I10"}],
        "Услуги": [{"КодЕГИСЗ": code, "Наименование": "Приём невролога"}],
        "ДанныеОсмотра": [{"Параметр": param, "Значение": said}],
    }


async def test_repeat_visit_billed_as_primary_is_flagged():
    """МДС выставляет все консультации кодом .001 независимо от записи.

    Замер на выгрузках: 697 карт из 8874. 804н разводит окончания .001 и .002
    («…врача-невролога первичный» / «…повторный»), клиника этим не пользуется.
    """
    assert _CONTRADICTION_FLAG in await _flags(_consultation("B01.023.001", "Повторная"))


async def test_the_opposite_direction_is_flagged_too():
    assert _CONTRADICTION_FLAG in await _flags(_consultation("B01.023.002", "Первичная"))


async def test_agreement_is_silent():
    for code, said in (("B01.023.001", "Первичная"), ("B01.023.002", "Повторная")):
        assert _CONTRADICTION_FLAG not in await _flags(_consultation(code, said)), code


async def test_value_outside_the_pair_is_not_a_contradiction():
    """В поле встречается «Прием в медицинском центре» — это не вид приёма."""
    got = await _flags(_consultation("B01.023.001", "Прием в медицинском центре"))
    assert _CONTRADICTION_FLAG not in got


async def test_planning_field_is_not_read_as_a_visit_type():
    """«Консультация (повт.план итд)» — план следующей явки, не текущий приём.

    Там пишут «повторная с результатами обследования»; искать в этом поле
    «повторн» нельзя. Правило читает только поля из своего списка.
    """
    got = await _flags(_consultation(
        "B01.023.001", "повторная с результатами обследования",
        param="Консультация (повт.план итд)",
    ))
    assert _CONTRADICTION_FLAG not in got


async def test_both_halves_of_the_pair_in_one_card_are_not_a_contradiction():
    """Приём и первичный, и повторный в одной карте — сравнивать не с чем."""
    card = _consultation("B01.023.001", "Повторная")
    card["Услуги"].append({"КодЕГИСЗ": "B01.023.002", "Наименование": "Приём невролога повторный"})
    assert _CONTRADICTION_FLAG not in await _flags(card)


async def test_verdict_comes_from_the_code_not_from_the_service_name():
    """Наименование и запись пишет одна и та же клиника — сверять их бессмысленно.

    Услуга без кода: вердикту кода взяться неоткуда, расхождения нет.
    """
    card = _consultation("", "Повторная")
    card["Услуги"] = [{"Наименование": "Прием (осмотр, консультация) врача-невролога первичный"}]
    assert _CONTRADICTION_FLAG not in await _flags(card)



# ── наименование услуги по ЕГИСЗ ──────────────────────────────────────────────

_NAME_FLAG = "НАИМЕНОВАНИЕ_УСЛУГИ_НЕ_ПО_НОМЕНКЛАТУРЕ"
_NEUROLOGIST = "Прием (осмотр, консультация) врача-невролога первичный"


def _egisz(code, name, **extra):
    return {
        "Прием": {"GUID": "g"}, "Пациент": {"AGE": 40},
        "Диагнозы": [{"КодМКБ": "I10"}], "ДанныеОсмотра": [],
        "Услуги": [{"КодЕГИСЗ": code, "НаименованиеЕГИСЗ": name, **extra}],
    }


async def test_egisz_name_matching_the_order_is_clean():
    assert _NAME_FLAG not in await _flags(_egisz("B01.023.001", _NEUROLOGIST))


async def test_typography_is_not_a_mismatch():
    """Клиника роняет дефис, ставит двойные пробелы и «е» вместо «ё»."""
    assert _NAME_FLAG not in await _flags(
        _egisz("B01.023.001", "Прием  (осмотр, консультация) врача невролога первичный")
    )


async def test_name_with_the_doctor_surname_is_a_mismatch():
    got = await _flags(_egisz("B01.023.001", _NEUROLOGIST + " Чепухина Л.А."))
    assert _NAME_FLAG in got


async def test_own_free_form_name_is_never_compared():
    """Поле «Наименование» клиника заполняет свободно — сверять его не с чем.

    Замер: у МДС НаименованиеЕГИСЗ во всех 18 175 строках услуг — побайтовая
    копия собственного наименования, а КодЕГИСЗ пуст.
    """
    card = _egisz("", "Приём невролога Чепухина Л.А.", Артикул="B01.023.001",
                  Наименование="Приём невролога Чепухина Л.А.")
    assert _NAME_FLAG not in await _flags(card)


async def test_empty_egisz_name_is_not_a_mismatch():
    """Пустое поле — это другой дефект, у него своё правило про код."""
    assert _NAME_FLAG not in await _flags(_egisz("B01.023.001", ""))


async def test_code_missing_from_our_dump_is_skipped():
    """Выгрузка заморожена на редакции 24.09.2020: незнание не дефект карты."""
    assert _NAME_FLAG not in await _flags(_egisz("B01.999.999", "Что-то новое"))


# ── нормативный минимум записи по 274н ────────────────────────────────────────

_MINIMUM_FLAG = "НЕ_ЗАПОЛНЕН_НОРМАТИВНЫЙ_МИНИМУМ_ЗАПИСИ"
_PRIMARY_SVC = [{"Код": "B01.047.001", "Наименование": "Приём терапевта первичный"}]
_REPEAT_SVC = [{"Код": "B01.047.002", "Наименование": "Приём терапевта повторный"}]


def _record(fields, services):
    return {
        "Прием": {"GUID": "g"}, "Пациент": {"AGE": 40},
        "Диагнозы": [{"КодМКБ": "I10"}], "Услуги": services,
        "ДанныеОсмотра": [{"Параметр": n, "Значение": "заполнено"} for n in fields],
    }


async def _minimum_issue(card):
    for f in await DeterministicValidator().validate(card):
        if f["flag"] == _MINIMUM_FLAG:
            return f["issue"]
    return None


async def test_primary_visit_needs_complaints_anamnesis_and_objective_data():
    """Раздел «Записи врачей-специалистов» формы 025/у."""
    issue = await _minimum_issue(_record([], _PRIMARY_SVC))
    assert issue is not None
    for slot in ("Жалобы", "Анамнез заболевания, жизни", "Объективные данные"):
        assert slot in issue, slot


async def test_full_primary_record_is_clean():
    card = _record(
        ["Жалобы на момент осмотра", "Анамнез заболевания", "Объективные данные"],
        _PRIMARY_SVC,
    )
    assert await _minimum_issue(card) is None


async def test_slot_is_matched_by_word_stem_not_by_exact_name():
    """У МДС 180 разных имён: «Жалобы невролог», «Анамнез кардиолог»."""
    card = _record(
        ["Жалобы невролог", "Анамнез невролог", "Объективные данные невролог"],
        _PRIMARY_SVC,
    )
    assert await _minimum_issue(card) is None


async def test_repeat_visit_is_not_asked_for_anamnesis():
    """Раздел «Медицинское наблюдение в динамике» анамнеза не содержит.

    Замер это подтверждает: у Алёнки анамнез есть на 91% первичных приёмов и
    на 11,4% повторных.
    """
    issue = await _minimum_issue(_record(["Жалобы", "Динамика состояния"], _REPEAT_SVC))
    assert issue is None


async def test_repeat_visit_needs_dynamics_instead():
    issue = await _minimum_issue(_record(["Жалобы"], _REPEAT_SVC))
    assert issue is not None
    assert "Данные наблюдения в динамике" in issue
    assert "Анамнез" not in issue


async def test_empty_field_does_not_count_as_filled():
    """1С не присылает незаполненное поле, но пустая строка встречается."""
    card = _record([], _PRIMARY_SVC)
    card["ДанныеОсмотра"] = [{"Параметр": "Жалобы", "Значение": "  "}]
    assert "Жалобы" in (await _minimum_issue(card) or "")


async def test_minimum_does_not_depend_on_a_known_template():
    """274н обязателен для всех — в отличие от required_fields.json.

    Отсутствие слота в шаблоне клиники это дефект шаблона, а не повод
    промолчать; шаблон здесь не спрашивается вовсе.
    """
    card = _record(["Совершенно незнакомое поле"], _PRIMARY_SVC)
    assert await _minimum_issue(card) is not None


async def test_prescriptions_are_not_in_the_minimum():
    """Визит может законно ничего не назначать.

    У МДС слот назначений заполнен в 22,9% карт и называется иначе — «План
    лечения», «Рекомендованное лечение».
    """
    card = _record(
        ["Жалобы", "Анамнез заболевания", "Объективные данные"], _PRIMARY_SVC,
    )
    assert await _minimum_issue(card) is None
