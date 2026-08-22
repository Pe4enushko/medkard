import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.formal_structure.validator import NMU_RE, FormalValidator, VisitType


def _visit(diagnoses=None, services=None):
    return {
        "Прием": {"GUID": "test-guid"},
        "Диагнозы": diagnoses or [],
        "Услуги": services or [{"Наименование": "Приём первичный"}],
    }


async def test_z11_1_in_diagnoses_gives_tuberculin_type():
    """Z11.1 читается из Диагнозы[].КодМКБ — контракт clinic-data-requirements.md."""
    got = await FormalValidator().get_visit_types(_visit(diagnoses=[{"КодМКБ": "Z11.1"}]))
    assert VisitType.PROPHYLACTIC_TUBERCULIN in got


async def test_z11_1_is_case_and_space_insensitive():
    got = await FormalValidator().get_visit_types(_visit(diagnoses=[{"КодМКБ": " z11.1 "}]))
    assert VisitType.PROPHYLACTIC_TUBERCULIN in got


async def test_z11_1_found_among_several_diagnoses():
    visit = _visit(diagnoses=[{"КодМКБ": "J06.9"}, {"КодМКБ": "Z11.1"}])
    got = await FormalValidator().get_visit_types(visit)
    assert VisitType.PROPHYLACTIC_TUBERCULIN in got


async def test_other_diagnosis_does_not_give_tuberculin_type():
    got = await FormalValidator().get_visit_types(_visit(diagnoses=[{"КодМКБ": "J06.9"}]))
    assert VisitType.PROPHYLACTIC_TUBERCULIN not in got


async def test_no_diagnoses_does_not_crash():
    got = await FormalValidator().get_visit_types(_visit())
    assert got  # тип определился по услуге, исключения нет


# Реальные коды из выгрузки МДС (~/projects/mdsgrep).
_REAL_A_CODES = [
    "A01.01.002",
    "A04.01.001",
    "A04.10.002",
    "A04.16.001",
    "A04.12.018",
    "A04.22.001",
    "A04.28.002",
    "A11.01.009",
    "A16.01.017",
    "A04.12.005.005",      # четыре сегмента
    "A04.20.001.001",
    "A11.22.002.001",
]

_REAL_B_CODES = [
    "B01.023.001",
    "B01.004.001",
    "B01.058.001",
    "B01.070.001",
    "B04.031.002",
    "B02.031.001",
]

# Внутренние артикулы МДС — номенклатурными кодами не являются и матчиться
# не должны ни до, ни после фикса.
_INTERNAL_ARTICLES = ["4.1.A2.201", "50.0.H95.201", "1.0.D2.202", "6.1.D1.401", "-"]


@pytest.mark.parametrize("code", _REAL_A_CODES)
def test_real_a_codes_match_nmu_re(code):
    """A-коды номенклатуры имеют 2 цифры в среднем сегменте."""
    assert NMU_RE.match(code), f"{code} не распознан как код номенклатуры"


@pytest.mark.parametrize("code", _REAL_B_CODES)
def test_real_b_codes_still_match(code):
    assert NMU_RE.match(code), f"{code} перестал распознаваться"


@pytest.mark.parametrize("code", _INTERNAL_ARTICLES)
def test_internal_articles_do_not_match(code):
    """Внутренние артикулы клиники не должны считаться кодами номенклатуры."""
    assert not NMU_RE.match(code), f"{code} ошибочно распознан как код номенклатуры"


@pytest.mark.parametrize("code", ["A04.16.001", "A09.05.023", "A11.22.002.001"])
async def test_a_code_service_gives_lab_research_type(code):
    visit = _visit(services=[{"Наименование": "Исследование", "Артикул": code}])
    got = await FormalValidator().get_visit_types(visit)
    assert VisitType.LAB_RESEARCH_INTERVENTION in got


async def test_a_code_with_trailing_space_is_handled():
    """В боевых данных встречаются коды с хвостовым пробелом."""
    visit = _visit(services=[{"Наименование": "УЗИ", "Артикул": "A04.12.018 "}])
    got = await FormalValidator().get_visit_types(visit)
    assert VisitType.LAB_RESEARCH_INTERVENTION in got


async def test_real_specialist_codes_give_primary_and_repeat():
    """Тип приёма даёт последний сегмент кода, а не специальность врача.

    Средний сегмент — это специальность (023 невролог, 031 педиатр, 015
    кардиолог). Пока классификатор требовал B01.070.*, каждая боевая карта
    поликлиники становилась OTHER и теряла 34 правила из 42.
    """
    v = FormalValidator()
    for code, expected in [
        ("B01.023.001", VisitType.PRIMARY),    # невролог первичный
        ("B01.023.002", VisitType.REPEAT),     # невролог повторный
        ("B01.031.002", VisitType.REPEAT),     # педиатр повторный
        ("B01.015.001", VisitType.PRIMARY),    # кардиолог первичный
        ("B01.047.001", VisitType.PRIMARY),    # терапевт первичный
        ("B04.031.002", VisitType.PROPHYLACTIC),  # профилактический приём
        ("B04.047.001", VisitType.DISPENSARY),    # диспансерный приём
    ]:
        got = await v.get_visit_types(_visit(services=[{"Наименование": "x", "Код": code}]))
        assert expected in got, f"{code} → {got}"


async def test_specialties_where_the_suffix_is_not_a_pair_are_excluded():
    """.001/.002 — приём не у каждой специальности; сверено чекером по 804н.

    B01.054.001 — «Осмотр (консультация) врача-физиотерапевта», единственная
    запись специальности; B01.030.002 — «Проведение комплексного аутопсийного
    исследования»; B04.015.001 — «Школа для больных с артериальной
    гипертензией». Чистое правило по окончанию назвало бы их первичным,
    повторным приёмом и профилактическим приёмом соответственно.
    """
    v = FormalValidator()
    for code in ("B01.054.001", "B01.030.002", "B01.052.001", "B04.015.001"):
        got = await v.get_visit_types(_visit(services=[{"Наименование": "x", "Код": code}]))
        assert got == {VisitType.OTHER}, f"{code} → {got}"


async def test_rare_appointment_pairs_are_left_to_the_service_name():
    """Устойчиво разбираются только .001/.002; остальные пары — по наименованию.

    В 804н пары участкового, подросткового и «беременной» врача идут дальше по
    списку (B01.047.005/.006 терапевт участковый), а между ними встречаются
    записи, приёмом не являющиеся. Гадать по окончанию там нельзя, поэтому такие
    услуги распознаются по наименованию, где клиника сама пишет вид приёма.
    """
    v = FormalValidator()
    by_code_only = [{"Наименование": "x", "Код": "B01.047.005"}]
    assert await v.get_visit_types(_visit(services=by_code_only)) == {VisitType.OTHER}

    with_name = [{"Наименование": "Приём врача-терапевта участкового первичный",
                  "Код": "B01.047.005"}]
    assert VisitType.PRIMARY in await v.get_visit_types(_visit(services=with_name))


async def test_code_outside_the_dictionary_does_not_block_the_name():
    """Код, о котором таблица молчит, не должен глушить разбор наименования.

    B01.070.001 — это «Медицинское освидетельствование на состояние опьянения»,
    а не первичный приём; раньше он давал PRIMARY, а любой неопознанный B01 —
    сразу OTHER, из-за чего наименование услуги уже не читалось.
    """
    v = FormalValidator()
    services = [{"Наименование": "Приём повторный", "Код": "B01.070.001"}]
    assert VisitType.REPEAT in await v.get_visit_types(_visit(services=services))


async def test_non_appointment_code_without_name_hint_is_other():
    v = FormalValidator()
    # B01.070.011 — патронаж выездной паллиативной бригадой, не приём.
    got = await v.get_visit_types(_visit(services=[{"Наименование": "x", "Код": "B01.070.011"}]))
    assert got == {VisitType.OTHER}


async def test_nmu_contradiction_uses_the_dictionary():
    """Противоречие «код против наименования» считается по 804н, а не по догадке."""
    v = FormalValidator()
    visit = _visit(services=[
        {"Наименование": "Приём первичный", "Код": "B01.023.002"},  # код: повторный
    ])
    finding = v._check_nmu_keyword_contradiction(visit)
    assert finding is not None
    assert finding["flag"] == "NMU_CODE_CONTRADICTION"
    assert "B01.023.002" in finding["issue"]

    agreeing = _visit(services=[{"Наименование": "Приём первичный", "Код": "B01.023.001"}])
    assert v._check_nmu_keyword_contradiction(agreeing) is None

    # Код вне словаря никакого утверждения о типе приёма не делает.
    unknown = _visit(services=[{"Наименование": "Приём первичный", "Код": "B01.070.011"}])
    assert v._check_nmu_keyword_contradiction(unknown) is None


def test_visit_type_vocabulary_matches_the_rules():
    """Словарь типов существует ради rules.json и не должен его опережать.

    Новый тип визита без правил, которые его используют, — это ключ, по которому
    ничего не отбирается; правило с типом, которого нет в перечислении, упадёт на
    _VISIT_TYPE_RULE_KEY при первом же аудите.
    """
    import audit.formal_structure.validator as v

    declared = {key for key in v._VISIT_TYPE_RULE_KEY.values()}
    used = {t for rule in v._RULES for t in rule["applies_to"]["visit_types"]} - {"all"}

    assert used <= declared, f"в rules.json есть типы вне перечисления: {sorted(used - declared)}"
    # OTHER — служебный ответ «тип не определён», правил под ним нет и быть не должно.
    assert declared - used == {"other"}, f"типы без единого правила: {sorted(declared - used - {'other'})}"


def test_code_table_never_yields_a_type_no_rule_uses():
    import audit.formal_structure.validator as v

    used = {t for rule in v._RULES for t in rule["applies_to"]["visit_types"]} - {"all"}
    for rule in v._CODE_RULES:
        if rule.visit_type is None:
            continue
        assert v._VISIT_TYPE_RULE_KEY[rule.visit_type] in used, rule


async def test_dispensary_visit_is_not_a_prophylactic_examination():
    """Диспансерный приём (168н/192н) и профилактический осмотр (404н) — разное.

    Пока оба давали PROPHYLACTIC, на диспансерном приёме срабатывали четыре
    правила 404н про объём ПМО, а правила про само диспансерное наблюдение —
    нет, потому что были объявлены только на первичном и повторном приёме.
    """
    v = FormalValidator()
    dispensary = await v.get_visit_types(
        _visit(services=[{"Наименование": "x", "Код": "B04.047.001"}])
    )
    assert dispensary == {VisitType.DISPENSARY}

    prophylactic = await v.get_visit_types(
        _visit(services=[{"Наименование": "x", "Код": "B04.047.002"}])
    )
    assert prophylactic == {VisitType.PROPHYLACTIC}

    flags = {r["flag_code"] for r in v.get_rules(dispensary, 54, ["I10"])}
    assert "ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО" in flags
    assert not {f for f in flags if f.startswith("ПРОФ_ВЗРОСЛЫЙ_")}

    prophylactic_flags = {r["flag_code"] for r in v.get_rules(prophylactic, 54, ["I10"])}
    assert {f for f in prophylactic_flags if f.startswith("ПРОФ_ВЗРОСЛЫЙ_")}
    assert "ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО" not in prophylactic_flags


async def test_dispensary_name_does_not_catch_дispanserizatsiya():
    """«Диспансеризация» — это ПМО по 404н, а не диспансерное наблюдение."""
    v = FormalValidator()
    got = await v.get_visit_types(
        _visit(services=[{"Наименование": "Профилактический осмотр в рамках диспансеризации"}])
    )
    assert got == {VisitType.PROPHYLACTIC}
