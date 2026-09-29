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
        if not isinstance(rule.visit_type, v.VisitType):
            continue  # NO_GUESS — вердикта нет
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


async def test_prophylactic_counselling_is_not_a_prophylactic_examination():
    """B04.070.* — консультирование, а не профилактический осмотр по 404н.

    Наименование «Индивидуальное краткое профилактическое консультирование»
    содержит «профилактическ», и разбор наименования делал из него
    профилактический осмотр: на карту садились четыре правила 404н про объём
    ПМО, которых консультирование не обязано выполнять.
    """
    v = FormalValidator()
    got = await v.get_visit_types(
        _visit(services=[{
            "Код": "B04.070.002",
            "Наименование": "Индивидуальное краткое профилактическое консультирование "
                            "по коррекции факторов риска развития неинфекционных заболеваний",
        }])
    )
    assert got == {VisitType.OTHER}


async def test_counselling_marked_primary_is_not_a_primary_visit():
    """У B04.070.003/004 «первичное»/«повторное» сказано про консультирование."""
    v = FormalValidator()
    got = await v.get_visit_types(
        _visit(services=[{
            "Код": "B04.070.003",
            "Наименование": "Индивидуальное углубленное профилактическое консультирование "
                            "по коррекции факторов риска развития неинфекционных заболеваний первичное",
        }])
    )
    assert VisitType.PRIMARY not in got
    assert VisitType.PROPHYLACTIC not in got


async def test_barred_code_does_not_mute_a_decided_code_of_the_same_service():
    """Запрет глушит только разбор наименования, не вердикт соседнего кода.

    В одной строке услуги приходят и Артикул клиники, и Код, и КодЕГИСЗ:
    запрет по одному из них не должен отменять определённый вид приёма,
    вынесенный по другому.
    """
    v = FormalValidator()
    got = await v.get_visit_types(
        _visit(services=[{
            "Код": "B04.070.002",
            "КодЕГИСЗ": "B01.047.001",
            "Наименование": "Профилактическое консультирование",
        }])
    )
    assert got == {VisitType.PRIMARY}


async def test_name_still_decides_where_the_table_only_forbids_the_ending_rule():
    """Списки _NOT_A_PAIR запрещают правило окончания, но не наименование.

    У B01.070.006 окончание .006 ничего не значит, а наименование —
    единственный верный источник; сверено прогоном по всей номенклатуре 804н.
    """
    v = FormalValidator()
    got = await v.get_visit_types(
        _visit(services=[{
            "Код": "B01.070.006",
            "Наименование": "Прием (осмотр, консультация) врача по паллиативной "
                            "медицинской помощи первичный",
        }])
    )
    assert got == {VisitType.PRIMARY}


async def test_diagnostic_complex_is_not_a_primary_visit():
    """B03 — «сложные диагностические услуги» (п. 5.1 приказа), а не приём.

    B03.005.003 «Исследование сосудисто-тромбоцитарного первичного гемостаза»
    разбор наименования делал первичным приёмом: раздела B03 в таблице не было
    вовсе, и код проваливался в наименование.
    """
    got = await FormalValidator().get_visit_types(
        _visit(services=[{
            "Код": "B03.005.003",
            "Наименование": "Исследование сосудисто-тромбоцитарного первичного гемостаза",
        }])
    )
    assert got == {VisitType.LAB_RESEARCH_INTERVENTION}


async def test_nursing_care_and_rehabilitation_are_no_visit_type():
    """B02 сестринский уход и B05 реабилитация — не приём ни одного нашего типа."""
    v = FormalValidator()
    for code in ("B02.003.003", "B05.023.002.001"):
        got = await v.get_visit_types(
            _visit(services=[{"Код": code, "Наименование": "услуга"}])
        )
        assert got == {VisitType.OTHER}, code


async def test_four_group_code_never_reads_the_third_group_as_the_ending():
    """B01.003.004.001 — «Местная анестезия», а не первичный приём.

    Окончание читается только у кода из трёх групп. Брать последнюю группу
    тоже нельзя: тогда .001 и .002 анестезии станут первичным и повторным
    приёмом.
    """
    import audit.formal_structure.validator as v

    assert v.classify_code("B01.003.004.001") is None
    assert v.classify_code("B01.003.004.002") is None
    # три группы — правило работает как прежде
    assert v.classify_code("B01.003.001") is VisitType.PRIMARY


async def test_psychological_counselling_is_neither_visit_nor_research():
    """B03.070.001/002 — консультирование, но не диагностический комплекс.

    Именно два кода, а не весь B03.070: с .003 там идут настоящие комплексы.
    """
    import audit.formal_structure.validator as v

    assert v.classify_code("B03.070.001") is v.NO_GUESS
    assert v.classify_code("B03.070.002") is v.NO_GUESS
    assert v.classify_code("B03.070.003") is VisitType.LAB_RESEARCH_INTERVENTION


async def test_other_names_the_services_it_could_not_classify(caplog):
    """OTHER должен говорить, на чём именно споткнулся.

    Без этого по логу не понять ни причину, ни сколько карт туда падает, —
    а решать, какие правила давать типу OTHER, можно только посмотрев на них.
    """
    import logging

    v = FormalValidator()
    with caplog.at_level(logging.WARNING):
        got = await v.get_visit_types(
            _visit(services=[
                {"Код": "B05.023.002.001", "Наименование": "Услуги по медицинской реабилитации"},
                {"Артикул": "X-1", "Наименование": "Забор материала"},
            ])
        )
    assert got == {VisitType.OTHER}
    text = caplog.text
    assert "B05.023.002.001" in text
    assert "Забор материала" in text
    assert "без кода" in text


# Услуга, о которой не может сказать ни код, ни наименование: только на такой
# карте Z-код и решает вид приёма.
_SILENT_SERVICE = [{"Наименование": "Осмотр"}]


async def test_z_codes_name_the_reason_when_services_say_nothing():
    """274н прил. 4 п. 9.13 делит посещения на «по заболеванию» и «Z00-Z99».

    До этого из всего класса Z читался один Z11.1.
    """
    v = FormalValidator()
    cases = {
        "Z00.1": VisitType.PROPHYLACTIC,
        "Z12.4": VisitType.PROPHYLACTIC,
        "Z13.9": VisitType.PROPHYLACTIC,
        "Z10.0": VisitType.PROPHYLACTIC,
        "Z34.0": VisitType.DISPENSARY,
        "Z35.5": VisitType.DISPENSARY,
    }
    for code, expected in cases.items():
        got = await v.get_visit_types(
            _visit(diagnoses=[{"КодМКБ": code}], services=_SILENT_SERVICE)
        )
        assert expected in got, (code, got)


async def test_z_code_does_not_add_a_type_over_a_service_verdict():
    """Услуга сказала «первичный» — Z-код поверх него тип не добавляет.

    Безусловным этот разбор был с 5b5c3f4 до 2026-09-29, и это была регрессия:
    на приёме педиатра с Z00.1 к PRIMARY добавлялся PROPHYLACTIC, шаблон
    обязательных полей выбирался профилактический, и 127 карт Алёнки теряли 125
    замечаний о незаполненных полях. Z-код говорит, зачем пришёл пациент; вид
    приёма называет услуга.
    """
    v = FormalValidator()
    for code in ("Z00.1", "Z12.4", "Z34.0"):
        got = await v.get_visit_types(
            _visit(
                diagnoses=[{"КодМКБ": code}],
                services=[{"Наименование": "Приём первичный"}],
            )
        )
        assert got == {VisitType.PRIMARY}, (code, got)


async def test_z11_1_does_not_also_match_the_z11_prefix():
    """У Z11.1 свой вердикт, и запасной разбор не должен добавить ей PROPHYLACTIC.

    Ловушка: Z11.1 лежит под префиксом Z11, который означает обычный скрининг.
    """
    got = await FormalValidator().get_visit_types(
        _visit(diagnoses=[{"КодМКБ": "Z11.1"}], services=_SILENT_SERVICE)
    )
    assert got == {VisitType.PROPHYLACTIC_TUBERCULIN}


async def test_z11_1_stays_unconditional_over_a_service_verdict():
    """Исключение — Z11.1: правила 190н критичные, а проба Манту бывает услугой.

    Закодированная как исследование, она дала бы LAB_RESEARCH_INTERVENTION, и
    условный разбор выключил бы проверку самой туберкулинодиагностики.
    """
    got = await FormalValidator().get_visit_types(
        _visit(
            diagnoses=[{"КодМКБ": "Z11.1"}],
            services=[{"КодЕГИСЗ": "A12.26.002", "Наименование": "Проба с туберкулином"}],
        )
    )
    assert VisitType.PROPHYLACTIC_TUBERCULIN in got
    assert VisitType.LAB_RESEARCH_INTERVENTION in got


async def test_tuberculin_screening_is_not_a_general_prophylactic_examination():
    """Z11.1 — свой порядок 190н; 211н п. 3 выводит его из детских профосмотров.

    Иначе на туберкулинодиагностику сели бы правила 404н про объём ПМО.
    """
    got = await FormalValidator().get_visit_types(_visit(diagnoses=[{"КодМКБ": "Z11.1"}]))
    assert VisitType.PROPHYLACTIC_TUBERCULIN in got
    assert VisitType.PROPHYLACTIC not in got
    # соседи по рубрике — обычный скрининг, но уже запасным разбором
    got = await FormalValidator().get_visit_types(
        _visit(diagnoses=[{"КодМКБ": "Z11.8"}], services=_SILENT_SERVICE)
    )
    assert VisitType.PROPHYLACTIC in got
    assert VisitType.PROPHYLACTIC_TUBERCULIN not in got


async def test_negative_z_codes_add_nothing():
    """Z02, Z08/Z09, Z37/Z38 означали бы «приём НЕ такой-то» — отменять нечем.

    Шаг по диагнозам умеет только добавлять тип. Требований к полям под эти
    коды в нормативке тоже не нашлось.
    """
    v = FormalValidator()
    for code in ("Z02.0", "Z08.1", "Z09.9", "Z37.0", "Z38.0"):
        got = await v.get_visit_types(
            _visit(diagnoses=[{"КодМКБ": code}], services=[{"Наименование": "нечто"}])
        )
        assert got == {VisitType.OTHER}, (code, got)


async def test_z95_implant_codes_are_not_a_visit_type():
    """Z95.x — импланты, повод обращения ими не задан.

    В перечнях 168н они есть как диагнозы для наблюдения, но вид приёма
    определяет не диагноз, а услуга.
    """
    got = await FormalValidator().get_visit_types(
        _visit(diagnoses=[{"КодМКБ": "Z95.1"}],
               services=[{"Код": "B01.047.002", "Наименование": "Прием терапевта повторный"}])
    )
    assert got == {VisitType.REPEAT}


async def test_dispensary_visit_is_recognised_by_the_code_the_order_names():
    """Приказ сам называет услугу диспансерным приёмом — наименованию клиники не верим.

    B04.015.003 «Диспансерный прием (осмотр, консультация) врача-кардиолога»
    таблица окончаний не ловит: .003 не входит в пару .001/.002. До явного
    списка такой код держался на том, что клиника напишет слово «диспансерный»
    в своём наименовании; если она назовёт приём обычным, вид визита терялся.
    """
    got = await FormalValidator().get_visit_types(
        _visit(services=[{"Код": "B04.015.003", "Наименование": "Приём кардиолога"}])
    )
    assert got == {VisitType.DISPENSARY}


async def test_prophylactic_visit_is_recognised_the_same_way():
    got = await FormalValidator().get_visit_types(
        _visit(services=[{"Код": "B04.015.004", "Наименование": "Приём детского кардиолога"}])
    )
    assert got == {VisitType.PROPHYLACTIC}


def test_every_code_the_order_names_is_covered():
    """Список берётся из выгрузки 804н, а не пишется руками."""
    import audit.formal_structure.validator as v

    named = v._codes_named_in_the_order()
    assert len(named) > 80, len(named)
    assert named["B04.047.001"] is VisitType.DISPENSARY
    assert named["B04.047.002"] is VisitType.PROPHYLACTIC
    # школы и консультирование так не называются и в список не попадают
    assert "B04.070.002" not in named


def test_form_025u_rules_are_not_scoped_to_children_only():
    """Правило на форме 025/у не может применяться только к детям.

    274н прил. 2 п. 1 адресует форму 025/у ВЗРОСЛОМУ населению. Правило с
    source=274n и age_group=child ссылается на источник, который к его
    собственной аудитории не относится, и при этом молчит на всех, к кому
    источник относится. Так было у diagnosis_required до 2026-09-09: критичная
    проверка «в записи должен быть диагноз» не запускалась ни на одном взрослом
    приёме — остаток времён единственного детского клиента.
    """
    import audit.formal_structure.validator as v

    child_only = [
        rule["rule_id"]
        for rule in v._RULES
        if rule.get("source") == "274n"
        and rule["applies_to"].get("age_group") == "child"
    ]
    assert not child_only, f"правила на 025/у заперты на детях: {child_only}"


def test_diagnosis_required_covers_adults():
    import audit.formal_structure.validator as v

    rule = next(r for r in v._RULES if r["rule_id"] == "diagnosis_required")
    assert rule["applies_to"]["age_group"] == "all"


def test_repeat_visit_rules_stay_inside_the_dynamics_section():
    """На повторном приёме нельзя требовать того, чего нет в разделе формы.

    Форма 025/у делит запись надвое. «Записи врачей-специалистов» (первичный
    приём) содержат строки «Объективные данные» и «Диагноз основного
    заболевания: код по МКБ»; «Медицинское наблюдение в динамике» (повторный) —
    только Дату, Жалобы, Данные наблюдения в динамике, Назначения,
    Лекарственные препараты, Листок нетрудоспособности, Льготные рецепты и
    Врача. Приложение 2 п. 14 требований сверх строк не добавляет.

    До 2026-09-09 два критичных правила требовали на повторном приёме
    объективный осмотр и диагноз, ссылаясь на 274н.
    """
    import audit.formal_structure.validator as v

    forbidden = {"objective_exam", "diagnosis", "anamnesis"}
    offenders = []
    for rule in v._RULES:
        if rule.get("source") != "274n":
            continue
        types = rule["applies_to"]["visit_types"]
        if "repeat" not in types and "all" not in types:
            continue
        overreach = forbidden & set(rule.get("targets") or [])
        if overreach:
            offenders.append((rule["rule_id"], sorted(overreach)))

    assert not offenders, (
        "правила на 274н требуют на повторном приёме строк, которых нет "
        f"в разделе «Медицинское наблюдение в динамике»: {offenders}"
    )
