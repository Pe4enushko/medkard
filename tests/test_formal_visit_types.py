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


async def test_b_code_classification_unchanged():
    """Фикс среднего сегмента не задевает разбор B-кодов."""
    v = FormalValidator()

    primary = await v.get_visit_types(_visit(services=[{"Наименование": "x", "Код": "B01.070.001"}]))
    assert VisitType.PRIMARY in primary

    repeat = await v.get_visit_types(_visit(services=[{"Наименование": "x", "Код": "B01.070.011"}]))
    assert VisitType.REPEAT in repeat

    prof = await v.get_visit_types(_visit(services=[{"Наименование": "x", "Код": "B04.031.002"}]))
    assert VisitType.PROPHYLACTIC in prof

    # B01 с middle != 070 по действующей ветке — намеренно OTHER
    other = await v.get_visit_types(_visit(services=[{"Наименование": "x", "Код": "B01.058.001"}]))
    assert VisitType.OTHER in other
