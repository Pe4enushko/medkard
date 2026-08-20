import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.formal_structure.validator import FormalValidator, VisitType


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
