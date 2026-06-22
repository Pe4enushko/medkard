import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.filters import AnalysisFilter, CardFilter, IcdFilter, KDLFilter
from parsers.filter_config import load_card_filter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _visit(diagnoses=None, services=None):
    return {
        "Прием": {"GUID": "test-guid"},
        "Диагнозы": diagnoses or [],
        "Услуги": services or [],
    }


def _dx(code):
    return {"КодМКБ": code}


def _svc(name="", артикул="", egisz="", uid=""):
    return {"Наименование": name, "Артикул": артикул, "КодЕГИСЗ": egisz, "УИДЕГИСЗ": uid}


# ── IcdFilter ─────────────────────────────────────────────────────────────────

class TestIcdFilter:
    def test_skip_when_all_diagnoses_match(self):
        f = IcdFilter(["Z00.0", "J06.9"])
        visit = _visit(diagnoses=[_dx("Z00.0"), _dx("J06.9")])
        assert f.should_skip(visit) is True

    def test_no_skip_when_one_diagnosis_outside(self):
        f = IcdFilter(["Z00.0"])
        visit = _visit(diagnoses=[_dx("Z00.0"), _dx("A01.0")])
        assert f.should_skip(visit) is False

    def test_no_skip_when_codes_empty(self):
        f = IcdFilter([])
        visit = _visit(diagnoses=[_dx("Z00.0")])
        assert f.should_skip(visit) is False

    def test_no_skip_when_no_diagnoses(self):
        f = IcdFilter(["Z00.0"])
        assert f.should_skip(_visit(diagnoses=[])) is False

    def test_case_insensitive(self):
        f = IcdFilter(["z00.0"])
        visit = _visit(diagnoses=[_dx("Z00.0")])
        assert f.should_skip(visit) is True

    def test_skip_diagnosis_matching_code(self):
        f = IcdFilter(["Z00.0"])
        assert f.skip_diagnosis(_dx("Z00.0")) is True

    def test_skip_diagnosis_non_matching_code(self):
        f = IcdFilter(["Z00.0"])
        assert f.skip_diagnosis(_dx("J06.9")) is False

    def test_skip_diagnosis_case_insensitive(self):
        f = IcdFilter(["z00.0"])
        assert f.skip_diagnosis(_dx("Z00.0")) is True

    def test_skip_diagnosis_empty_codes(self):
        f = IcdFilter([])
        assert f.skip_diagnosis(_dx("Z00.0")) is False


# ── KDLFilter ─────────────────────────────────────────────────────────────────

class TestKDLFilter:
    def test_skip_when_all_services_have_kdl_suffix(self):
        f = KDLFilter()
        visit = _visit(services=[
            _svc(name="Анализ крови (КДЛ)"),
            _svc(name="Биохимия (КДЛ)"),
        ])
        assert f.should_skip(visit) is True

    def test_skip_when_all_services_have_matching_code(self):
        f = KDLFilter()
        visit = _visit(services=[
            _svc(артикул="6.2.A5.101"),
            _svc(артикул="1.3.B7.009"),
        ])
        assert f.should_skip(visit) is True

    def test_skip_mixed_conditions_all_match(self):
        f = KDLFilter()
        visit = _visit(services=[
            _svc(name="Анализ (КДЛ)"),          # matched via name
            _svc(артикул="6.2.A5.101"),         # matched via code
        ])
        assert f.should_skip(visit) is True

    def test_no_skip_when_one_service_unmatched(self):
        f = KDLFilter()
        visit = _visit(services=[
            _svc(name="Анализ (КДЛ)"),
            _svc(name="Прием врача"),            # neither suffix nor code
        ])
        assert f.should_skip(visit) is False

    def test_no_skip_name_not_suffix(self):
        f = KDLFilter()
        # (КДЛ) in the middle — not a suffix
        visit = _visit(services=[_svc(name="(КДЛ) результаты")])
        assert f.should_skip(visit) is False

    def test_no_skip_empty_services(self):
        f = KDLFilter()
        assert f.should_skip(_visit(services=[])) is False

    def test_skip_via_egisz_field(self):
        f = KDLFilter()
        visit = _visit(services=[_svc(egisz="6.2.A5.101")])
        assert f.should_skip(visit) is True

    def test_skip_via_uid_field(self):
        f = KDLFilter()
        visit = _visit(services=[_svc(uid="6.2.A5.101")])
        assert f.should_skip(visit) is True

    def test_skip_diagnosis_always_false(self):
        f = KDLFilter()
        assert f.skip_diagnosis(_dx("Z00.0")) is False


# ── AnalysisFilter ────────────────────────────────────────────────────────────

class TestAnalysisFilter:
    def test_skip_when_all_services_have_a_prefix_code(self):
        f = AnalysisFilter()
        visit = _visit(services=[
            _svc(артикул="A04.16.001"),
            _svc(артикул="A04.28.002"),
        ])
        assert f.should_skip(visit) is True

    def test_no_skip_when_one_service_has_b_prefix(self):
        f = AnalysisFilter()
        visit = _visit(services=[
            _svc(артикул="A04.16.001"),
            _svc(артикул="B01.023.001"),
        ])
        assert f.should_skip(visit) is False

    def test_no_skip_empty_services(self):
        f = AnalysisFilter()
        assert f.should_skip(_visit(services=[])) is False

    def test_skip_via_egisz_field(self):
        f = AnalysisFilter()
        visit = _visit(services=[_svc(egisz="A04.16.001")])
        assert f.should_skip(visit) is True

    def test_skip_via_uid_field(self):
        f = AnalysisFilter()
        visit = _visit(services=[_svc(uid="A04.16.001")])
        assert f.should_skip(visit) is True

    def test_skip_with_cyrillic_a(self):
        f = AnalysisFilter()
        visit = _visit(services=[_svc(артикул="А04.16.001")])  # Cyrillic А
        assert f.should_skip(visit) is True

    def test_no_skip_with_c_prefix(self):
        f = AnalysisFilter()
        visit = _visit(services=[_svc(артикул="C04.16.001")])
        assert f.should_skip(visit) is False

    def test_skip_diagnosis_always_false(self):
        f = AnalysisFilter()
        assert f.skip_diagnosis(_dx("Z00.0")) is False


# ── CardFilter ────────────────────────────────────────────────────────────────

class TestCardFilter:
    def test_skip_by_any_strategy(self):
        cf = CardFilter([IcdFilter(["Z00.0"]), KDLFilter()])
        visit = _visit(diagnoses=[_dx("Z00.0")])
        pending, ignored, done, strat = cf.filter([visit], set())
        assert len(pending) == 0
        assert len(ignored) == 1
        assert strat == 1

    def test_pending_when_no_strategy_matches(self):
        cf = CardFilter([IcdFilter(["Z00.0"])])
        visit = _visit(diagnoses=[_dx("A01.0")])
        pending, ignored, done, strat = cf.filter([visit], set())
        assert len(pending) == 1
        assert strat == 0

    def test_guid_dedup(self):
        cf = CardFilter([])
        visit = _visit()
        visit["Прием"]["GUID"] = "abc-123"
        pending, ignored, done, strat = cf.filter([visit], {"abc-123"})
        assert len(pending) == 0
        assert done == 1

    def test_should_skip_diagnosis_delegates_to_strategies(self):
        cf = CardFilter([IcdFilter(["Z00.0"])])
        assert cf.should_skip_diagnosis(_dx("Z00.0")) is True
        assert cf.should_skip_diagnosis(_dx("A01.0")) is False

    def test_empty_filter_passes_everything(self):
        cf = CardFilter([])
        visits = [_visit(diagnoses=[_dx("Z00.0")]), _visit(diagnoses=[_dx("A01.0")])]
        pending, ignored, done, strat = cf.filter(visits, set())
        assert len(pending) == 2
        assert strat == 0


# ── load_card_filter ──────────────────────────────────────────────────────────

class TestLoadCardFilter:
    def test_loads_alenka(self):
        cf = load_card_filter("Alenka")
        assert len(cf.strategies) == 3
        assert isinstance(cf.strategies[0], IcdFilter)
        assert isinstance(cf.strategies[1], KDLFilter)
        assert isinstance(cf.strategies[2], AnalysisFilter)

    def test_loads_mds(self):
        cf = load_card_filter("MDS")
        assert len(cf.strategies) == 3

    def test_unknown_org_returns_empty_filter(self):
        cf = load_card_filter("Unknown")
        assert len(cf.strategies) == 0
