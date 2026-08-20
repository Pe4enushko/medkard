import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.formal_structure.validator import FormalValidator, VisitType


def _rule(**over):
    """Минимальное правило; поля перекрываются через kwargs."""
    base = {
        "rule_id": "r",
        "flag_code": "FLAG",
        "rule_type": "required_field",
        "applies_to": {"visit_types": ["all"], "specialties": ["all"], "age_group": "all"},
        "expectation": "ожидание",
    }
    base.update(over)
    return base


def _flags(rules):
    return [r["flag_code"] for r in rules]


def test_icd_prefix_matches(monkeypatch):
    import audit.formal_structure.validator as v

    rule = _rule(
        rule_id="dispensary",
        flag_code="ДИСПАНСЕРНОЕ",
        applies_to={
            "visit_types": ["primary"],
            "specialties": ["all"],
            "age_group": "adult",
            "icd_prefixes": ["I10", "I11"],
        },
    )
    monkeypatch.setattr(v, "_RULES", [rule])

    got = FormalValidator().get_rules({VisitType.PRIMARY}, 40, ["I11.9"])
    assert _flags(got) == ["ДИСПАНСЕРНОЕ"]


def test_icd_prefix_does_not_match(monkeypatch):
    import audit.formal_structure.validator as v

    rule = _rule(
        flag_code="ДИСПАНСЕРНОЕ",
        applies_to={
            "visit_types": ["primary"],
            "specialties": ["all"],
            "age_group": "adult",
            "icd_prefixes": ["I10", "I11"],
        },
    )
    monkeypatch.setattr(v, "_RULES", [rule])

    assert FormalValidator().get_rules({VisitType.PRIMARY}, 40, ["J06.9"]) == []


def test_icd_prefix_without_codes_is_skipped(monkeypatch):
    import audit.formal_structure.validator as v

    rule = _rule(
        flag_code="ДИСПАНСЕРНОЕ",
        applies_to={
            "visit_types": ["primary"],
            "specialties": ["all"],
            "age_group": "all",
            "icd_prefixes": ["I10"],
        },
    )
    monkeypatch.setattr(v, "_RULES", [rule])

    assert FormalValidator().get_rules({VisitType.PRIMARY}, 40, None) == []
    assert FormalValidator().get_rules({VisitType.PRIMARY}, 40, []) == []


def test_icd_prefix_is_case_and_space_insensitive(monkeypatch):
    import audit.formal_structure.validator as v

    rule = _rule(
        flag_code="ДИСПАНСЕРНОЕ",
        applies_to={
            "visit_types": ["primary"],
            "specialties": ["all"],
            "age_group": "all",
            "icd_prefixes": ["I11"],
        },
    )
    monkeypatch.setattr(v, "_RULES", [rule])

    assert _flags(FormalValidator().get_rules({VisitType.PRIMARY}, 40, [" i11.9 "])) == ["ДИСПАНСЕРНОЕ"]


def test_rule_without_icd_prefixes_ignores_codes(monkeypatch):
    import audit.formal_structure.validator as v

    monkeypatch.setattr(v, "_RULES", [_rule(flag_code="ОБЫЧНОЕ")])

    assert _flags(FormalValidator().get_rules({VisitType.PRIMARY}, 40, None)) == ["ОБЫЧНОЕ"]
    assert _flags(FormalValidator().get_rules({VisitType.PRIMARY}, 40, ["J06.9"])) == ["ОБЫЧНОЕ"]


def test_adult_prophylactic_rules_do_not_leak_to_children():
    """404н-правила — только взрослым, 211н-исключения — только детям."""
    v = FormalValidator()

    adult = _flags(v.get_rules({VisitType.PROPHYLACTIC}, 45, ["Z00.0"]))
    child = _flags(v.get_rules({VisitType.PROPHYLACTIC}, 10, ["Z00.1"]))

    assert "ПРОФ_ВЗРОСЛЫЙ_НЕПОЛНЫЙ_ОБЪЁМ" in adult
    assert "ПРОФ_ВЗРОСЛЫЙ_НЕТ_ГРУППЫ_ЗДОРОВЬЯ" in adult
    assert "ПРОФ_ВЗРОСЛЫЙ_НЕПОЛНЫЙ_ОБЪЁМ" not in child

    assert "ДОПУСТИМО_БЕЗ_ЖАЛОБ_ПРОФИЛАКТИКА" in child
    assert "ДОПУСТИМО_БЕЗ_ЖАЛОБ_ПРОФИЛАКТИКА" not in adult
