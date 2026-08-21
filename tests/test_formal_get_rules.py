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


def test_dispensary_adult_requires_matching_icd():
    """Взрослое ДН-правило включается только на кодах из перечня 168н."""
    v = FormalValidator()

    assert "ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО" in _flags(
        v.get_rules({VisitType.PRIMARY}, 55, ["I11.9"])
    )
    assert "ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО" not in _flags(
        v.get_rules({VisitType.PRIMARY}, 55, ["J06.9"])
    )
    assert "ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО" not in _flags(
        v.get_rules({VisitType.PRIMARY}, 55, [])
    )


def test_dispensary_child_rule_needs_no_icd():
    """Детское ДН-правило — без перечня кодов, но только детям."""
    v = FormalValidator()

    child = v.get_rules({VisitType.PRIMARY}, 10, ["J06.9"])
    assert "ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО" in _flags(child)
    assert [r["rule_id"] for r in child if r["flag_code"] == "ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО"] == [
        "dispensary_followup_child"
    ]

    adult = v.get_rules({VisitType.PRIMARY}, 55, ["I11.9"])
    assert [r["rule_id"] for r in adult if r["flag_code"] == "ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО"] == [
        "dispensary_followup_adult"
    ]


def test_format_rules_renders_new_rules():
    """_format_rules не падает на правилах с condition и без него."""
    v = FormalValidator()
    rules = v.get_rules({VisitType.PRIMARY}, 55, ["I11.9"])
    text = v._format_rules(rules)

    assert "(ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО)" in text
    assert len(text.splitlines()) == len(rules)


def test_flag_source_lookup_covers_new_flags():
    """_FLAG_SOURCE строится по всем правилам, включая новые."""
    import audit.formal_structure.validator as v

    assert v._FLAG_SOURCE["ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО"] in {"168n", "192n"}
    assert v._FLAG_SOURCE["ПРОФ_ВЗРОСЛЫЙ_НЕПОЛНЫЙ_ОБЪЁМ"] == "404n"
    assert v._FLAG_SOURCE["НАЗНАЧЕНИЕ_ПО_ТОРГОВОМУ_БЕЗ_МНН"] == "1094n"
