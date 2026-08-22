import sys
from pathlib import Path

import pytest

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


def _service_visit(code: str, name: str) -> dict:
    return {"Услуги": [{"КодЕГИСЗ": code, "Наименование": name}]}


def test_consent_rule_is_prefiltered_to_performed_intervention_services() -> None:
    validator = FormalValidator()

    ordinary_visit = validator.get_rules(
        {VisitType.PRIMARY},
        45,
        ["J18.9"],
        _service_visit("B01.047.001", "Приём врача-терапевта первичный"),
    )
    injection_visit = validator.get_rules(
        {VisitType.LAB_RESEARCH_INTERVENTION},
        45,
        ["Z25.1"],
        _service_visit("A11.02.002", "Внутримышечное введение лекарственного препарата"),
    )

    assert "consent_for_intervention_outside_list" not in {
        rule["rule_id"] for rule in ordinary_visit
    }
    assert "consent_for_intervention_outside_list" in {
        rule["rule_id"] for rule in injection_visit
    }


def test_a_code_rules_are_prefiltered_by_804n_service_type():
    validator = FormalValidator()
    visit_types = {VisitType.LAB_RESEARCH_INTERVENTION}

    laboratory = _flags(
        validator.get_rules(
            visit_types,
            45,
            ["Z00.0"],
            _service_visit("A09.05.023", "Исследование уровня глюкозы в крови"),
        )
    )
    surgery = _flags(
        validator.get_rules(
            visit_types,
            45,
            ["S51.8"],
            _service_visit("A16.01.004", "Хирургическая обработка раны"),
        )
    )

    assert "ЛАБОРАТОРИЯ_НЕПОЛНЫЕ_ДАННЫЕ" in laboratory
    assert "МАНИПУЛЯЦИЯ_НЕПОЛНОЕ_ОПИСАНИЕ" not in laboratory
    assert "МАНИПУЛЯЦИЯ_НЕПОЛНОЕ_ОПИСАНИЕ" in surgery
    assert "ЛАБОРАТОРИЯ_НЕПОЛНЫЕ_ДАННЫЕ" not in surgery


@pytest.mark.parametrize(
    ("code", "name", "expected"),
    [
        ("A04.16.001", "Ультразвуковое исследование органов брюшной полости", "УЗИ_НЕПОЛНЫЙ_ПРОТОКОЛ"),
        ("A05.10.006", "Регистрация электрокардиограммы", "ФУНКЦИОНАЛЬНОЕ_ИССЛЕДОВАНИЕ_НЕПОЛНОЕ"),
        ("A12.09.001", "Исследование неспровоцированных дыхательных объёмов", "ФУНКЦИОНАЛЬНОЕ_ИССЛЕДОВАНИЕ_НЕПОЛНОЕ"),
        ("A06.09.007", "Рентгенография лёгких", "РЕНТГЕН_НЕПОЛНЫЙ_ПРОТОКОЛ"),
    ],
)
def test_804n_research_type_selects_only_its_specific_rule(code, name, expected):
    validator = FormalValidator()
    selected = set(
        _flags(
            validator.get_rules(
                {VisitType.LAB_RESEARCH_INTERVENTION},
                45,
                ["Z00.0"],
                _service_visit(code, name),
            )
        )
    )
    research_flags = {
        "ЛАБОРАТОРИЯ_НЕПОЛНЫЕ_ДАННЫЕ",
        "УЗИ_НЕПОЛНЫЙ_ПРОТОКОЛ",
        "РЕНТГЕН_НЕПОЛНЫЙ_ПРОТОКОЛ",
        "ФУНКЦИОНАЛЬНОЕ_ИССЛЕДОВАНИЕ_НЕПОЛНОЕ",
        "МАНИПУЛЯЦИЯ_НЕПОЛНОЕ_ОПИСАНИЕ",
        "ИНЪЕКЦИЯ_НЕПОЛНОЕ_ОПИСАНИЕ",
    }

    assert selected & research_flags == {expected}


def test_injection_rule_uses_real_a11_code_and_service_name():
    validator = FormalValidator()
    visit_types = {VisitType.LAB_RESEARCH_INTERVENTION}

    injection = _flags(
        validator.get_rules(
            visit_types,
            45,
            ["J18.9"],
            _service_visit(
                "A11.02.002",
                "Внутримышечное введение лекарственных препаратов",
            ),
        )
    )
    sample_collection = _flags(
        validator.get_rules(
            visit_types,
            45,
            ["Z00.0"],
            _service_visit("A11.05.001", "Взятие крови из пальца"),
        )
    )
    obsolete_fake_code = _flags(
        validator.get_rules(
            visit_types,
            45,
            ["J18.9"],
            _service_visit("A03.31.001", "Процедура введения препарата"),
        )
    )

    assert "ИНЪЕКЦИЯ_НЕПОЛНОЕ_ОПИСАНИЕ" in injection
    assert "МАНИПУЛЯЦИЯ_НЕПОЛНОЕ_ОПИСАНИЕ" not in injection
    assert "ИНЪЕКЦИЯ_НЕПОЛНОЕ_ОПИСАНИЕ" not in sample_collection
    assert "ИНЪЕКЦИЯ_НЕПОЛНОЕ_ОПИСАНИЕ" not in obsolete_fake_code


def test_unknown_age_keeps_only_age_agnostic_rules(monkeypatch):
    """Нечитаемый возраст сужает набор правил, а не расширяет его.

    Раньше patient_age=None выключал возрастной фильтр целиком, и на карте без
    пригодного AGE детское правило уезжало взрослому пациенту вместе со
    взрослыми правилами.
    """
    import audit.formal_structure.validator as v

    rules = [
        _rule(rule_id="any", flag_code="ЛЮБОЙ_ВОЗРАСТ"),
        _rule(
            rule_id="child",
            flag_code="ТОЛЬКО_ДЕТИ",
            applies_to={"visit_types": ["all"], "specialties": ["all"], "age_group": "child"},
        ),
        _rule(
            rule_id="adult",
            flag_code="ТОЛЬКО_ВЗРОСЛЫЕ",
            applies_to={"visit_types": ["all"], "specialties": ["all"], "age_group": "adult"},
        ),
    ]
    monkeypatch.setattr(v, "_RULES", rules)
    validator = FormalValidator()

    assert _flags(validator.get_rules({VisitType.PRIMARY}, None, None)) == ["ЛЮБОЙ_ВОЗРАСТ"]
    assert _flags(validator.get_rules({VisitType.PRIMARY}, 8, None)) == [
        "ЛЮБОЙ_ВОЗРАСТ",
        "ТОЛЬКО_ДЕТИ",
    ]
    assert _flags(validator.get_rules({VisitType.PRIMARY}, 40, None)) == [
        "ЛЮБОЙ_ВОЗРАСТ",
        "ТОЛЬКО_ВЗРОСЛЫЕ",
    ]


def test_infant_is_a_child_not_an_unknown_age(monkeypatch):
    """AGE=0 — ребёнок до года, а не отсутствие данных."""
    import audit.formal_structure.validator as v

    rules = [
        _rule(
            rule_id="child",
            flag_code="ТОЛЬКО_ДЕТИ",
            applies_to={"visit_types": ["all"], "specialties": ["all"], "age_group": "child"},
        ),
    ]
    monkeypatch.setattr(v, "_RULES", rules)

    assert _flags(FormalValidator().get_rules({VisitType.PRIMARY}, 0, None)) == ["ТОЛЬКО_ДЕТИ"]


def test_eighteen_is_an_adult(monkeypatch):
    """Граница 404н: «граждане в возрасте 18 лет и старше»."""
    import audit.formal_structure.validator as v

    rules = [
        _rule(
            rule_id="child",
            flag_code="ТОЛЬКО_ДЕТИ",
            applies_to={"visit_types": ["all"], "specialties": ["all"], "age_group": "child"},
        ),
        _rule(
            rule_id="adult",
            flag_code="ТОЛЬКО_ВЗРОСЛЫЕ",
            applies_to={"visit_types": ["all"], "specialties": ["all"], "age_group": "adult"},
        ),
    ]
    monkeypatch.setattr(v, "_RULES", rules)
    validator = FormalValidator()

    assert _flags(validator.get_rules({VisitType.PRIMARY}, 17, None)) == ["ТОЛЬКО_ДЕТИ"]
    assert _flags(validator.get_rules({VisitType.PRIMARY}, 18, None)) == ["ТОЛЬКО_ВЗРОСЛЫЕ"]
