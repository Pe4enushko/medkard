"""Слепок правила в формальном замечании.

Замечание несёт поля правила, которые берутся из rules.json один в один — без
разбора строк. Дословная цитата (`cite`) и разложение `source_ref` на документ и
раздел сюда НЕ входят: они требуют вычитки выжимок и приедут отдельной задачей.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.formal_structure.validator import _rule_snapshot, SNAPSHOT_FIELDS
from reporting.result_parser import parse_formal
from storage.models.result import FormalFinding

_RULES = json.loads(
    (ROOT / "src" / "audit" / "formal_structure" / "rules.json").read_text(encoding="utf-8")
)["rules"]


def test_snapshot_copies_rule_fields_verbatim():
    """Каждое поле слепка — значение из правила без преобразований."""
    for rule in _RULES:
        snap = _rule_snapshot(rule)
        assert set(snap) == set(SNAPSHOT_FIELDS)
        for key in SNAPSHOT_FIELDS:
            assert snap[key] == rule.get(key, ""), f"{rule['rule_id']}.{key}"


def test_snapshot_of_every_rule_is_non_empty():
    """У всех правил каталога поля слепка заполнены — пустых слепков быть не должно.

    Исключение только у `source`: у `has_typos` источника нет по решению реестра
    («внутренний стандарт качества»), и пустая строка там — верное значение.
    """
    for rule in _RULES:
        snap = _rule_snapshot(rule)
        for key in SNAPSHOT_FIELDS:
            if key == "source" and rule["rule_id"] == "has_typos":
                continue
            assert snap[key], f"{rule['rule_id']}.{key} пусто"


def test_finding_without_rule_keeps_empty_snapshot():
    """Синтетические замечания (незаполненные поля шаблона, противоречие НМУ)
    правила не имеют — слепок остаётся пустым, а не выдуманным."""
    f = FormalFinding(flag="НЕЗАПОЛНЕНЫ_ПОЛЯ_ШАБЛОНА", issue="Поля осмотра не заполнены: X")
    d = f.to_dict()
    for key in SNAPSHOT_FIELDS:
        assert d[key] == ""


def test_to_dict_roundtrips_through_parse_formal():
    """Слепок переживает сериализацию в jsonb и чтение обратно."""
    rule = next(r for r in _RULES if r["rule_id"] == "visit_meta_required")
    f = FormalFinding(flag=rule["flag_code"], issue="что-то не так", **_rule_snapshot(rule))
    restored = parse_formal([f.to_dict()]).findings[0]
    assert restored == f
