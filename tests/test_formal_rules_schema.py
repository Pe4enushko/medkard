import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_RULES_PATH = ROOT / "src" / "audit" / "formal_structure" / "rules.json"

_ISO = __import__("re").compile(r"^\d{4}-\d{2}-\d{2}$")


def _doc() -> dict:
    return json.loads(_RULES_PATH.read_text(encoding="utf-8"))


def test_file_is_object_with_meta():
    doc = _doc()
    assert isinstance(doc, dict), "rules.json должен быть объектом, а не списком"
    assert _ISO.match(doc["revised_at"]), f"revised_at не ISO: {doc['revised_at']!r}"
    assert doc["sources_doc"] == "docs/formal-rules-sources.md"
    assert isinstance(doc["rules"], list) and doc["rules"]


_SEVERITIES = {"критичный", "значительный", "незначительный"}
_AGE_GROUPS = {"all", "child", "adult"}
_VISIT_TYPES = {
    "all", "primary", "repeat", "prophylactic",
    "prophylactic_tuberculin", "lab_research_intervention", "other",
}

# Снимок: обновляется задачами, которые добавляют правила.
EXPECTED_RULE_COUNT = 42

# Флаги, сознательно разделённые между child/adult-парой правил.
_SHARED_FLAGS = {"ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО"}


def test_every_rule_has_source_ref_and_verified_at():
    doc = _doc()
    revised = doc["revised_at"]
    for r in doc["rules"]:
        rid = r["rule_id"]
        assert r.get("source_ref", "").strip(), f"{rid}: пустой source_ref"
        va = r.get("verified_at", "")
        assert _ISO.match(va), f"{rid}: verified_at не ISO: {va!r}"
        assert va <= revised, f"{rid}: verified_at {va} позже revised_at {revised}"


def test_rule_ids_are_unique():
    ids = [r["rule_id"] for r in _doc()["rules"]]
    assert len(ids) == len(set(ids)), f"дубли rule_id: {sorted({i for i in ids if ids.count(i) > 1})}"


def test_flag_codes_are_unique_except_shared_pairs():
    flags = [r["flag_code"] for r in _doc()["rules"]]
    dupes = {f for f in flags if flags.count(f) > 1}
    assert dupes <= _SHARED_FLAGS, f"неожиданные дубли flag_code: {sorted(dupes - _SHARED_FLAGS)}"


def test_enums_are_valid():
    for r in _doc()["rules"]:
        rid = r["rule_id"]
        assert r["severity"] in _SEVERITIES, f"{rid}: severity {r['severity']!r}"
        applies = r["applies_to"]
        assert applies.get("age_group", "all") in _AGE_GROUPS, f"{rid}: age_group"
        assert set(applies["visit_types"]) <= _VISIT_TYPES, f"{rid}: visit_types {applies['visit_types']}"


def test_icd_prefixes_are_upper_nonempty_strings():
    for r in _doc()["rules"]:
        prefixes = r["applies_to"].get("icd_prefixes")
        if prefixes is None:
            continue
        assert isinstance(prefixes, list) and prefixes, f"{r['rule_id']}: пустой icd_prefixes"
        for p in prefixes:
            assert isinstance(p, str) and p.strip(), f"{r['rule_id']}: пустой префикс"
            assert p == p.upper().strip(), f"{r['rule_id']}: префикс {p!r} не в верхнем регистре"


def test_rule_count_snapshot():
    assert len(_doc()["rules"]) == EXPECTED_RULE_COUNT


def test_no_rule_references_retired_203n():
    """203н-2017 утратил силу; ярлык source больше не используется (спека §4.6)."""
    offenders = [r["rule_id"] for r in _doc()["rules"] if r.get("source") == "203n"]
    assert not offenders, f"правила всё ещё на ярлыке 203n: {offenders}"
