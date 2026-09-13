"""scripts/hacks/backfill-rule-snapshot.py — the pure pass over findings, not the DB."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "backfill_rule_snapshot",
    Path(__file__).resolve().parent.parent / "scripts" / "hacks" / "backfill-rule-snapshot.py")
backfill = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = backfill
_spec.loader.exec_module(backfill)

RULES = [
    {"rule_id": "visit_meta_required", "flag_code": "ОТСУТСТВУЮТ_МЕТАДАННЫЕ_ВИЗИТА", "source": "274n",
     "severity": "критичный", "source_ref": "приказ 274н", "expectation": "Дата, возраст, пол",
     "verified_at": "2026-08-19", "applies_to": {"age_group": "all"}},
    {"rule_id": "dispensary_followup_adult", "flag_code": "ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО",
     "source": "168n", "severity": "значимый", "source_ref": "приказ 168н", "expectation": "ДН взрослых",
     "verified_at": "2026-08-19", "applies_to": {"age_group": "adult"}},
    {"rule_id": "dispensary_followup_child", "flag_code": "ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО",
     "source": "192n", "severity": "значимый", "source_ref": "приказ 192н", "expectation": "ДН детей",
     "verified_at": "2026-08-19", "applies_to": {"age_group": "child"}},
]

META = {"flag": "ОТСУТСТВУЮТ_МЕТАДАННЫЕ_ВИЗИТА", "issue": "нет даты", "source": "274n", "comment": ""}
DISP = {"flag": "ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО", "issue": "нет ДН", "source": "168n", "comment": ""}


def test_finding_gets_the_rule_snapshot():
    findings, counts = backfill.snapshot([META], {"AGE": 40}, RULES)
    assert findings[0]["rule_id"] == "visit_meta_required"
    assert findings[0]["expectation"] == "Дата, возраст, пол"
    assert findings[0]["verified_at"] == "2026-08-19"
    assert counts["snapshotted"] == 1


def test_finding_keeps_its_own_issue_and_comment():
    findings, _ = backfill.snapshot([{**META, "comment": "врач дописал"}], {"AGE": 40}, RULES)
    assert findings[0]["issue"] == "нет даты"
    assert findings[0]["comment"] == "врач дописал"


def test_shared_flag_resolved_by_adult_age():
    findings, _ = backfill.snapshot([DISP], {"AGE": 40}, RULES)
    assert findings[0]["rule_id"] == "dispensary_followup_adult"


def test_shared_flag_resolved_by_child_age():
    findings, _ = backfill.snapshot([DISP], {"AGE": "1"}, RULES)
    assert findings[0]["rule_id"] == "dispensary_followup_child"


def test_shared_flag_without_age_is_left_alone():
    findings, counts = backfill.snapshot([DISP], {}, RULES)
    assert findings is None
    assert counts["ambiguous"] == 1


def test_unknown_flag_gets_the_empty_snapshot():
    synthetic = {"flag": "НЕЗАПОЛНЕНО_ПОЛЕ_ШАБЛОНА", "issue": "Жалобы", "source": "", "comment": ""}
    findings, counts = backfill.snapshot([synthetic], {"AGE": 40}, RULES)
    assert findings[0]["rule_id"] == ""
    assert findings[0]["expectation"] == ""
    assert counts["no_rule"] == 1


def test_finding_with_snapshot_is_not_touched():
    done = {**META, "rule_id": "visit_meta_required", "severity": "x", "source_ref": "y",
            "expectation": "old text", "verified_at": "2026-01-01"}
    findings, counts = backfill.snapshot([done], {"AGE": 40}, RULES)
    assert findings is None
    assert counts["already"] == 1


def test_empty_result_means_nothing_to_write():
    findings, counts = backfill.snapshot([], {"AGE": 40}, RULES)
    assert findings is None
    assert sum(counts.values()) == 0


class _FakeStorage:
    def __init__(self, rows):
        self.rows = rows
        self.written: dict[str, list] = {}

    async def list_formal_results_to_backfill(self, *, limit, after_id):
        rows = [r for r in self.rows if r["id"] > after_id]
        return rows[:limit]

    async def set_formal_result(self, *, card_id, formal_json):
        self.written[card_id] = json.loads(formal_json)
        return 1


def test_run_writes_changed_cards_only():
    rows = [
        {"id": "a", "formal_result": [META], "patient": {"AGE": 40}},
        {"id": "b", "formal_result": [DISP], "patient": {}},
    ]
    storage = _FakeStorage(rows)
    totals = asyncio.run(backfill._run(storage, rules=RULES, limit=0, batch=10, apply=True))
    assert totals["cards"] == 1
    assert list(storage.written) == ["a"]
    assert storage.written["a"][0]["rule_id"] == "visit_meta_required"


def test_run_dry_run_writes_nothing():
    storage = _FakeStorage([{"id": "a", "formal_result": [META], "patient": {"AGE": 40}}])
    totals = asyncio.run(backfill._run(storage, rules=RULES, limit=0, batch=10, apply=False))
    assert totals["cards"] == 1
    assert storage.written == {}
