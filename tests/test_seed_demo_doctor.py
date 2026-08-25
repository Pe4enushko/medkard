"""Unit tests for scripts/seed-demo-doctor.py — the demo-doctor stamper.

Pure functions only: no DB. The storage method the script leans on
(list_audited_by_visit_date) is covered by tests/test_seed_demo_doctor_storage.py.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "seed_demo_doctor",
    Path(__file__).resolve().parent.parent / "scripts" / "seed-demo-doctor.py")
seed = importlib.util.module_from_spec(_spec)
# Registered before exec: @dataclass resolves annotations through
# sys.modules[cls.__module__], which is None for a module loaded by spec alone.
sys.modules[_spec.name] = seed
_spec.loader.exec_module(seed)


def _row(guid: str, *, code: str | None = None, formal: int = 0, diag: int = 0, icd: int = 0):
    return {"card_guid": guid, "doctor_code": code,
            "formal_n": formal, "diag_n": diag, "icd_n": icd}


# ── --code validation ────────────────────────────────────────────────────────

def test_code_accepts_digits_letters_dash_underscore():
    assert seed._validate_code("90001") == "90001"
    assert seed._validate_code("demo_doc-1") == "demo_doc-1"


def test_code_rejects_empty():
    with pytest.raises(SystemExit):
        seed._validate_code("")


def test_code_rejects_cyrillic():
    # Python's \w matches Cyrillic, so the API's own [\w-] pattern would let
    # this through — we mint the code ourselves, so keep it ASCII.
    with pytest.raises(SystemExit):
        seed._validate_code("врач1")


def test_code_rejects_space():
    with pytest.raises(SystemExit):
        seed._validate_code("90 01")


def test_code_rejects_longer_than_64():
    with pytest.raises(SystemExit):
        seed._validate_code("9" * 65)


# ── --date parsing ───────────────────────────────────────────────────────────

def test_parse_date_accepts_iso():
    assert seed._parse_date("2026-08-20").isoformat() == "2026-08-20"


def test_parse_date_rejects_1c_format():
    with pytest.raises(SystemExit):
        seed._parse_date("20.08.2026")


# ── candidate ranking and top-up ─────────────────────────────────────────────

def test_plan_prefers_cards_with_more_findings():
    rows = [_row("a", formal=0), _row("b", diag=3), _row("c", formal=1, icd=1)]
    plan = seed._plan(rows, code="90001", limit=2)
    assert [r["card_guid"] for r in plan.to_stamp] == ["b", "c"]


def test_plan_ranks_ties_by_guid_for_a_stable_rerun():
    rows = [_row("b", formal=1), _row("a", formal=1)]
    plan = seed._plan(rows, code="90001", limit=1)
    assert [r["card_guid"] for r in plan.to_stamp] == ["a"]


def test_plan_tops_up_to_limit_counting_cards_already_ours():
    rows = [_row("a", code="90001"), _row("b", diag=2), _row("c", diag=1)]
    plan = seed._plan(rows, code="90001", limit=2)
    assert [r["card_guid"] for r in plan.mine] == ["a"]
    assert [r["card_guid"] for r in plan.to_stamp] == ["b"]


def test_plan_stamps_nothing_when_limit_already_reached():
    rows = [_row("a", code="90001"), _row("b", code="90001"), _row("c", diag=5)]
    plan = seed._plan(rows, code="90001", limit=2)
    assert plan.to_stamp == []


def test_plan_never_takes_a_card_stamped_for_another_doctor():
    rows = [_row("a", code="00012", diag=9), _row("b")]
    plan = seed._plan(rows, code="90001", limit=5)
    assert [r["card_guid"] for r in plan.to_stamp] == ["b"]
    assert [r["card_guid"] for r in plan.foreign] == ["a"]


def test_plan_treats_empty_string_code_as_unstamped():
    # 1C sends the key with an empty value on cards it has no doctor for.
    plan = seed._plan([_row("a", code="")], code="90001", limit=1)
    assert [r["card_guid"] for r in plan.to_stamp] == ["a"]


# ── the Прием block edit ─────────────────────────────────────────────────────

def test_stamp_sets_both_keys_and_keeps_the_rest():
    priem = {"GUID": "ab-1", "DATE": "20.08.2026", "Num": "000048874"}
    stamped = seed._stamp(priem, name="Панкратов Эдуард Рашитович", code="90001")
    assert stamped == {"GUID": "ab-1", "DATE": "20.08.2026", "Num": "000048874",
                       "Врач": "Панкратов Эдуард Рашитович", "Врач_код": "90001"}


def test_stamp_does_not_mutate_the_stored_block():
    priem = {"GUID": "ab-1"}
    seed._stamp(priem, name="Панкратов Эдуард Рашитович", code="90001")
    assert priem == {"GUID": "ab-1"}


def test_stamp_returns_none_without_a_priem_block():
    assert seed._stamp(None, name="Панкратов Эдуард Рашитович", code="90001") is None


def test_unstamp_drops_both_keys():
    priem = {"GUID": "ab-1", "Врач": "Панкратов Эдуард Рашитович", "Врач_код": "90001"}
    assert seed._unstamp(priem) == {"GUID": "ab-1"}


def test_unstamp_returns_none_when_there_is_nothing_to_remove():
    assert seed._unstamp({"GUID": "ab-1"}) is None


# ── the run: dry-run writes nothing, --apply writes the stamped block ────────

class _FakeStorage:
    def __init__(self, rows, priems):
        self.rows = rows
        self.priems = priems
        self.written: list[tuple[str, dict]] = []

    async def list_audited_by_visit_date(self, *, organization_id, visit_date):
        return list(self.rows)

    async def get_priem(self, card_guid):
        return self.priems.get(card_guid)

    async def replace_priem(self, *, card_guid, priem):
        import json
        self.written.append((card_guid, json.loads(priem)))
        return True


def _run(storage, **kwargs):
    kwargs.setdefault("org_id", "11111111-1111-1111-1111-111111111111")
    kwargs.setdefault("visit_date", seed._parse_date("2026-08-20"))
    kwargs.setdefault("code", "90001")
    kwargs.setdefault("name", "Панкратов Эдуард Рашитович")
    kwargs.setdefault("limit", 10)
    kwargs.setdefault("apply", False)
    kwargs.setdefault("revert", False)
    return asyncio.run(seed._run(storage, **kwargs))


def test_dry_run_writes_nothing():
    storage = _FakeStorage([_row("a", diag=1)], {"a": {"GUID": "a"}})
    _run(storage)
    assert storage.written == []


def test_apply_writes_the_stamped_block():
    storage = _FakeStorage([_row("a", diag=1)], {"a": {"GUID": "a", "DATE": "20.08.2026"}})
    _run(storage, apply=True)
    assert storage.written == [
        ("a", {"GUID": "a", "DATE": "20.08.2026",
               "Врач": "Панкратов Эдуард Рашитович", "Врач_код": "90001"})]


def test_apply_skips_a_card_whose_priem_block_is_gone():
    storage = _FakeStorage([_row("a", diag=1)], {})     # get_priem -> None
    summary = _run(storage, apply=True)
    assert storage.written == []
    assert summary.skipped == 1


def test_revert_unstamps_only_our_own_cards():
    rows = [_row("a", code="90001"), _row("b", code="00012")]
    priems = {"a": {"GUID": "a", "Врач": "Панкратов Эдуард Рашитович", "Врач_код": "90001"},
              "b": {"GUID": "b", "Врач": "Губарева Елена Александровна", "Врач_код": "00012"}}
    storage = _FakeStorage(rows, priems)
    _run(storage, apply=True, revert=True)
    assert storage.written == [("a", {"GUID": "a"})]


def test_revert_dry_run_writes_nothing():
    rows = [_row("a", code="90001")]
    priems = {"a": {"GUID": "a", "Врач": "П", "Врач_код": "90001"}}
    storage = _FakeStorage(rows, priems)
    _run(storage, revert=True)
    assert storage.written == []
