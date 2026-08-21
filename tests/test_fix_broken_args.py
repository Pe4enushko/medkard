"""Argument and confirmation tests for scripts/fix-broken.py."""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest

from audit.broken_replay import BrokenGroup

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "fix_broken_script",
    ROOT / "scripts" / "fix-broken.py",
)
assert SPEC and SPEC.loader
fix_broken = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fix_broken)


def test_org_and_all_modes_parse() -> None:
    org_args = fix_broken.build_parser().parse_args(["Alenka"])
    all_args = fix_broken.build_parser().parse_args(["--all"])

    assert (org_args.org, org_args.all) == ("Alenka", False)
    assert (all_args.org, all_args.all) == (None, True)
    assert org_args.num_batches == 5


@pytest.mark.parametrize("argv", [[], ["Nope"], ["Alenka", "--all"]])
def test_invalid_target_combinations_fail(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        fix_broken.build_parser().parse_args(argv)


def test_optional_flags_parse() -> None:
    args = fix_broken.build_parser().parse_args(
        ["MDS", "-y", "--dry-run", "--num-batches", "2"]
    )
    assert (args.y, args.dry_run, args.num_batches) == (True, True, 2)


def test_dry_run_shows_plan_without_prompt_or_replay(monkeypatch, capsys) -> None:
    group = BrokenGroup(
        org_id="org-1",
        org_name="Alenka",
        visits=[{"Прием": {"GUID": "a"}}],
        guids={"a"},
    )
    args = Namespace(all=True, org=None, dry_run=True, y=False)
    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail("prompted"))

    assert fix_broken._confirmed([group], args) is False
    output = capsys.readouterr().out
    assert "Alenka: 1" in output
    assert "Dry run — nothing was written." in output


async def test_replay_uses_group_org_and_disables_dedup(monkeypatch) -> None:
    calls = {}

    class Pipeline:
        def __init__(self, *, org_id, card_filter):
            calls["org_id"] = org_id
            calls["filter"] = card_filter

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def run_batched(self, visits, *, num_batches, done_guids):
            calls["visits"] = visits
            calls["num_batches"] = num_batches
            calls["done_guids"] = done_guids
            return []

    monkeypatch.setattr(fix_broken, "AuditPipeline", Pipeline)
    monkeypatch.setattr(fix_broken, "_filter_for", lambda _group: "filter")
    group = BrokenGroup(
        org_id="org-1",
        org_name="Alenka",
        visits=[{"Прием": {"GUID": "a"}}],
        guids={"a"},
    )

    await fix_broken._replay(group, 3, fix_broken.logging.getLogger("test"))

    assert calls == {
        "org_id": "org-1",
        "filter": "filter",
        "visits": group.visits,
        "num_batches": 3,
        "done_guids": set(),
    }
