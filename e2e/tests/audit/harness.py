"""
harness.py — shared runner for the audit e2e scripts.

Each audit script declares a list of `Case`s and hands them to `run_cases`.
A case is one fixture card carrying exactly one defect plus the flag that
defect must produce.

The run happens in two stages:

  Stage 1 — parsing, no LLM. Confirms the card is classified as the fixture
  intended (get_visit_types) and that the rule under test actually reaches
  the prompt (get_rules). This is where a mis-built fixture fails, before a
  single token is spent. If any case fails stage 1 the script stops — stage
  2 is not attempted.

  Stage 2 — the full audit. AuditPipeline._audit_visit runs the formal
  validator, the ICD checker and DiagnosisValidator against the live LLM,
  exactly as production does.

Stage 2 asserts the **complete** set of formal flags equals the one
expected flag. That is deliberate: because every fixture carries exactly
one defect, a rule that fires indiscriminately shows up as an extra flag
and fails the case. A presence-only assert could never catch that — see
docs/e2e-testing.md for the full rationale.

Nothing is persisted. _audit_visit calls _upsert_done_card, but that
returns immediately while self._done_cards is None, and that field is only
set by AuditPipeline.__aenter__ — so the pipeline is deliberately used
*without* `async with`, and no teardown is needed. The database is still
required: the ICD checker reads the guidelines catalogue.

An empty finding list is never taken at face value. LLM.validations
returns [] both when the model reported no defects and when its answer
failed to parse — the parse failure only shows up as a log record. The
runner listens for that record so an unparsed answer can never be read as
"no violations".

There are no command-line flags. These scripts are meant for unattended
batch runs; anything that changes what they assert would make a green run
mean different things on different days.
"""

from __future__ import annotations

import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from audit.formal_structure.validator import FormalValidator, VisitType  # noqa: E402
from audit.pipeline import AuditPipeline  # noqa: E402

_OK, _BAD = "  \033[32mok\033[0m    ", "  \033[31mFAILED\033[0m"


@dataclass(frozen=True)
class Case:
    """One fixture card and the single flag its single defect must raise."""

    name: str
    visit: dict[str, Any]
    expect: str
    visit_types: set[VisitType]


class _Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        if ok:
            print(f"{_OK}{label}")
        else:
            print(f"{_BAD}{label}")
            for line in (detail or "").splitlines():
                print(f"          {line}")
            self.failures.append(label)
        return ok


def _flags(result: Any) -> set[str]:
    return {f.flag for f in result.formal.findings}


def _describe(result: Any) -> str:
    if not result.formal.findings:
        return "(no findings)"
    return "\n".join(f"{f.flag}: {f.issue}" for f in result.formal.findings)


class _FormalCallWatch(logging.Handler):
    """Watches the formal-validator call so an unparsed answer is not read as a clean card.

    LLM.validations returns an empty list when the model's answer cannot be
    parsed and only records logger.error(...), which makes a parse failure
    indistinguishable from "no defects" at the Result level.
    """

    _MARKER = "failed to parse JSON response"
    _TALLY = "LLM returned"
    _DROPPED = "dropping unrecognised flag"
    _FUZZY = "fuzzy-matched to"

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.parse_failed = False
        self.tally: str = ""
        self.dropped: list[str] = []
        self.fuzzy: list[str] = []

    def reset(self) -> None:
        self.parse_failed = False
        self.tally = ""
        self.dropped = []
        self.fuzzy = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            return
        if self._MARKER in message:
            self.parse_failed = True
        elif self._DROPPED in message:
            self.dropped.append(message.split(self._DROPPED, 1)[1].strip())
        elif self._FUZZY in message:
            self.fuzzy.append(message.split("] ", 1)[-1].strip())
        elif self._TALLY in message:
            self.tally = message.split(self._TALLY, 1)[1].split(":", 1)[0].strip()

    def install(self) -> None:
        for name in ("LLM.validations", "audit.formal_structure.validator"):
            logger = logging.getLogger(name)
            logger.addHandler(self)
            logger.setLevel(min(logger.level or logging.INFO, logging.INFO))


async def _stage_one(cases: list[Case], report: _Report) -> None:
    """Classification and rule selection — deterministic, no LLM, no database."""
    validator = FormalValidator()
    for case in cases:
        visit = case.visit
        got_types = await validator.get_visit_types(visit)
        report.check(
            f"[{case.name}] visit type — {', '.join(sorted(t.name for t in case.visit_types))}",
            got_types == case.visit_types,
            f"got: {', '.join(sorted(t.name for t in got_types)) or '(empty)'}",
        )

        age = visit["Пациент"]["AGE"]
        rules = validator.get_rules(got_types, age)
        selected = [r["flag_code"] for r in rules]
        report.check(
            f"[{case.name}] rule {case.expect} reached the prompt ({len(rules)} rules)",
            case.expect in selected,
            "selected: " + ", ".join(selected),
        )


async def _stage_two(cases: list[Case], report: _Report) -> None:
    """The real audit — formal validator, ICD checker and DiagnosisValidator."""
    watch = _FormalCallWatch()
    watch.install()
    for case in cases:
        print(f"\n  {case.name}")
        watch.reset()
        pipeline = AuditPipeline()  # deliberately not `async with` — see module docstring
        try:
            result = await pipeline._audit_visit(case.visit)
        except Exception:
            report.check(f"[{case.name}] audit ran without error", False, traceback.format_exc())
            continue

        if watch.tally:
            print(f"          formal call: {watch.tally}")
        for line in watch.fuzzy:
            print(f"          flag fuzzy-matched: {line}")
        if watch.dropped:
            report.check(
                f"[{case.name}] no flag dropped as unrecognised",
                False,
                "model returned flags not in rules.json: " + "; ".join(watch.dropped),
            )

        if not report.check(
            f"[{case.name}] formal validator response parsed",
            not watch.parse_failed,
            "LLM.validations could not parse the model's answer and returned an empty list; "
            "see the log for 'failed to parse JSON response'",
        ):
            continue

        got = _flags(result)
        report.check(
            f"[{case.name}] exactly one flag found — {case.expect}",
            got == {case.expect},
            f"extra: {', '.join(sorted(got - {case.expect})) or '—'}\n"
            f"missing: {', '.join(sorted({case.expect} - got)) or '—'}\n"
            f"full findings:\n{_describe(result)}",
        )

        expected_dx = len(case.visit["Диагнозы"])
        report.check(
            f"[{case.name}] DiagnosisValidator ran for all {expected_dx} diagnoses",
            len(result.diagnosis) >= expected_dx,
            f"got {len(result.diagnosis)} result(s)",
        )
        for dr in result.diagnosis:
            found = dr.guideline_file_id or "no guideline found"
            print(f"          diagnosis {dr.icd_code}: {found}, {len(dr.issues)} issue(s)")


async def run_cases(title: str, cases: list[Case]) -> int:
    """Run every case through both stages and return a process exit code."""
    report = _Report()
    print(f"\n{title}")
    print(f"  {len(cases)} fixture(s), one violation each\n")

    print("Stage 1 — parsing fixtures (no LLM)")
    await _stage_one(cases, report)
    if report.failures:
        print(
            f"\n\033[31mStage 1 failed ({len(report.failures)}), "
            f"full audit not run — no tokens spent.\033[0m"
        )
        for f in report.failures:
            print(f"  - {f}")
        return 1

    print("\nStage 2 — full audit (LLM + DB)")
    await _stage_two(cases, report)

    if report.failures:
        print(f"\n\033[31m{len(report.failures)} check(s) failed\033[0m")
        for f in report.failures:
            print(f"  - {f}")
        return 1
    print("\n\033[32mAll checks passed.\033[0m")
    return 0
