"""
harness.py — shared runner for the audit e2e scripts.

Each audit script declares a list of `Case`s and hands them to `run_cases`.
A case is one fixture card carrying exactly one defect plus the flag that
defect must produce.

The run happens in two stages:

  Stage 1 — parsing, no LLM.  Confirms the card is classified as the fixture
  intended (`get_visit_types`) and that the rule under test actually reaches
  the prompt (`get_rules`).  This is where a mis-built fixture fails, before
  a single token is spent, and it is also the stage that exercises the rule
  wiring this branch revised: visit_types, age_group and icd_prefixes.
  If any case fails stage 1 the script stops — stage 2 is not attempted.

  Stage 2 — the full audit.  `AuditPipeline._audit_visit` runs the formal
  validator, the ICD checker and DiagnosisValidator against the live LLM,
  exactly as production does.

Stage 2 asserts the **complete** set of formal flags equals the one expected
flag.  That is deliberate and it is stricter than the presence-only rule in
docs/superpowers/specs/2026-08-20-e2e-full-suite-design.md: because every
fixture carries exactly one defect, a rule that fires indiscriminately shows
up as an extra flag on the other nineteen fixtures and fails them.  A
presence-only assert could never catch that.

Nothing is persisted.  `_audit_visit` calls `_upsert_done_card`, but that
returns immediately while `self._done_cards is None`, and it is only set by
`AuditPipeline.__aenter__` — so the pipeline is deliberately used *without*
`async with`, and no teardown is needed.  The database is still required:
the ICD checker reads the guidelines catalogue.

An empty finding list is never taken at face value.  `LLM.validations`
returns `[]` both when the model reported no defects and when its answer
failed to parse — the parse failure only shows up as a log record.  The
runner listens for that record and for the validator's own tally, so
«ответ не разобран» can never be read as «нарушений нет».

There are no command-line flags.  These scripts are meant for unattended
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from audit.formal_structure.validator import FormalValidator, VisitType  # noqa: E402
from audit.pipeline import AuditPipeline  # noqa: E402

_OK, _BAD = "  \033[32mok\033[0m    ", "  \033[31mПРОВАЛ\033[0m"


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
        return "(находок нет)"
    return "\n".join(f"{f.flag}: {f.issue}" for f in result.formal.findings)


class _FormalCallWatch(logging.Handler):
    """Watches the formal-validator call so an unparsed answer is not read as a clean card.

    `LLM.validations._parse_findings` returns an empty list when the model's
    answer cannot be parsed and only records `logger.error(...)`, which makes
    a parse failure indistinguishable from «no defects» at the Result level.
    Two records are captured per case: that error, and the validator's own
    "[formal] LLM returned N finding(s), tokens=T" tally.
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
            # _enrich_flags silently discards a flag it cannot match to rules.json,
            # which is another way an empty finding list can mislead.
            self.dropped.append(message.split(self._DROPPED, 1)[1].strip())
        elif self._FUZZY in message:
            self.fuzzy.append(message.split("] ", 1)[-1].strip())
        elif self._TALLY in message:
            # keep only the tail: "N finding(s), tokens=T: [...]"
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
        # get_visit_types is declared async but does no I/O.
        got_types = await validator.get_visit_types(visit)
        report.check(
            f"[{case.name}] тип визита — {', '.join(sorted(t.name for t in case.visit_types))}",
            got_types == case.visit_types,
            f"получено: {', '.join(sorted(t.name for t in got_types)) or '(пусто)'}",
        )

        age = visit["Пациент"]["AGE"]
        icd = [d["КодМКБ"] for d in visit["Диагнозы"]]
        rules = validator.get_rules(got_types, age, icd, visit)
        selected = [r["flag_code"] for r in rules]
        report.check(
            f"[{case.name}] правило {case.expect} попало в промпт ({len(rules)} правил)",
            case.expect in selected,
            "в наборе: " + ", ".join(selected),
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
            report.check(f"[{case.name}] аудит отработал", False, traceback.format_exc())
            continue

        if watch.tally:
            print(f"          формальный вызов: {watch.tally}")
        for line in watch.fuzzy:
            print(f"          флаг приведён по опечатке: {line}")
        if watch.dropped:
            report.check(
                f"[{case.name}] ни один флаг не отброшен как неизвестный",
                False,
                "модель вернула флаги, которых нет в rules.json: " + "; ".join(watch.dropped),
            )

        # An unparsed answer also yields zero findings — separate it out first,
        # otherwise it reads as «карта без нарушений».
        if not report.check(
            f"[{case.name}] ответ формального валидатора разобран",
            not watch.parse_failed,
            "LLM.validations не смогла разобрать ответ модели и вернула пустой список; "
            "текст ответа — в логе записью «failed to parse JSON response»",
        ):
            continue

        got = _flags(result)
        report.check(
            f"[{case.name}] найден ровно один флаг — {case.expect}",
            got == {case.expect},
            f"лишние: {', '.join(sorted(got - {case.expect})) or '—'}\n"
            f"не найдено: {', '.join(sorted({case.expect} - got)) or '—'}\n"
            f"находки целиком:\n{_describe(result)}",
        )

        expected_dx = len(case.visit["Диагнозы"])
        report.check(
            f"[{case.name}] DiagnosisValidator отработал по всем {expected_dx} диагнозам",
            len(result.diagnosis) >= expected_dx,
            f"получено результатов: {len(result.diagnosis)}",
        )
        for dr in result.diagnosis:
            found = dr.guideline_file_id or "клинрек не найден"
            print(f"          диагноз {dr.icd_code}: {found}, замечаний {len(dr.issues)}")


async def run_cases(title: str, cases: list[Case]) -> int:
    """Run every case through both stages and return a process exit code."""
    report = _Report()
    print(f"\n{title}")
    print(f"  {len(cases)} фикстур(ы), по одному нарушению в каждой\n")

    print("Этап 1 — разбор фикстур (без LLM)")
    await _stage_one(cases, report)
    if report.failures:
        print(
            f"\n\033[31mЭтап 1 провален ({len(report.failures)}), "
            f"полный аудит не запускался — токены не потрачены.\033[0m"
        )
        for f in report.failures:
            print(f"  - {f}")
        return 1

    print("\nЭтап 2 — полный аудит (LLM + БД)")
    await _stage_two(cases, report)

    if report.failures:
        print(f"\n\033[31mПровалено проверок: {len(report.failures)}\033[0m")
        for f in report.failures:
            print(f"  - {f}")
        return 1
    print("\n\033[32mВсе проверки пройдены.\033[0m")
    return 0
