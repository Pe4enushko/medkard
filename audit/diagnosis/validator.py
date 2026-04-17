"""
DiagnosisValidator — clinical-guideline checker for a single diagnosis.

Workflow::
    validator = DiagnosisValidator(visit)
    result    = await validator.validate_diagnosis(diagnosis)
    # DiagnosisAuditResult(anamnesis_issues, inspection_issues, treatment_issues, ...)

Responsibilities (narrow):
- Look up the relevant guideline via ClinicRecs.
- Run the three checker agents (anamnesis / inspection / treatment) in parallel.
- Return a DiagnosisAuditResult.

Formal structure checking and Excel logging are handled by audit.pipeline.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from LLM.rag_agent import create_checker_agent
from LLM.tools import (
    get_anamnesis_tools_for,
    get_inspection_tools_for,
    get_treatment_tools_for,
)
from audit.diagnosis.clinic_recs import ClinicRecs
from audit.models import DiagnosisAuditResult
from storage.models.result import Issue, IssueSource

# ── Checker prompts ───────────────────────────────────────────────────────────
_PROMPTS_DIR = Path(__file__).parent.parent.parent / "LLM" / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


_ANAMNESIS_PROMPT: str = _load_prompt("anamnesis_checker.txt")
_INSPECTION_PROMPT: str = _load_prompt("inspection_checker.txt")
_TREATMENT_PROMPT: str = _load_prompt("treatment_checker.txt")
# ─────────────────────────────────────────────────────────────────────────────


def _parse_inspection_data(raw_visit: dict[str, Any]) -> str:
    items: list[dict] = raw_visit.get("ДанныеОсмотра", [])
    lines: list[str] = []
    for item in items:
        key = str(item.get("Параметр", "")).strip()
        value = str(item.get("Значение", "")).strip()
        if key and value:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _format_diagnosis(diagnosis: dict[str, Any]) -> str:
    code = diagnosis.get("КодМКБ", "—")
    name = diagnosis.get("НаименованиеМКБ", "—")
    detail = diagnosis.get("Детализация", "")
    first = diagnosis.get("ВыявленВпервые")

    lines = [f"Код МКБ: {code}", f"Наименование МКБ: {name}"]
    if detail:
        lines.append(f"Детализация: {detail}")
    if first is not None:
        lines.append(f"Выявлен впервые: {'да' if first else 'нет'}")
    return "\n".join(lines)


def _parse_issues(output: str) -> list[Issue]:
    """Parse a checker agent's JSON output into a list of Issue objects."""
    text = output.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        raw: list[dict] = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(raw, list):
        return []

    issues: list[Issue] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        issue_text = item.get("issue", "")
        if not issue_text:
            continue
        sources = [
            IssueSource(
                doc_title=s.get("doc_title", ""),
                section=s.get("section"),
            )
            for s in item.get("sources", [])
            if isinstance(s, dict)
        ]
        issues.append(Issue(issue=issue_text, sources=sources))
    return issues


async def _run_checker(system_prompt: str, tools: list, human_message: str) -> list[Issue]:
    agent = await create_checker_agent(system_prompt, tools)
    result = await agent.ainvoke({"input": human_message})
    return _parse_issues(result.get("output", "[]"))


class DiagnosisValidator:
    """Checks a single diagnosis against its clinical guideline via three agents.

    Args:
        visit: Raw visit dict (as parsed from the source JSON).
    """

    def __init__(self, visit: dict[str, Any]) -> None:
        self._visit = visit
        self._clinic_recs = ClinicRecs()

    async def validate_diagnosis(
        self,
        diagnosis: dict[str, Any],
    ) -> DiagnosisAuditResult:
        """Run anamnesis / inspection / treatment checker agents for *diagnosis*.

        Args:
            diagnosis: A single entry from the visit's «Диагнозы» list.

        Returns:
            :class:`DiagnosisAuditResult` with issues grouped by checker type.
        """
        patient: dict = self._visit.get("Пациент", {})
        file_id = await self._clinic_recs.pick_recs(patient, diagnosis)

        anamnesis_issues: list[Issue] = []
        inspection_issues: list[Issue] = []
        treatment_issues: list[Issue] = []

        if file_id:
            human_message = (
                "## Диагноз\n"
                f"{_format_diagnosis(diagnosis)}\n\n"
                "## Клинический контекст (данные осмотра)\n"
                f"{_parse_inspection_data(self._visit)}"
            )

            anamnesis_issues, inspection_issues, treatment_issues = await asyncio.gather(
                _run_checker(_ANAMNESIS_PROMPT, get_anamnesis_tools_for(file_id), human_message),
                _run_checker(_INSPECTION_PROMPT, get_inspection_tools_for(file_id), human_message),
                _run_checker(_TREATMENT_PROMPT, get_treatment_tools_for(file_id), human_message),
            )

        return DiagnosisAuditResult(
            anamnesis_issues=anamnesis_issues,
            inspection_issues=inspection_issues,
            treatment_issues=treatment_issues,
            guideline_file_id=file_id,
        )

