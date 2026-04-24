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
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from LLM.rag_agent import create_checker_agent
from LLM.tools import (
    get_anamnesis_tools_for,
    get_inspection_tools_for,
    get_treatment_tools_for,
)
from audit.diagnosis.clinic_recs import ClinicRecs
from audit.models import DiagnosisAuditResult
from storage.models.result import DiagnisisIssue, IssueSource

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


def _parse_issues(output: str) -> list[DiagnisisIssue]:
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

    issues: list[DiagnisisIssue] = []
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
        issues.append(DiagnisisIssue(issue=issue_text, sources=sources))
    return issues


def _format_message_for_log(message: Any, idx: int) -> str:
    if isinstance(message, tuple) and len(message) == 2:
        role, content = message
        return f"{idx}. {role}\n{content}"

    message_type = getattr(message, "type", None) or message.__class__.__name__
    content = getattr(message, "content", "")
    tool_calls = getattr(message, "tool_calls", None) or []
    tool_call_id = getattr(message, "tool_call_id", None)
    name = getattr(message, "name", None)

    header_parts = [f"{idx}. {message_type}"]
    if name:
        header_parts.append(f"name={name}")
    if tool_call_id:
        header_parts.append(f"tool_call_id={tool_call_id}")

    lines = [" | ".join(header_parts)]
    if content:
        lines.append(str(content))
    if tool_calls:
        lines.append("tool_calls:")
        for call in tool_calls:
            call_name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            call_args = call.get("args") if isinstance(call, dict) else getattr(call, "args", "")
            lines.append(f"  - {call_name}: {call_args}")

    return "\n".join(lines)


def _log_checker_messages(checker_label: str, system_prompt: str, messages: list[Any]) -> None:
    formatted_messages = "\n\n".join(
        [_format_message_for_log(("system", system_prompt), 0)]
        + [_format_message_for_log(message, idx) for idx, message in enumerate(messages, start=1)]
    )
    logger.info("[checker:%s] LLM message trace:\n%s", checker_label, formatted_messages)


async def _run_checker(
    system_prompt: str,
    tools: list,
    human_message: str,
    checker_label: str = "checker",
) -> list[DiagnisisIssue]:
    tool_names = [t.name for t in tools]
    logger.debug("[checker:%s] START — tools=%s", checker_label, tool_names)
    logger.debug("[checker:%s] system_prompt:\n%s", checker_label, system_prompt)
    agent = create_checker_agent(system_prompt, tools)
    result = await agent.ainvoke({"messages": [("user", human_message)]})
    _log_checker_messages(checker_label, system_prompt, result["messages"])
    last_msg = result["messages"][-1]
    raw_answer = last_msg.content
    finish_reason = (getattr(last_msg, "response_metadata", {}) or {}).get("finish_reason")
    if finish_reason and finish_reason != "stop":
        logger.error(
            "[checker:%s] unexpected finish_reason=%r; response_metadata: %s",
            checker_label,
            finish_reason,
            getattr(last_msg, "response_metadata", {}),
        )
    logger.debug("[checker:%s] raw LLM answer:\n%s", checker_label, raw_answer)
    issues = _parse_issues(raw_answer)
    logger.debug("[checker:%s] parsed %d issue(s)", checker_label, len(issues))
    return issues


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
        dx_code = diagnosis.get("КодМКБ", "?")
        logger.info("[diagnosis] validate_diagnosis START — dx=%s", dx_code)
        logger.debug(
            "[diagnosis] diagnosis input:\n%s",
            json.dumps(diagnosis, ensure_ascii=False, indent=2),
        )
        logger.debug(
            "[diagnosis] patient context:\n%s",
            json.dumps(patient, ensure_ascii=False, indent=2),
        )

        file_id = await self._clinic_recs.pick_recs(patient, diagnosis)
        logger.info("[diagnosis] guideline file_id picked: %s", file_id)

        anamnesis_issues: list[DiagnisisIssue] = []
        inspection_issues: list[DiagnisisIssue] = []
        treatment_issues: list[DiagnisisIssue] = []

        if file_id:
            patient_info = "\n".join(f"{k}: {v}" for k, v in patient.items() if v is not None)
            human_message = (
                "## Пациент\n"
                f"{patient_info}\n\n"
                "## Диагноз\n"
                f"{_format_diagnosis(diagnosis)}\n\n"
                "## Клинический контекст (данные осмотра)\n"
                f"{_parse_inspection_data(self._visit)}"
            )
            logger.debug("[diagnosis] human_message sent to checkers:\n%s", human_message)
            logger.info("[diagnosis] launching anamnesis / inspection / treatment checkers in parallel")

            anamnesis_issues, inspection_issues, treatment_issues = await asyncio.gather(
                _run_checker(
                    _ANAMNESIS_PROMPT,
                    get_anamnesis_tools_for(file_id),
                    human_message,
                    checker_label="anamnesis",
                ),
                _run_checker(
                    _INSPECTION_PROMPT,
                    get_inspection_tools_for(file_id),
                    human_message,
                    checker_label="inspection",
                ),
                _run_checker(
                    _TREATMENT_PROMPT,
                    get_treatment_tools_for(file_id),
                    human_message,
                    checker_label="treatment",
                ),
            )
            logger.info(
                "[diagnosis] checkers done — anamnesis=%d inspection=%d treatment=%d",
                len(anamnesis_issues), len(inspection_issues), len(treatment_issues),
            )
            logger.debug("[diagnosis] anamnesis_issues: %s", anamnesis_issues)
            logger.debug("[diagnosis] inspection_issues: %s", inspection_issues)
            logger.debug("[diagnosis] treatment_issues: %s", treatment_issues)
        else:
            logger.warning("[diagnosis] no guideline file_id — skipping checker agents for dx=%s", dx_code)

        return DiagnosisAuditResult(
            anamnesis_issues=anamnesis_issues,
            inspection_issues=inspection_issues,
            treatment_issues=treatment_issues,
            guideline_file_id=file_id,
        )
