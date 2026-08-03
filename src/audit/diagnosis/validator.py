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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from LLM.chinese_detector import ChineseDetector
from LLM.client import LLMClient
from LLM.tools import (
    get_anamnesis_tools_for,
    get_inspection_tools_for,
    get_treatment_tools_for,
)
from audit.diagnosis.schemas import CheckerIssue, CheckerOutput

_chinese_detector = ChineseDetector()
_client = LLMClient()
from audit.diagnosis.clinic_recs import ClinicRecs
from audit.models import DiagnosisAuditResult
from storage.models.result import DiagnosisIssue, IssueSource

# ── Checker prompts ───────────────────────────────────────────────────────────
_PROMPTS_DIR = Path(__file__).parent.parent.parent / "LLM" / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


_ANAMNESIS_PROMPT: str = _load_prompt("anamnesis_checker.txt")
_INSPECTION_PROMPT: str = _load_prompt("inspection_checker.txt")
_TREATMENT_PROMPT: str = _load_prompt("treatment_checker.txt")
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _CheckerRun:
    issues: list[DiagnosisIssue]


def _issue_from_schema(item: CheckerIssue) -> DiagnosisIssue | None:
    if not item.issue:
        return None

    sources = [
        IssueSource(doc_title=s.doc_title, section=s.section, cite=s.cite)
        for s in item.sources
    ]
    return DiagnosisIssue(issue=item.issue, sources=sources)


def _load_checker_json(output: str) -> Any | None:
    text = output.strip()
    candidates = [text]

    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        candidates.insert(0, match.group(1).strip())

    for marker in ("{", "["):
        idx = text.find(marker)
        if idx >= 0:
            candidates.append(text[idx:].strip())

    decoder = json.JSONDecoder()
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate)
            return parsed
        except json.JSONDecodeError:
            continue
    return None


def _parse_inspection_data(raw_visit: dict[str, Any]) -> str:
    items: list[dict] = raw_visit.get("ДанныеОсмотра", [])
    lines: list[str] = []
    for item in items:
        key = str(item.get("Параметр", "")).strip()
        value = str(item.get("Значение", "")).strip()
        if key and value:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _format_visit_value(value: Any, indent: int = 0) -> str:
    prefix = " " * indent

    if isinstance(value, dict):
        if not value:
            return f"{prefix}—"

        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_format_visit_value(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {item if item is not None else '—'}")
        return "\n".join(lines)

    if isinstance(value, list):
        if not value:
            return f"{prefix}—"

        lines: list[str] = []
        for idx, item in enumerate(value, start=1):
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{idx}.")
                lines.append(_format_visit_value(item, indent + 2))
            else:
                lines.append(f"{prefix}{idx}. {item if item is not None else '—'}")
        return "\n".join(lines)

    return f"{prefix}{value if value is not None else '—'}"


def _format_visit_context(raw_visit: dict[str, Any]) -> str:
    """Render all clinically relevant visit fields for guideline checkers."""
    excluded = {"Пациент", "Диагнозы", "Прием", "Врач"}
    preferred = [
        "Жалобы",
        "Анамнез",
        "ОбъективныйОсмотр",
        "Рекомендации",
        "Назначения",
        "Секции",
        "ДанныеОсмотра",
        "Услуги",
    ]

    keys: list[str] = []
    for key in preferred:
        if key in raw_visit:
            keys.append(key)
    keys.extend(key for key in raw_visit if key not in excluded and key not in keys)

    parts: list[str] = []
    for key in keys:
        value = raw_visit.get(key)
        if not _has_content(value):
            continue

        if key == "ДанныеОсмотра" and isinstance(value, list):
            rendered = _parse_inspection_data(raw_visit) or _format_visit_value(value)
        else:
            rendered = _format_visit_value(value)
        parts.append(f"## {key}\n{rendered}")

    return "\n\n".join(parts) if parts else "—"


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


def _parse_issues(output: str | CheckerOutput) -> list[DiagnosisIssue]:
    """Parse a checker agent's JSON output into a list of Issue objects."""
    if isinstance(output, CheckerOutput):
        return [
            issue
            for item in output.issues
            if (issue := _issue_from_schema(item)) is not None
        ]

    raw = _load_checker_json(output)
    if raw is None:
        return []

    if isinstance(raw, dict):
        try:
            parsed = CheckerOutput.model_validate(raw)
        except Exception:
            return []
        return [
            issue
            for item in parsed.issues
            if (issue := _issue_from_schema(item)) is not None
        ]

    if not isinstance(raw, list):
        return []

    issues: list[DiagnosisIssue] = []
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
                cite=s.get("cite"),
            )
            for s in item.get("sources", [])
            if isinstance(s, dict)
        ]
        issues.append(DiagnosisIssue(issue=issue_text, sources=sources))
    return issues


async def _run_checker(
    system_prompt: str,
    tools: list,
    human_message: str,
    checker_label: str = "checker",
    metadata: dict[str, Any] | None = None,
) -> tuple[_CheckerRun, int]:
    tool_names = [t.name for t in tools]
    logger.debug("[checker:%s] START — tools=%s", checker_label, tool_names)
    raw_answer, tokens = await _client.call_agent(
        system_prompt,
        tools,
        human_message,
        response_format=CheckerOutput,
        metadata={"checker": checker_label, **(metadata or {})},
    )
    logger.info("🤖 [checker:%s] raw LLM answer:\n%s", checker_label, raw_answer)
    issues = _parse_issues(raw_answer)
    for i, issue in enumerate(issues):
        if _chinese_detector.check_str(issue.issue):
            repaired, repair_tokens = await _chinese_detector.repair_issue(issue.issue)
            issues[i] = DiagnosisIssue(issue=repaired, sources=issue.sources)
            tokens += repair_tokens
    logger.debug("[checker:%s] parsed %d issue(s), tokens=%d", checker_label, len(issues), tokens)
    return _CheckerRun(issues=issues), tokens


class DiagnosisValidator:
    """Checks a single diagnosis against its clinical guideline via three agents.

    Args:
        visit: Raw visit dict (as parsed from the source JSON).
    """

    def __init__(self, visit: dict[str, Any]) -> None:
        # TODO: refactor to use parsers.json_parser.AppointmentParser.parse() instead of direct key access
        self._visit = visit
        self._clinic_recs = ClinicRecs()

    async def validate_diagnosis(
        self,
        diagnosis: dict[str, Any],
    ) -> tuple[DiagnosisAuditResult, int]:
        """Run anamnesis / inspection / treatment checker agents for *diagnosis*.

        Args:
            diagnosis: A single entry from the visit's «Диагнозы» list.

        Returns:
            (DiagnosisAuditResult, total_tokens) where total_tokens covers
            guideline lookup and all three checker agents.
        """
        patient: dict = self._visit.get("Пациент", {})
        dx_code = diagnosis.get("КодМКБ", "?")
        card_guid = (self._visit.get("Прием") or {}).get("GUID")
        logger.info("[diagnosis] validate_diagnosis START — dx=%s", dx_code)

        file_id, clinic_tokens = await self._clinic_recs.pick_recs(patient, diagnosis)
        logger.info("[diagnosis] guideline file_id picked: %s", file_id)

        anamnesis_issues: list[DiagnosisIssue] = []
        inspection_issues: list[DiagnosisIssue] = []
        treatment_issues: list[DiagnosisIssue] = []
        checker_tokens = 0

        if file_id:
            patient_info = "\n".join(f"{k}: {v}" for k, v in patient.items() if v is not None)
            human_message = (
                "## Пациент\n"
                f"{patient_info}\n\n"
                "## Диагноз\n"
                f"{_format_diagnosis(diagnosis)}\n\n"
                "## Клинический контекст записи\n"
                f"{_format_visit_context(self._visit)}"
            )
            logger.info("📨 [diagnosis] checker user prompt for dx=%s:\n%s", dx_code, human_message)
            logger.info("[diagnosis] launching anamnesis / inspection / treatment checkers in parallel")

            (anamnesis_run, a_tokens), (inspection_run, i_tokens), (treatment_run, t_tokens) = await asyncio.gather(
                _run_checker(
                    _ANAMNESIS_PROMPT,
                    get_anamnesis_tools_for(file_id),
                    human_message,
                    checker_label="anamnesis",
                    metadata={"card_guid": card_guid, "dx_code": dx_code},
                ),
                _run_checker(
                    _INSPECTION_PROMPT,
                    get_inspection_tools_for(file_id),
                    human_message,
                    checker_label="inspection",
                    metadata={"card_guid": card_guid, "dx_code": dx_code},
                ),
                _run_checker(
                    _TREATMENT_PROMPT,
                    get_treatment_tools_for(file_id),
                    human_message,
                    checker_label="treatment",
                    metadata={"card_guid": card_guid, "dx_code": dx_code},
                ),
            )
            anamnesis_issues = anamnesis_run.issues
            inspection_issues = inspection_run.issues
            treatment_issues = treatment_run.issues
            checker_tokens = a_tokens + i_tokens + t_tokens
            logger.info(
                "[diagnosis] checkers done — anamnesis=%d inspection=%d treatment=%d tokens=%d",
                len(anamnesis_issues), len(inspection_issues), len(treatment_issues), checker_tokens,
            )
        else:
            logger.warning(
                "[diagnosis] dx=%s — Для такого МКБ кода нет прямых клинических рекоммендаций",
                dx_code,
            )

        total_tokens = clinic_tokens + checker_tokens
        return DiagnosisAuditResult(
            anamnesis_issues=anamnesis_issues,
            inspection_issues=inspection_issues,
            treatment_issues=treatment_issues,
            guideline_file_id=file_id,
            icd_code=dx_code,
        ), total_tokens
