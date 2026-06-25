"""
audit/icd_check/validator.py — ICD-10 coding correctness check for a full visit.

Runs once per visit (not per diagnosis) so the agent sees the complete picture
of the doctor's coding decisions before flagging any individual code.

Usage::
    from audit.icd_check.validator import check_icd_codes

    issues, tokens = await check_icd_codes(patient, diagnoses, manifest_rows)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from LLM.client import LLMClient
from LLM.tools import get_icd_checker_tools
from storage.models.result import IcdCodingIssue, IssueSource

logger = logging.getLogger(__name__)

_client = LLMClient()


class _IcdSource(BaseModel):
    doc_title: str = Field(default="")
    section: str | None = Field(default=None)
    cite: str | None = Field(default=None)


class _IcdFinding(BaseModel):
    dx_index: int
    correct: bool
    confidence: int
    suggested_code: str = Field(default="")
    comment: str = Field(default="")
    sources: list[_IcdSource] = Field(default_factory=list)


class _IcdCheckerOutput(BaseModel):
    findings: list[_IcdFinding] = Field(default_factory=list)


_PROMPT_PATH = Path(__file__).parent.parent.parent / "LLM" / "prompts" / "icd_checker.txt"
_SYSTEM_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")


def _render_manifest_table(rows: list[dict[str, str]]) -> str:
    """Render age-filtered manifest rows as a plain text table for the agent."""
    header = "ID | Наименование | МКБ-10 | Возрастная категория"
    sep = "-" * len(header)
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"{row.get('ID', '')} | {row.get('Наименование', '')} | "
            f"{row.get('МКБ-10', '')} | {row.get('Возрастная категория', '')}"
        )
    return "\n".join(lines)


def _format_diagnoses(diagnoses: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, dx in enumerate(diagnoses):
        code = dx.get("КодМКБ", "—")
        name = dx.get("НаименованиеМКБ", "—")
        detail = dx.get("Детализация", "")
        line = f"{i}. Код МКБ: {code} — {name}"
        if detail:
            line += f" ({detail})"
        parts.append(line)
    return "\n".join(parts)


def _format_patient(patient: dict[str, Any]) -> str:
    return "\n".join(
        f"{k}: {v}" for k, v in patient.items() if v is not None
    )


def _format_inspection(inspection_data: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in inspection_data:
        key = str(item.get("Параметр", "")).strip()
        value = str(item.get("Значение", "")).strip()
        if key and value:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


async def check_icd_codes(
    patient: dict[str, Any],
    diagnoses: list[dict[str, Any]],
    manifest_rows: list[dict[str, str]],
    inspection_data: list[dict[str, Any]] | None = None,
) -> tuple[list[IcdCodingIssue], int]:
    """Check ICD-10 coding correctness for all diagnoses of a single visit.

    Runs one ReAct agent that sees the full coding picture of the visit at once.
    Only findings with confidence ≥ 8 are returned.

    Args:
        patient:         Patient info dict.
        diagnoses:       Full list of diagnosis dicts from visit["Диагнозы"].
        manifest_rows:   Age-filtered rows from manifest.csv.
        inspection_data: Optional inspection data from visit["ДанныеОсмотра"].

    Returns:
        (list[IcdCodingIssue], total_tokens)
    """
    if not diagnoses:
        return [], 0

    manifest_table = _render_manifest_table(manifest_rows)
    diagnoses_text = _format_diagnoses(diagnoses)
    patient_text = _format_patient(patient)
    inspection_text = _format_inspection(inspection_data or [])

    human_message = (
        "## Пациент\n"
        f"{patient_text}\n\n"
        "## Диагнозы врача (все диагнозы визита)\n"
        f"{diagnoses_text}\n\n"
    )
    if inspection_text:
        human_message += (
            "## Клинический контекст (данные осмотра)\n"
            f"{inspection_text}\n\n"
        )
    human_message += (
        "## Доступные клинические рекомендации (отфильтровано по возрасту пациента)\n"
        f"{manifest_table}"
    )

    logger.info(
        "[icd_check] launching ICD checker agent for %d diagnosis(es), age-filtered manifest rows=%d",
        len(diagnoses),
        len(manifest_rows),
    )
    logger.debug("[icd_check] diagnosis codes: %s", [dx.get("КодМКБ", "?") for dx in diagnoses])

    tools = get_icd_checker_tools()
    output, tokens = await _client.call_agent(
        _SYSTEM_PROMPT, tools, human_message, response_format=_IcdCheckerOutput
    )

    if not isinstance(output, _IcdCheckerOutput):
        logger.warning("[icd_check] unexpected agent output type: %s", type(output))
        return [], tokens

    issues: list[IcdCodingIssue] = []

    for entry in output.findings:
        if entry.correct:
            continue
        if entry.confidence < 8:
            continue
        if not entry.suggested_code or not entry.comment:
            continue
        if entry.dx_index < 0 or entry.dx_index >= len(diagnoses):
            logger.warning("[icd_check] dx_index %d out of range (len=%d)", entry.dx_index, len(diagnoses))
            continue

        initial_code = diagnoses[entry.dx_index].get("КодМКБ", "?")
        sources = [
            IssueSource(doc_title=s.doc_title, section=s.section, cite=s.cite)
            for s in entry.sources
        ]
        issues.append(IcdCodingIssue(
            dx_index=entry.dx_index,
            initial_code=initial_code,
            suggested_code=entry.suggested_code,
            confidence=entry.confidence,
            comment=entry.comment,
            sources=sources,
        ))
        logger.info(
            "[icd_check] selected ICD correction dx_index=%d: %s → %s "
            "(confidence=%d, comment=%s)",
            entry.dx_index,
            initial_code,
            entry.suggested_code,
            entry.confidence,
            entry.comment,
        )

    logger.info("[icd_check] done — %d issue(s), tokens=%d", len(issues), tokens)
    return issues, tokens
