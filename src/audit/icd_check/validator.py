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
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from LLM.client import LLMClient
from LLM.tools import get_icd_checker_tools
from storage.models.guideline import Guideline
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
_MANIFEST_MAX_ROWS = int(os.environ.get("ICD_CHECK_MANIFEST_MAX_ROWS", "120"))


def _render_manifest_table(rows: list["Guideline"]) -> str:
    """Render age-filtered guideline rows as a plain text table for the agent."""
    header = "ID | Наименование | МКБ-10 | Возрастная категория"
    sep = "-" * len(header)
    lines = [header, sep]
    for g in rows:
        lines.append(
            f"{g.file_id} | {g.name or ''} | "
            f"{', '.join(g.mkb)} | {', '.join(g.age_category)}"
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


def _code_compact(value: str) -> str:
    return "".join(ch for ch in value.upper() if ch.isalnum())


def _diagnosis_codes(diagnoses: list[dict[str, Any]]) -> list[str]:
    return [
        str(dx.get("КодМКБ", "")).strip().upper()
        for dx in diagnoses
        if str(dx.get("КодМКБ", "")).strip()
    ]


def _select_manifest_rows(
    rows: list[Guideline],
    diagnoses: list[dict[str, Any]],
    *,
    limit: int = _MANIFEST_MAX_ROWS,
) -> list[Guideline]:
    """Keep the ICD agent prompt bounded while preserving likely alternatives."""
    if limit <= 0 or len(rows) <= limit:
        return rows

    dx_codes = _diagnosis_codes(diagnoses)
    if not dx_codes:
        return rows[:limit]

    dx_compact = [_code_compact(code) for code in dx_codes]
    dx_chapters = {code[0] for code in dx_compact if code}
    dx_three = {code[:3] for code in dx_compact if len(code) >= 3}

    exact_or_prefix: list[Guideline] = []
    same_chapter: list[Guideline] = []
    for row in rows:
        row_codes = [_code_compact(code) for code in row.mkb]
        if any(
            row_code == dx
            or row_code.startswith(dx)
            or dx.startswith(row_code)
            or row_code[:3] in dx_three
            for row_code in row_codes
            for dx in dx_compact
            if row_code and dx
        ):
            exact_or_prefix.append(row)
            continue
        if any(row_code[:1] in dx_chapters for row_code in row_codes if row_code):
            same_chapter.append(row)

    selected = [*exact_or_prefix, *same_chapter]
    return selected[:limit] if selected else rows[:limit]


def _is_length_limit_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "lengthfinishreasonerror" in text or "length limit was reached" in text


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
    manifest_rows: list[Guideline],
    inspection_data: list[dict[str, Any]] | None = None,
    card_guid: str | None = None,
) -> tuple[list[IcdCodingIssue], int]:
    """Check ICD-10 coding correctness for all diagnoses of a single visit.

    Runs one ReAct agent that sees the full coding picture of the visit at once.
    Only findings with confidence ≥ 8 are returned.

    Args:
        patient:         Patient info dict.
        diagnoses:       Full list of diagnosis dicts from visit["Диагнозы"].
        manifest_rows:   Age-filtered Guideline rows from GuidelinesStorage.
        inspection_data: Optional inspection data from visit["ДанныеОсмотра"].

    Returns:
        (list[IcdCodingIssue], total_tokens)
    """
    if not diagnoses:
        return [], 0

    selected_manifest_rows = _select_manifest_rows(manifest_rows, diagnoses)
    manifest_table = _render_manifest_table(selected_manifest_rows)
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
        "[icd_check] launching ICD checker agent for %d diagnosis(es), manifest rows=%d/%d",
        len(diagnoses),
        len(selected_manifest_rows),
        len(manifest_rows),
    )
    logger.debug("[icd_check] diagnosis codes: %s", [dx.get("КодМКБ", "?") for dx in diagnoses])

    tools = get_icd_checker_tools()
    try:
        output, tokens = await _client.call_agent(
            _SYSTEM_PROMPT,
            tools,
            human_message,
            response_format=_IcdCheckerOutput,
            metadata={"card_guid": card_guid, "checker": "icd"},
        )
    except Exception as exc:
        if _is_length_limit_error(exc):
            logger.warning(
                "[icd_check] agent hit output length limit; skipping ICD findings for this visit"
            )
            return [], 0
        raise

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
