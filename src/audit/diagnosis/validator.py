"""Clinical-guideline audit for one diagnosis through a deterministic graph."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from audit.diagnosis.clinic_recs import ClinicRecs
from audit.models import DiagnosisAuditResult
from storage.models.result import (
    DiagnosisIssue,
    GuidelineSource,
    GuidelineSourceSection,
    IssueSource,
)

logger = logging.getLogger(__name__)

_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        from LLM.graphs.diagnosis import build_diagnosis_graph

        _compiled_graph = build_diagnosis_graph()
    return _compiled_graph


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
        lines = []
        for index, item in enumerate(value, start=1):
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{index}.")
                lines.append(_format_visit_value(item, indent + 2))
            else:
                lines.append(f"{prefix}{index}. {item if item is not None else '—'}")
        return "\n".join(lines)
    return f"{prefix}{value if value is not None else '—'}"


def _format_visit_context(raw_visit: dict[str, Any]) -> str:
    """Render all clinically relevant visit fields for graph nodes."""
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
    keys = [key for key in preferred if key in raw_visit]
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
    lines = [
        f"Код МКБ: {diagnosis.get('КодМКБ', '—')}",
        f"Наименование МКБ: {diagnosis.get('НаименованиеМКБ', '—')}",
    ]
    if diagnosis.get("Детализация"):
        lines.append(f"Детализация: {diagnosis['Детализация']}")
    if diagnosis.get("ВыявленВпервые") is not None:
        lines.append(
            f"Выявлен впервые: {'да' if diagnosis['ВыявленВпервые'] else 'нет'}"
        )
    return "\n".join(lines)


def _visit_date(raw: object) -> date | None:
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    value = raw.strip()
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        pass
    try:
        day, month, year = (int(part) for part in value.split("."))
        return date(year, month, day)
    except (TypeError, ValueError):
        return None


def _issue_from_graph(raw: dict[str, Any]) -> DiagnosisIssue:
    return DiagnosisIssue(
        issue=raw["issue"],
        aspect=raw.get("aspect"),
        sources=[
            IssueSource(
                doc_title=source.get("doc_title", ""),
                section=source.get("section"),
                cite=source.get("cite"),
                chunk_id=source.get("chunk_id"),
                chunk_index=source.get("chunk_index"),
            )
            for source in raw.get("sources", [])
        ],
    )


def _guideline_source_from_graph(raw: dict[str, Any]) -> GuidelineSource:
    return GuidelineSource(
        file_id=raw.get("file_id", ""),
        doc_title=raw.get("doc_title", ""),
        sections=[
            GuidelineSourceSection(
                section=section.get("section"),
                chunk_indices=list(section.get("chunk_indices") or []),
                cited=bool(section.get("cited", False)),
            )
            for section in raw.get("sections", [])
        ],
    )


class DiagnosisValidator:
    def __init__(self, visit: dict[str, Any]) -> None:
        self._visit = visit
        self._clinic_recs = ClinicRecs()

    async def validate_diagnosis(
        self,
        diagnosis: dict[str, Any],
    ) -> tuple[DiagnosisAuditResult, int]:
        patient = self._visit.get("Пациент") or {}
        dx_code = diagnosis.get("КодМКБ", "?")
        file_id, clinic_tokens = await self._clinic_recs.pick_recs(patient, diagnosis)
        if not file_id:
            return DiagnosisAuditResult(
                guideline_file_id=None, icd_code=dx_code
            ), clinic_tokens

        from RAG.retrieval.searches import get_sections_for_file
        from storage.guidelines_storage import GuidelinesStorage

        async with GuidelinesStorage() as storage:
            guideline = await storage.get(file_id)
        doc_title = guideline.name if guideline and guideline.name else file_id
        toc = await get_sections_for_file(file_id)
        patient_block = (
            "\n".join(
                f"{key}: {value}" for key, value in patient.items() if value is not None
            )
            or "—"
        )
        visit_meta = self._visit.get("Прием") or {}
        initial_state = {
            "visit_context": _format_visit_context(self._visit),
            "patient_block": patient_block,
            "diagnosis_block": _format_diagnosis(diagnosis),
            "visit_date": _visit_date(visit_meta.get("DATE")),
            "file_id": file_id,
            "doc_title": doc_title,
            "toc": toc,
            "card_guid": visit_meta.get("GUID"),
            "dx_code": dx_code,
            "pools": {},
            "issues": {},
            "errors": [],
            "tokens": 0,
        }
        graph_result = await _get_graph().ainvoke(initial_state)
        issues = graph_result.get("issues", {})
        result = DiagnosisAuditResult(
            anamnesis_issues=[
                _issue_from_graph(item) for item in issues.get("anamnesis", [])
            ],
            inspection_issues=[
                _issue_from_graph(item) for item in issues.get("inspection", [])
            ],
            treatment_issues=[
                _issue_from_graph(item) for item in issues.get("treatment", [])
            ],
            criteria_issues=[
                _issue_from_graph(item) for item in issues.get("criteria", [])
            ],
            guideline_file_id=file_id,
            icd_code=dx_code,
            guideline_sources=[
                _guideline_source_from_graph(source)
                for source in graph_result.get("sources", [])
            ],
            errors=list(graph_result.get("errors", [])),
        )
        return result, clinic_tokens + int(graph_result.get("tokens", 0))
