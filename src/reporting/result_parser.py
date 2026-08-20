"""
reporting/result_parser.py — turn stored done_cards JSON columns into the
typed result dataclasses (storage/models/result.py).

Shared by both audit/excel_formatter.py (Excel export) and
reporting/api_formatter.py (pull API) — this is the single place that knows
how a done_cards row's formal_result/diag_result/icd_check_result JSON maps
onto FormalStructureResult / DiagnosisResult / IcdCodingIssue.
"""

from __future__ import annotations

from storage.models.guideline import Guideline
from storage.models.result import (
    DiagnosisResult,
    FormalFinding,
    FormalStructureResult,
    GuidelineSource,
    GuidelineSourceSection,
    IcdCodingIssue,
    IssueSource,
)


def build_manifest_meta(guidelines: list[Guideline]) -> dict[str, dict]:
    """Return {file_id: {name, date, age_group}} from Guideline objects."""
    return {
        g.file_id: {
            "name": g.name or "",
            "date": g.published_at or "",
            "age_group": ", ".join(g.age_category),
        }
        for g in guidelines
        if g.file_id
    }


def parse_formal(data: list[dict]) -> FormalStructureResult:
    return FormalStructureResult(
        findings=[
            FormalFinding(flag=f["flag"], issue=f.get("issue", ""), source=f.get("source", ""), comment=f.get("comment", ""))
            for f in (data or [])
        ]
    )


def parse_icd_check(data: list[dict] | None) -> list[IcdCodingIssue]:
    issues = []
    for entry in (data or []):
        sources = [
            IssueSource(
                doc_title=s.get("doc_title", ""),
                section=s.get("section"),
                cite=s.get("cite"),
            )
            for s in entry.get("sources", [])
            if isinstance(s, dict)
        ]
        issues.append(IcdCodingIssue(
            dx_index=entry.get("dx_index", 0),
            initial_code=entry.get("initial_code", ""),
            suggested_code=entry.get("suggested_code", ""),
            confidence=entry.get("confidence", 0),
            comment=entry.get("comment", ""),
            sources=sources,
        ))
    return issues


def parse_diagnosis(data: list[dict], manifest_meta: dict[str, dict] | None = None) -> list[DiagnosisResult]:
    from storage.models.result import DiagnosisIssue, IssueSource

    results = []
    for entry in (data or []):
        issues = [
            DiagnosisIssue(
                issue=iss["issue"],
                sources=[
                    IssueSource(
                        doc_title=s["doc_title"],
                        section=s.get("section"),
                        cite=s.get("cite"),
                        chunk_id=s.get("chunk_id"),
                        chunk_index=s.get("chunk_index"),
                    )
                    for s in iss.get("sources", [])
                ],
                aspect=iss.get("aspect"),
            )
            for iss in entry.get("issues", [])
        ]
        file_id = entry.get("guideline_file_id")
        meta = manifest_meta.get(file_id) if (manifest_meta and file_id) else None
        guideline_sources = [
            GuidelineSource(
                file_id=source.get("file_id", ""),
                doc_title=source.get("doc_title", ""),
                sections=[
                    GuidelineSourceSection(
                        section=section.get("section"),
                        chunk_indices=list(section.get("chunk_indices") or []),
                        cited=bool(section.get("cited", False)),
                    )
                    for section in source.get("sections", [])
                    if isinstance(section, dict)
                ],
            )
            for source in entry.get("guideline_sources", [])
            if isinstance(source, dict)
        ]
        results.append(DiagnosisResult(
            icd_code=entry["icd_code"],
            issues=issues,
            guideline_file_id=file_id,
            guideline_meta=meta,
            guideline_sources=guideline_sources,
            errors=list(entry.get("errors") or []),
        ))
    return results
