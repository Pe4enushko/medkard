"""
reporting/result_parser.py — turn stored done_cards JSON columns into the
typed result dataclasses (storage/models/result.py).

Shared by both audit/excel_formatter.py (Excel export) and
reporting/api_formatter.py (pull API) — this is the single place that knows
how a done_cards row's formal_result/diag_result/icd_check_result JSON maps
onto FormalStructureResult / DiagnosisResult / IcdCodingIssue.
"""

from __future__ import annotations

import csv
from pathlib import Path

from storage.models.result import DiagnosisResult, FormalFinding, FormalStructureResult, IcdCodingIssue, IssueSource

_MANIFEST_PATH = Path(__file__).resolve().parent.parent.parent / "resources" / "manifest.csv"


def load_manifest_meta() -> dict[str, dict]:
    """Return {ID: {name, date, age_group}} from manifest.csv."""
    if not _MANIFEST_PATH.exists():
        return {}
    with open(_MANIFEST_PATH, newline="", encoding="utf-8") as fh:
        return {
            row["ID"]: {
                "name": row.get("Наименование", ""),
                "date": row.get("Дата размещения", ""),
                "age_group": row.get("Возрастная категория", ""),
            }
            for row in csv.DictReader(fh)
            if row.get("ID")
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
                    )
                    for s in iss.get("sources", [])
                ],
            )
            for iss in entry.get("issues", [])
        ]
        file_id = entry.get("guideline_file_id")
        meta = manifest_meta.get(file_id) if (manifest_meta and file_id) else None
        results.append(DiagnosisResult(
            icd_code=entry["icd_code"],
            issues=issues,
            guideline_file_id=file_id,
            guideline_meta=meta,
        ))
    return results
