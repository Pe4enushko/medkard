"""
audit/models.py — typed result dataclasses for each audit dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# FormalFinding and FormalStructureResult live in storage.models.result to avoid
# circular imports; re-exported here for convenience.
from storage.models.result import (
    DiagnosisIssue,
    DiagnosisResult,
    FormalFinding,
    FormalStructureResult,
    GuidelineSource,
)

__all__ = ["DiagnosisAuditResult", "DiagnosisResult", "FormalFinding", "FormalStructureResult"]

# ── Formal structure (re-exported from storage.models.result) ─────────────────


# ── Diagnosis (clinical-guideline checkers) ───────────────────────────────────

@dataclass
class DiagnosisAuditResult:
    """Issues found by the three checker agents for a single diagnosis."""

    anamnesis_issues: list[DiagnosisIssue] = field(default_factory=list)
    inspection_issues: list[DiagnosisIssue] = field(default_factory=list)
    treatment_issues: list[DiagnosisIssue] = field(default_factory=list)
    criteria_issues: list[DiagnosisIssue] = field(default_factory=list)
    guideline_file_id: str | None = None
    # {name, date, age_group} той редакции клинрека, против которой шла проверка.
    # Снимается здесь и доезжает до done_cards: file_id живёт до следующей
    # редакции, а карта проверена навсегда.
    guideline_meta: dict | None = None
    icd_code: str | None = None
    guideline_sources: list[GuidelineSource] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def all_issues(self) -> list[DiagnosisIssue]:
        return (
            self.anamnesis_issues
            + self.inspection_issues
            + self.treatment_issues
            + self.criteria_issues
        )

    def to_dict(self) -> dict:
        def _issue_list(issues: list[DiagnosisIssue], aspect: str) -> list[dict]:
            return [
                {
                    "issue": iss.issue,
                    "aspect": iss.aspect or aspect,
                    "sources": [
                        {
                            "doc_title": s.doc_title,
                            "section": s.section,
                            "cite": s.cite,
                            "chunk_id": s.chunk_id,
                            "chunk_index": s.chunk_index,
                        }
                        for s in iss.sources
                    ],
                }
                for iss in issues
            ]

        return {
            "guideline_file_id": self.guideline_file_id,
            "icd_code": self.icd_code,
            "anamnesis": _issue_list(self.anamnesis_issues, "anamnesis"),
            "inspection": _issue_list(self.inspection_issues, "inspection"),
            "treatment": _issue_list(self.treatment_issues, "treatment"),
            "criteria": _issue_list(self.criteria_issues, "criteria"),
            "guideline_sources": [
                {
                    "file_id": source.file_id,
                    "doc_title": source.doc_title,
                    "sections": [
                        {
                            "section": section.section,
                            "chunk_indices": section.chunk_indices,
                            "cited": section.cited,
                        }
                        for section in source.sections
                    ],
                }
                for source in self.guideline_sources
            ],
            "errors": self.errors,
        }

    def pretty_format(self) -> str:
        if self.guideline_file_id is None:
            code = f" ({self.icd_code})" if self.icd_code else ""
            return f"Для такого МКБ кода нет прямых клинических рекоммендаций{code}"

        code = self.icd_code or "—"

        def _section(label: str, issues: list[DiagnosisIssue]) -> str:
            if not issues:
                return f"  {label}: OK"
            lines = [f"  {label}:"]
            lines.extend(iss.pretty_format() for iss in issues)
            return "\n".join(lines)

        parts = [
            f"Результат аудита диагноза(icd_code={code}, номер клин.рек.={self.guideline_file_id})",
            _section("Анамнез", self.anamnesis_issues),
            _section("Осмотр", self.inspection_issues),
            _section("Лечение", self.treatment_issues),
            _section("Критерии качества", self.criteria_issues),
        ]
        if self.errors:
            parts.append("  Деградация: " + "; ".join(self.errors))
        return "\n".join(parts)
