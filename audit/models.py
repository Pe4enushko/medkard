"""
audit/models.py — typed result dataclasses for each audit dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# FormalFinding and FormalStructureResult live in storage.models.result to avoid
# circular imports; re-exported here for convenience.
from storage.models.result import FormalFinding, FormalStructureResult, DiagnisisIssue

__all__ = ["FormalFinding", "FormalStructureResult", "DiagnosisAuditResult"]

# ── Formal structure (re-exported from storage.models.result) ─────────────────


# ── Diagnosis (clinical-guideline checkers) ───────────────────────────────────

@dataclass
class DiagnosisAuditResult:
    """Issues found by the three checker agents for a single diagnosis."""

    anamnesis_issues: list[DiagnisisIssue] = field(default_factory=list)
    inspection_issues: list[DiagnisisIssue] = field(default_factory=list)
    treatment_issues: list[DiagnisisIssue] = field(default_factory=list)
    guideline_file_id: str | None = None

    @property
    def all_issues(self) -> list[DiagnisisIssue]:
        return self.anamnesis_issues + self.inspection_issues + self.treatment_issues

    def to_dict(self) -> dict:
        def _issue_list(issues: list[DiagnisisIssue]) -> list[dict]:
            return [
                {
                    "issue": iss.issue,
                    "sources": [
                        {"doc_title": s.doc_title, "section": s.section}
                        for s in iss.sources
                    ],
                }
                for iss in issues
            ]

        return {
            "guideline_file_id": self.guideline_file_id,
            "anamnesis": _issue_list(self.anamnesis_issues),
            "inspection": _issue_list(self.inspection_issues),
            "treatment": _issue_list(self.treatment_issues),
        }

    def pretty_format(self) -> str:
        def _section(label: str, issues: list[DiagnisisIssue]) -> str:
            if not issues:
                return f"  {label}: OK"
            lines = [f"  {label}:"]
            lines.extend(iss.pretty_format() for iss in issues)
            return "\n".join(lines)

        parts = [
            f"DiagnosisAuditResult(guideline={self.guideline_file_id})",
            _section("Anamnesis", self.anamnesis_issues),
            _section("Inspection", self.inspection_issues),
            _section("Treatment", self.treatment_issues),
        ]
        return "\n".join(parts)
