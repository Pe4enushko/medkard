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
