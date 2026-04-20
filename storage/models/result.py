from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IssueSource:
    """A single document source reference that supports an audit issue."""

    doc_title: str          # Manifest «Наименование» value
    section: str | None = None  # TOC section title, if known


@dataclass
class Issue:
    """One audit finding together with the guideline sources that support it."""

    issue: str
    sources: list[IssueSource] = field(default_factory=list)


@dataclass
class FormalFinding:
    """One finding from the formal-structure check."""

    flag: str
    issue: str


@dataclass
class FormalStructureResult:
    """All findings from the formal-structure audit dimension."""

    findings: list[FormalFinding] = field(default_factory=list)

    @property
    def flags(self) -> list[str]:
        return [f.flag for f in self.findings]

    def to_dict(self) -> dict:
        return {
            "findings": [{"flag": f.flag, "issue": f.issue} for f in self.findings]
        }


@dataclass
class Result:
    """Audit result for a single ambulatory card."""

    input: dict                             # Raw JSON payload from 1C
    formal: FormalStructureResult = field(default_factory=FormalStructureResult)
    issues: list[Issue] = field(default_factory=list)

    # Assigned by the database on insert; None before insertion.
    id: str | None = None
