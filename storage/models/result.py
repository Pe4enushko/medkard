from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IssueSource:
    """A single document source reference that supports an audit issue."""

    doc_title: str          # Manifest «Наименование» value
    section: str | None = None  # TOC section title, if known

    def pretty_format(self) -> str:
        s = f"      source: {self.doc_title}"
        if self.section:
            s += f" / {self.section}"
        return s


@dataclass
class DiagnisisIssue:
    """One audit finding together with the guideline sources that support it."""

    issue: str
    sources: list[IssueSource] = field(default_factory=list)

    def pretty_format(self) -> str:
        lines = [f"    • {self.issue}"]
        lines.extend(src.pretty_format() for src in self.sources)
        return "\n".join(lines)


@dataclass
class FormalFinding:
    """One finding from the formal-structure check."""

    flag: str
    issue: str

    def pretty_format(self) -> str:
        return f"    [{self.flag}] {self.issue}"


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

    def pretty_format(self) -> str:
        if not self.findings:
            return "  Formal structure: OK"
        lines = ["  Formal structure:"]
        lines.extend(f.pretty_format() for f in self.findings)
        return "\n".join(lines)


@dataclass
class DiagnosisResult:
    """Audit result for a single diagnosis entry from the visit."""

    icd_code: str
    issues: list[DiagnisisIssue] = field(default_factory=list)

    def pretty_format(self) -> str:
        if not self.issues:
            return f"  [{self.icd_code}] OK"
        lines = [f"  [{self.icd_code}]"]
        lines.extend(iss.pretty_format() for iss in self.issues)
        return "\n".join(lines)


@dataclass
class Result:
    """Audit result for a single ambulatory card."""

    input: dict                             # Raw JSON payload from 1C
    formal: FormalStructureResult = field(default_factory=FormalStructureResult)
    diagnosis: list[DiagnosisResult] = field(default_factory=list)

    # Assigned by the database on insert; None before insertion.
    id: str | None = None

    def pretty_format(self) -> str:
        lines = [f"Result(id={self.id})"]
        lines.append(self.formal.pretty_format())
        if self.diagnosis:
            lines.append("  Diagnosis results:")
            lines.extend(dr.pretty_format() for dr in self.diagnosis)
        else:
            lines.append("  Diagnosis results: none")
        return "\n".join(lines)
