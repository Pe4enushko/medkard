from __future__ import annotations

from dataclasses import dataclass, field

_DELIM = "─" * 10


@dataclass
class IssueSource:
    """A single document source reference that supports an audit issue."""

    doc_title: str          # Manifest «Наименование» value
    section: str | None = None  # TOC section title, if known
    cite: str | None = None

    def pretty_format(self) -> str:
        s = f"      [ИСТОЧНИК]: {self.doc_title}"
        if self.section:
            s += f" / [раздел]{self.section}"
        if self.cite:
            s += f" — [цитата] {self.cite}"
        return s


@dataclass
class DiagnosisIssue:
    """One audit finding together with the guideline sources that support it."""

    issue: str
    sources: list[IssueSource] = field(default_factory=list)

    def pretty_format(self) -> str:
        lines = [f"    • [ЗАМЕЧАНИЕ] {self.issue}"]
        lines.extend(src.pretty_format() for src in self.sources)
        return "\n".join(lines)


@dataclass
class FormalFinding:
    """One finding from the formal-structure check."""

    flag: str
    issue: str
    source: str = ""
    comment: str = ""

    def pretty_format(self) -> str:
        s = f"    [{self.flag}] {self.issue}"
        if self.comment:
            s += f"\n    [Наблюдения]: {self.comment}"
        if self.source:
            s += f"\n    [Источник: {self.source}]"
        return s

    def to_dict(self) -> dict:
        return {"flag": self.flag, "issue": self.issue, "source": self.source, "comment": self.comment}


@dataclass
class FormalStructureResult:
    """All findings from the formal-structure audit dimension."""

    findings: list[FormalFinding] = field(default_factory=list)

    @property
    def flags(self) -> list[str]:
        return [f.flag for f in self.findings]

    def to_dict(self) -> dict:
        return {"findings": [f.to_dict() for f in self.findings]}

    def pretty_format(self) -> str:
        if not self.findings:
            return "Замечаний по формальной структуре нет"
        parts = [f.pretty_format() for f in self.findings]
        return f"\n{_DELIM}\n".join(parts)


@dataclass
class DiagnosisResult:
    """Audit result for a single diagnosis entry from the visit."""

    icd_code: str
    issues: list[DiagnosisIssue] = field(default_factory=list)
    guideline_file_id: str | None = None
    guideline_meta: dict | None = None  # {name, date, age_group} — populated at export time, not stored in DB

    def pretty_format(self) -> str:
        if self.guideline_file_id is None:
            return f"[{self.icd_code}] Клинические рекоммендации для данного кода не найдены"

        header_parts = [f"[{self.icd_code}]"]
        if self.guideline_meta:
            m = self.guideline_meta
            header_parts.append(
                f"Клинические рекомендации: {m.get('name', '—')} "
                f"(ID: {self.guideline_file_id}, дата: {m.get('date', '—')}, "
                f"категория: {m.get('age_group', '—')})"
            )
        else:
            header_parts.append(f"Клинические рекомендации ID: {self.guideline_file_id}")

        if not self.issues:
            header_parts.append("Замечаний по клин. рекоммендациям нет")
            return "\n".join(header_parts)

        issue_blocks = [iss.pretty_format() for iss in self.issues]
        body = f"\n{_DELIM}\n".join(issue_blocks)
        return "\n".join(header_parts) + "\n" + body


@dataclass
class Result:
    """Audit result for a single ambulatory card."""

    input: dict                             # Raw JSON payload from 1C
    formal: FormalStructureResult = field(default_factory=FormalStructureResult)
    diagnosis: list[DiagnosisResult] = field(default_factory=list)

    # Assigned by the database on insert; None before insertion.
    id: str | None = None
    token_count: int = 0

    def pretty_format(self) -> str:
        lines = [f"Result(id={self.id})"]
        lines.append(self.formal.pretty_format())
        if self.diagnosis:
            lines.append("  Diagnosis results:")
            lines.extend(dr.pretty_format() for dr in self.diagnosis)
        else:
            lines.append("  Diagnosis results: none")
        return "\n".join(lines)
