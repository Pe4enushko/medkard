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
class Result:
    """Audit result for a single ambulatory card."""

    input: dict                             # Raw JSON payload from 1C
    flags: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    # Assigned by the database on insert; None before insertion.
    id: str | None = None
