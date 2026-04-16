from dataclasses import dataclass, field


@dataclass
class Source:
    """A single document source reference within a clinical finding."""

    file: str           # Manifest ID, e.g. "340_2"
    file_metadata: dict # Full manifest.csv row for that file
    page: int           # 0-based page index
    section: str | None = None  # TOC section title, if known


@dataclass
class ClinicalSource:
    """One invalidation flag together with the document sources that support it."""

    flag: str
    sources: list[Source] = field(default_factory=list)


@dataclass
class Result:
    """Audit result for a single ambulatory card."""

    input: dict                                   # Raw JSON payload from 1C
    flags: list[str] = field(default_factory=list)
    clinical_sources: list[ClinicalSource] = field(default_factory=list)

    # Assigned by the database on insert; None before insertion.
    id: str | None = None
