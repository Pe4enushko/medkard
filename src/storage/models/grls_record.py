from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class GrlsRecord:
    """One row of a GRLS status sheet (see migration 027)."""

    status: str
    reg_number: str
    trade_name: str
    row_hash: str
    registered_at: date | None = None
    expires_at: date | None = None
    annulled_at: date | None = None
    holder: str | None = None
    holder_country: str | None = None
    inn_name: str | None = None
    forms: list[str] = field(default_factory=list)
    forms_raw: str | None = None
    dosage_forms: list[str] = field(default_factory=list)
    dispensing: list[str] = field(default_factory=list)
    is_substance: bool = False
    production_stages: str | None = None
    normative_docs: str | None = None
    pharm_group: str | None = None
    is_vital: bool | None = None
    narcotic_list: str | None = None
    is_orphan: bool | None = None
    id: int | None = None
    imported_at: datetime | None = None


@dataclass
class GrlsImport:
    """One row of grls_imports — a registry version."""

    archive_name: str
    registry_date: date
    status_counts: dict[str, int]
    skipped_files: list[str] = field(default_factory=list)
    id: int | None = None
    imported_at: datetime | None = None
