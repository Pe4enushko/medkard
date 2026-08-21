"""Registration-certificate statuses and their meaning relative to a visit date.

Status is the truth (taken from the sheet the row came from); dates never
override it — they only soften a dead status when the certificate was still
valid on the visit date. See spec §5.1.
"""
from __future__ import annotations

from datetime import date
from enum import Enum

from storage.models.grls_record import GrlsRecord

STATUS_ACTIVE = "Действующий"
STATUS_EAEU = "Выдано по правилам ЕАЭС"
STATUS_CONFIRMING = "Действует, на подтверждении государственной регистрации"
STATUS_FOREIGN_PACK = "Действует, в иностранных упаковках"
STATUS_SUSPENDED = "Приостановлено применение"
STATUS_EXPIRED = "Истёкший"
STATUS_ANNULLED = "Исключённый"
STATUS_CHANGED = "Изменённый"  # revision journal, not loaded

ALL_STATUSES: tuple[str, ...] = (
    STATUS_ACTIVE, STATUS_EAEU, STATUS_CONFIRMING, STATUS_FOREIGN_PACK,
    STATUS_SUSPENDED, STATUS_EXPIRED, STATUS_ANNULLED,
)
LIVE_STATUSES: frozenset[str] = frozenset(
    {STATUS_ACTIVE, STATUS_EAEU, STATUS_CONFIRMING, STATUS_FOREIGN_PACK})
STATUS_RANK: dict[str, int] = {
    STATUS_ACTIVE: 0, STATUS_EAEU: 0, STATUS_CONFIRMING: 0, STATUS_FOREIGN_PACK: 0,
    STATUS_SUSPENDED: 1, STATUS_EXPIRED: 2, STATUS_ANNULLED: 3,
}


class StatusAtVisit(str, Enum):
    ACTIVE = "active"
    ACTIVE_WITH_NOTE = "active_note"      # confirming / foreign pack / suspended
    VALID_AT_VISIT = "valid_at_visit"     # expired/annulled now, but valid on the visit date
    EXPIRED = "expired"
    ANNULLED = "annulled"
    UNKNOWN_END = "unknown_end"           # dead status without a usable boundary date


def status_at(record: GrlsRecord, on: date | None) -> StatusAtVisit:
    """Interpret record.status relative to visit date `on` (None = no softening)."""
    status = record.status
    if status in (STATUS_ACTIVE, STATUS_EAEU):
        return StatusAtVisit.ACTIVE
    if status in (STATUS_CONFIRMING, STATUS_FOREIGN_PACK, STATUS_SUSPENDED):
        return StatusAtVisit.ACTIVE_WITH_NOTE
    if status == STATUS_EXPIRED:
        boundary = record.expires_at
        dead = StatusAtVisit.EXPIRED
    elif status == STATUS_ANNULLED:
        boundary = record.annulled_at or record.expires_at
        dead = StatusAtVisit.ANNULLED
    else:
        raise ValueError(f"unknown GRLS status: {status!r}")
    if boundary is None:
        return StatusAtVisit.UNKNOWN_END
    if on is not None and boundary >= on:
        return StatusAtVisit.VALID_AT_VISIT
    return dead
