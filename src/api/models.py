"""
api/models.py — Pydantic request/response schemas for the pull API.

`pull` itself returns raw xlsx bytes (not JSON) since the integrating
service stores the file and runs it through its own RAG ingestion pipeline
— only `check`, the cheap row-count route, has a JSON contract. The
integrating service compares `count` against how many rows it ingested
from the last report and re-pulls on a mismatch.
"""

from __future__ import annotations

from pydantic import BaseModel


class CheckResponse(BaseModel):
    date: str
    count: int


class PushResponse(BaseModel):
    card_guid: str
    status: str


class DoctorEntry(BaseModel):
    code: str
    name: str


class StorageStatsResponse(BaseModel):
    """Per-organization stored-data size, in kilobytes.

    Payload size only (see StatsStorage.storage_kb): indexes and page overhead
    are excluded, so these figures are smaller than the disk a delete frees.
    """

    organization: str
    done_cards_kb: float
    audit_overwrite_journal_kb: float
    total_kb: float
