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
