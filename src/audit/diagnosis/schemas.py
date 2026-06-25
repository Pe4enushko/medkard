from __future__ import annotations

from pydantic import BaseModel, Field


class CheckerSource(BaseModel):
    doc_title: str = Field(default="")
    section: str | None = Field(default=None)
    cite: str | None = Field(default=None)


class CheckerIssue(BaseModel):
    issue: str = Field(default="")
    sources: list[CheckerSource] = Field(default_factory=list)


class CheckerOutput(BaseModel):
    issues: list[CheckerIssue] = Field(default_factory=list)
