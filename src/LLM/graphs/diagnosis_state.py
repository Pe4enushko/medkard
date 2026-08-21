from __future__ import annotations

import operator
from datetime import date
from typing import Annotated, Literal, TypedDict, TypeVar

from pydantic import BaseModel, Field

Aspect = Literal["anamnesis", "inspection", "treatment", "criteria"]
QuestionAspect = Literal["anamnesis", "inspection", "treatment"]


class Question(TypedDict):
    aspect: QuestionAspect
    text: str


class DrugMention(TypedDict):
    as_written: str
    normalized: str


class Chunk(TypedDict):
    ref: int
    id: str
    file_id: str
    doc_title: str
    section: str | None
    chunk_index: int | None
    content_type: str
    page: int | None
    table_index: int | None
    text: str
    rrf_score: float
    rerank_score: float | None
    questions: list[str]


class ResolvedSource(TypedDict):
    doc_title: str
    section: str | None
    cite: str | None
    chunk_id: str | None
    chunk_index: int | None


class ResolvedIssue(TypedDict):
    aspect: Aspect
    issue: str
    sources: list[ResolvedSource]


class GuidelineSourceSection(TypedDict):
    section: str | None
    chunk_indices: list[int]
    cited: bool


class GuidelineSource(TypedDict):
    file_id: str
    doc_title: str
    sections: list[GuidelineSourceSection]


class QuestionSet(BaseModel):
    anamnesis: list[str] = Field(min_length=1)
    inspection: list[str] = Field(min_length=1)
    treatment: list[str] = Field(min_length=1)


class DrugMentionOutput(BaseModel):
    as_written: str
    normalized: str


class DrugList(BaseModel):
    items: list[DrugMentionOutput] = Field(default_factory=list)


class JudgeIssue(BaseModel):
    issue: str
    chunk_refs: list[int] = Field(min_length=1)


class JudgeOutput(BaseModel):
    issues: list[JudgeIssue] = Field(default_factory=list)


_K = TypeVar("_K")
_V = TypeVar("_V")


def merge_dicts(left: dict[_K, _V], right: dict[_K, _V]) -> dict[_K, _V]:
    """Merge independent parallel-node updates without mutating either input."""
    return {**left, **right}


class DiagnosisState(TypedDict, total=False):
    visit_context: str
    patient_block: str
    diagnosis_block: str
    visit_date: date | None
    file_id: str
    doc_title: str
    toc: list[str]
    card_guid: str | None
    correlation_id: str
    dx_code: str

    questions: list[Question]
    drug_mentions: list[DrugMention]
    drug_context: str
    pools: Annotated[dict[Aspect, list[Chunk]], merge_dicts]

    issues: Annotated[dict[Aspect, list[ResolvedIssue]], merge_dicts]
    sources: list[GuidelineSource]
    errors: Annotated[list[str], operator.add]
    tokens: Annotated[int, operator.add]
