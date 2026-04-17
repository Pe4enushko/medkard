"""
tools.py — LangChain tools wrapping RAG/retrieval/searches.py public API.

Tools are created with ``file_id`` baked in via ``get_tools_for(file_id)``,
so agents only need to supply ``query``.

Tools
-----
search_guideline        General hybrid search within a guideline document.
search_anamnesis        Anamnesis / complaints sections.
search_inspection       Diagnostic investigation sections.
search_treatment        Treatment / recommendations sections.

Usage::
    from LLM.tools import get_tools_for

    tools = get_tools_for("581_2")   # file_id bound at creation time
"""

from __future__ import annotations

import json
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from RAG.retrieval.searches import (
    search_anamnesis,
    search_by_file_id,
    search_inspection,
    search_treatment,
)
from storage.models.doc import Doc


# ── Input schema (query only — file_id is bound at construction) ─────────────

class _QueryInput(BaseModel):
    query: str = Field(description="Natural-language search query in Russian.")


# ── Formatter ─────────────────────────────────────────────────────────────────

def _format_results(results: list[dict]) -> str:
    """Render a list of raw search result dicts as readable text for the LLM."""
    if not results:
        return "Ничего не найдено."

    parts: list[str] = []
    for i, raw in enumerate(results, start=1):
        meta = raw.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        doc = Doc(
            chunk=raw.get("chunk", ""),
            file_id=raw.get("file_id", ""),
            metadata=meta,
            id=raw.get("id"),
            fact_q=raw.get("fact_q"),
            procedure_q=raw.get("procedure_q"),
            constraint_q=raw.get("constraint_q"),
        )
        parts.append(f"--- Источник {i} ---\n{doc._format_chunk()}")

    return "\n\n".join(parts)


# ── Tool classes (file_id set as instance attribute) ──────────────────────────

class SearchGuidelineTool(BaseTool):
    """General hybrid search scoped to the bound guideline document."""

    name: str = "search_guideline"
    description: str = (
        "Search any section of the clinical-guideline document. "
        "Use when you need broad context without section filtering."
    )
    args_schema: Type[BaseModel] = _QueryInput
    file_id: str = ""

    async def _arun(self, query: str) -> str:  # type: ignore[override]
        results = await search_by_file_id(file_id=self.file_id, query=query)
        return _format_results(results)

    def _run(self, query: str) -> str:  # type: ignore[override]
        raise NotImplementedError("Use async invocation (_arun).")


class SearchAnamnesisTool(BaseTool):
    """Search anamnesis / complaints sections of the bound guideline document."""

    name: str = "search_anamnesis"
    description: str = (
        "Search anamnesis and complaints sections of the clinical-guideline. "
        "Use to retrieve recommended criteria for patient history collection."
    )
    args_schema: Type[BaseModel] = _QueryInput
    file_id: str = ""

    async def _arun(self, query: str) -> str:  # type: ignore[override]
        results = await search_anamnesis(file_id=self.file_id, query=query)
        return _format_results(results)

    def _run(self, query: str) -> str:  # type: ignore[override]
        raise NotImplementedError("Use async invocation (_arun).")


class SearchInspectionTool(BaseTool):
    """Search diagnostic investigation sections of the bound guideline document."""

    name: str = "search_inspection"
    description: str = (
        "Search diagnostic investigation / laboratory / instrumental sections. "
        "Use to retrieve recommended examinations and diagnostic criteria."
    )
    args_schema: Type[BaseModel] = _QueryInput
    file_id: str = ""

    async def _arun(self, query: str) -> str:  # type: ignore[override]
        results = await search_inspection(file_id=self.file_id, query=query)
        return _format_results(results)

    def _run(self, query: str) -> str:  # type: ignore[override]
        raise NotImplementedError("Use async invocation (_arun).")


class SearchTreatmentTool(BaseTool):
    """Search treatment sections of the bound guideline document."""

    name: str = "search_treatment"
    description: str = (
        "Search treatment and management sections. "
        "Use to retrieve recommended treatments, medications, and care plans."
    )
    args_schema: Type[BaseModel] = _QueryInput
    file_id: str = ""

    async def _arun(self, query: str) -> str:  # type: ignore[override]
        results = await search_treatment(file_id=self.file_id, query=query)
        return _format_results(results)

    def _run(self, query: str) -> str:  # type: ignore[override]
        raise NotImplementedError("Use async invocation (_arun).")


# ── Public factories ──────────────────────────────────────────────────────────

def get_tools_for(file_id: str) -> list[BaseTool]:
    """Return all four search tools with *file_id* bound as a class attribute."""
    return [
        SearchGuidelineTool(file_id=file_id),
        SearchAnamnesisTool(file_id=file_id),
        SearchInspectionTool(file_id=file_id),
        SearchTreatmentTool(file_id=file_id),
    ]


def get_anamnesis_tools_for(file_id: str) -> list[BaseTool]:
    """Return tools for the anamnesis checker agent."""
    return [
        SearchAnamnesisTool(file_id=file_id),
        SearchGuidelineTool(file_id=file_id),
    ]


def get_inspection_tools_for(file_id: str) -> list[BaseTool]:
    """Return tools for the inspection checker agent."""
    return [
        SearchInspectionTool(file_id=file_id),
        SearchGuidelineTool(file_id=file_id),
    ]


def get_treatment_tools_for(file_id: str) -> list[BaseTool]:
    """Return tools for the treatment checker agent."""
    return [
        SearchTreatmentTool(file_id=file_id),
        SearchGuidelineTool(file_id=file_id),
    ]
