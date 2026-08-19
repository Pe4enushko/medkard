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
    get_section_chunks,
    get_sections_for_file,
    search_anamnesis,
    search_by_file_id,
    search_inspection,
    search_treatment,
)
from grls.format import format_medicine_lookup
from grls.lookup import lookup_medicine
from storage.models.doc import Doc


# ── Input schema (query only — file_id is bound at construction) ─────────────

class _QueryInput(BaseModel):
    query: str = Field(description="Natural-language search query in Russian.")


# ── Formatter ─────────────────────────────────────────────────────────────────

def _format_results(results: list[dict]) -> str:
    """Render a list of raw search result dicts as readable text for the LLM."""
    if not results:
        return "Ничего не найдено. Не повторяй этот запрос — попробуй другой или переходи к выводу."

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


class SearchMedicineTool(BaseTool):
    """Look up a drug (GRLS, with certificate status) or a dietary supplement by name."""

    name: str = "search_medicine"
    description: str = (
        "Поиск препарата в ГРЛС (Государственный реестр лекарственных средств) по торговому "
        "названию или МНН, с фолбэком в реестр БАД. Передавай только название, без лишних слов. "
        "Ответ содержит статус регистрационного удостоверения (действующее / истёкшее / "
        "аннулированное / с предупреждением), лекарственные формы и условия отпуска. "
        "Используй, чтобы связать торговое название с действующим веществом и проверить, "
        "что назначенный препарат зарегистрирован."
    )
    args_schema: Type[BaseModel] = _QueryInput

    async def _arun(self, query: str) -> str:  # type: ignore[override]
        return format_medicine_lookup(await lookup_medicine(query))

    def _run(self, query: str) -> str:  # type: ignore[override]
        raise NotImplementedError("Use async invocation (_arun).")


# ── ICD checker tools (file_id supplied as agent input, not bound at construction) ──

class _GuidelineStructureInput(BaseModel):
    file_id: str = Field(description="Guideline file ID from the manifest (e.g. '581_2').")


class _SectionReadInput(BaseModel):
    file_id: str = Field(description="Guideline file ID from the manifest.")
    section: str = Field(description="Exact section name as returned by get_guideline_structure.")


class GetGuidelineStructureTool(BaseTool):
    """Return the ordered list of section names (TOC) for a guideline document."""

    name: str = "get_guideline_structure"
    description: str = (
        "Get the table of contents (ordered section names) for a clinical guideline. "
        "Use this first to see which sections exist before deciding what to read."
    )
    args_schema: Type[BaseModel] = _GuidelineStructureInput

    async def _arun(self, file_id: str) -> str:  # type: ignore[override]
        sections = await get_sections_for_file(file_id)
        if not sections:
            return f"Документ '{file_id}' не найден или не содержит разделов."
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sections, 1))
        return f"Разделы документа {file_id}:\n{numbered}"

    def _run(self, file_id: str) -> str:  # type: ignore[override]
        raise NotImplementedError("Use async invocation (_arun).")


class ReadGuidelineSectionTool(BaseTool):
    """Read all chunks of a specific section from a guideline document in order."""

    name: str = "read_guideline_section"
    description: str = (
        "Read the full text of a specific section from a clinical guideline, "
        "chunk by chunk in document order. Use after get_guideline_structure to "
        "read sections 1.x (definition, classification, diagnostic criteria)."
    )
    args_schema: Type[BaseModel] = _SectionReadInput

    async def _arun(self, file_id: str, section: str) -> str:  # type: ignore[override]
        chunks = await get_section_chunks(file_id, section)
        if not chunks:
            return f"Раздел '{section}' в документе '{file_id}' не найден."
        import json as _json
        parts: list[str] = []
        for raw in chunks:
            meta = raw.get("metadata") or {}
            if isinstance(meta, str):
                try:
                    meta = _json.loads(meta)
                except _json.JSONDecodeError:
                    meta = {}
            from storage.models.doc import Doc
            doc = Doc(
                chunk=raw.get("chunk", ""),
                file_id=raw.get("file_id", file_id),
                metadata=meta,
                id=raw.get("id"),
            )
            parts.append(doc._format_chunk())
        return "\n\n".join(parts)

    def _run(self, file_id: str, section: str) -> str:  # type: ignore[override]
        raise NotImplementedError("Use async invocation (_arun).")


# ── Public factories ──────────────────────────────────────────────────────────

def get_tools_for(file_id: str) -> list[BaseTool]:
    """Return all search tools with *file_id* bound as a class attribute."""
    return [
        SearchGuidelineTool(file_id=file_id),
        SearchAnamnesisTool(file_id=file_id),
        SearchInspectionTool(file_id=file_id),
        SearchTreatmentTool(file_id=file_id),
        SearchMedicineTool(),
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
        SearchMedicineTool(),
    ]


def get_icd_checker_tools() -> list[BaseTool]:
    """Return tools for the ICD checker agent.

    Unlike the clinical checker tools, these are not bound to a specific file_id —
    the agent supplies file_id as a tool argument based on the manifest table in context.
    """
    return [
        GetGuidelineStructureTool(),
        ReadGuidelineSectionTool(),
    ]
