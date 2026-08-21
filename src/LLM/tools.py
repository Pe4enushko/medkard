"""LangChain tools retained for the ICD ReAct checker.

The diagnosis-guideline audit uses the deterministic graph in ``LLM.graphs``
and calls retrieval/storage directly, so it has no tool factory here.
"""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from RAG.retrieval.searches import get_section_chunks, get_sections_for_file
from storage.models.doc import Doc


class _GuidelineStructureInput(BaseModel):
    file_id: str = Field(description="Guideline file ID from the manifest (e.g. '581_2').")


class _SectionReadInput(BaseModel):
    file_id: str = Field(description="Guideline file ID from the manifest.")
    section: str = Field(description="Exact section name returned by get_guideline_structure.")


class GetGuidelineStructureTool(BaseTool):
    """Return the ordered table of contents of one guideline."""

    name: str = "get_guideline_structure"
    description: str = (
        "Get ordered clinical-guideline section names. Use before reading a section."
    )
    args_schema: type[BaseModel] = _GuidelineStructureInput

    async def _arun(self, file_id: str) -> str:  # type: ignore[override]
        sections = await get_sections_for_file(file_id)
        if not sections:
            return f"Документ '{file_id}' не найден или не содержит разделов."
        numbered = "\n".join(f"{index}. {section}" for index, section in enumerate(sections, 1))
        return f"Разделы документа {file_id}:\n{numbered}"

    def _run(self, file_id: str) -> str:  # type: ignore[override]
        raise NotImplementedError("Use async invocation (_arun).")


class ReadGuidelineSectionTool(BaseTool):
    """Read every chunk of an exact guideline section in document order."""

    name: str = "read_guideline_section"
    description: str = (
        "Read one exact clinical-guideline section chunk by chunk. "
        "Use after get_guideline_structure."
    )
    args_schema: type[BaseModel] = _SectionReadInput

    async def _arun(self, file_id: str, section: str) -> str:  # type: ignore[override]
        chunks = await get_section_chunks(file_id, section)
        if not chunks:
            return f"Раздел '{section}' в документе '{file_id}' не найден."

        parts: list[str] = []
        for raw in chunks:
            metadata = raw.get("metadata") or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            doc = Doc(
                chunk=raw.get("chunk", ""),
                file_id=raw.get("file_id", file_id),
                metadata=metadata,
                id=raw.get("id"),
            )
            parts.append(doc._format_chunk())
        return "\n\n".join(parts)

    def _run(self, file_id: str, section: str) -> str:  # type: ignore[override]
        raise NotImplementedError("Use async invocation (_arun).")


def get_icd_checker_tools() -> list[BaseTool]:
    return [GetGuidelineStructureTool(), ReadGuidelineSectionTool()]
