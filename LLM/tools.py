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
import logging
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from RAG.retrieval.searches import (
    search_anamnesis,
    search_by_file_id,
    search_inspection,
    search_treatment,
)
from storage.dietary_supplements_storage import DietarySupplementsStorage
from storage.drugs_storage import DrugsStorage
from storage.models.dietary_supplement import DietarySupplement
from storage.models.doc import Doc
from storage.models.drug import Drug

logger = logging.getLogger(__name__)


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


# ── Drug lookup helpers ───────────────────────────────────────────────────────

_DRUG_SCORE_THRESHOLD = 0.85


def _format_drug(drug: Drug) -> str:
    parts = [f"Торговое название: {drug.trade_name}"]
    if drug.inn_name:
        parts.append(f"МНН: {drug.inn_name}")
    if drug.dosage_form:
        parts.append(f"Форма выпуска: {drug.dosage_form}")
    if drug.dosage:
        parts.append(f"Дозировка: {drug.dosage}")
    if drug.patient_exclusions:
        parts.append(f"Исключение отдельных групп пациентов: {drug.patient_exclusions}")
    return "\n".join(parts)


def _format_supplement(s: DietarySupplement) -> str:
    parts = [f"Наименование: {s.product_name}"]
    if s.status:
        parts.append(f"Статус: {s.status}")
    if s.label_info:
        parts.append(f"Информация на этикетке: {s.label_info}")
    return "\n".join(parts)


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
    """Look up a drug or dietary supplement by name."""

    name: str = "search_medicine"
    description: str = (
        "Look up a drug or dietary supplement by trade name or INN. Just provide the name as the query without extra information. "
        "First searches the drugs registry (trigram, threshold 0.85); "
        "if nothing found, falls back to the dietary supplements registry (full-text). "
        "Use when you need to verify whether a prescribed substance is a registered drug or supplement. And when you need to get its details (dosage, form, etc.) from the registry."
    )
    args_schema: Type[BaseModel] = _QueryInput

    async def _arun(self, query: str) -> str:  # type: ignore[override]
        async with DrugsStorage() as drugs_storage:
            inn_matches = await drugs_storage.search_by_inn(query)
            if inn_matches:
                inn_name = inn_matches[0].inn_name
                logger.info(
                    '💊 Medicine lookup for "%s": found INN "%s". Not searching for drugs anymore.',
                    query,
                    inn_name,
                )
                return f'В реестре наименование было определено как действующее вещество "{inn_name}"'

            drugs = await drugs_storage.search(query, threshold=_DRUG_SCORE_THRESHOLD)

        if drugs:
            logger.info(
                '💊 Medicine lookup for "%s": found %d drug(s)',
                query,
                len(drugs),
            )
            for i, drug in enumerate(drugs, 1):
                logger.info(
                    '💊 Drug finding %d for "%s": trade_name="%s", inn="%s"',
                    i,
                    query,
                    drug.trade_name,
                    drug.inn_name,
                )
            lines = [f"Найдено в реестре лекарственных препаратов ({len(drugs)}):\n"]
            lines += [f"--- {i} ---\n{_format_drug(d)}" for i, d in enumerate(drugs, 1)]
            return "\n\n".join(lines)

        async with DietarySupplementsStorage() as supps_storage:
            supplements = await supps_storage.search(query)

        if supplements:
            logger.info(
                '💊 Medicine lookup for "%s": found %d dietary supplement(s)',
                query,
                len(supplements),
            )
            for i, supplement in enumerate(supplements, 1):
                logger.info(
                    '💊 Dietary supplement finding %d for "%s": product_name="%s", registration_number="%s"',
                    i,
                    query,
                    supplement.product_name,
                    supplement.registration_number,
                )
            lines = [f"Найдено в реестре БАД ({len(supplements)}):\n"]
            lines += [f"--- {i} ---\n{_format_supplement(s)}" for i, s in enumerate(supplements, 1)]
            return "\n\n".join(lines)

        logger.info('💊 Medicine lookup for "%s": no registry findings', query)
        return "Препарат или БАД не найден в реестрах."

    def _run(self, query: str) -> str:  # type: ignore[override]
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
