from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from LLM.graphs.diagnosis_state import (
    Aspect,
    Chunk,
    DiagnosisState,
    DrugList,
    GuidelineSource,
    GuidelineSourceSection,
    JudgeOutput,
    Question,
    QuestionSet,
    ResolvedIssue,
    ResolvedSource,
)

logger = logging.getLogger(__name__)

ASPECT_POOL_MAX_CHUNKS = int(os.environ.get("DIAG_ASPECT_POOL_MAX_CHUNKS", "20"))
CITE_MAX_CHARS = int(os.environ.get("DIAG_CITE_MAX_CHARS", "300"))
QUESTIONS_PER_ASPECT_MAX = int(os.environ.get("DIAG_QUESTIONS_PER_ASPECT_MAX", "4"))
CANDIDATES_PER_QUESTION = int(os.environ.get("DIAG_CANDIDATES_PER_QUESTION", "40"))
TOP_K_PER_QUESTION = int(os.environ.get("DIAG_TOP_K_PER_QUESTION", "5"))
RETRIEVE_CONCURRENCY = int(os.environ.get("DIAG_RETRIEVE_CONCURRENCY", "8"))
CRITERIA_SECTION_PATTERN = os.environ.get(
    "DIAG_CRITERIA_SECTION_PATTERN",
    "%критерии оценки качества%",
)
CRITERIA_MAX_CHUNKS = int(os.environ.get("DIAG_CRITERIA_MAX_CHUNKS", "8"))

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class StructuredClient(Protocol):
    async def call(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        response_model: type[BaseModel] | None = None,
        reasoning_effort: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str | BaseModel, int]: ...


def _default_client() -> StructuredClient:
    from LLM.client import LLMClient

    return LLMClient()


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _validated_output(raw: str | BaseModel, model: type[BaseModel]) -> BaseModel:
    if isinstance(raw, model):
        return raw
    if isinstance(raw, BaseModel):
        return model.model_validate(raw.model_dump())
    return model.model_validate_json(raw)


def _error(label: str, exc: Exception) -> str:
    detail = str(exc).strip() or type(exc).__name__
    return f"{label}: {detail[:300]}"


def _fallback_questions(diagnosis_block: str) -> list[Question]:
    diagnosis = diagnosis_block.strip() or "указанном диагнозе"
    return [
        {
            "aspect": "anamnesis",
            "text": f"Какие жалобы и данные анамнеза обязательны при {diagnosis}?",
        },
        {
            "aspect": "anamnesis",
            "text": f"Какие факторы риска следует уточнить при {diagnosis}?",
        },
        {
            "aspect": "inspection",
            "text": f"Какие осмотры и исследования необходимы при {diagnosis}?",
        },
        {
            "aspect": "inspection",
            "text": f"Какие диагностические критерии применяются при {diagnosis}?",
        },
        {
            "aspect": "treatment",
            "text": f"Какая стартовая терапия рекомендована при {diagnosis}?",
        },
        {
            "aspect": "treatment",
            "text": f"Когда требуется изменить лечение при {diagnosis}?",
        },
    ]


def _questions_from_output(output: QuestionSet) -> list[Question]:
    questions: list[Question] = []
    for aspect in ("anamnesis", "inspection", "treatment"):
        seen: set[str] = set()
        for raw in getattr(output, aspect):
            text = raw.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            questions.append({"aspect": aspect, "text": text})
            if len(seen) >= QUESTIONS_PER_ASPECT_MAX:
                break
    return questions


async def generate_questions(
    state: DiagnosisState,
    *,
    client: StructuredClient | None = None,
) -> dict[str, Any]:
    client = client or _default_client()
    tokens = 0
    user = (
        f"## Пациент\n{state.get('patient_block', '—')}\n\n"
        f"## Диагноз\n{state.get('diagnosis_block', '—')}\n\n"
        f"## Клинический контекст записи\n{state.get('visit_context', '—')}\n\n"
        "## Оглавление клинических рекомендаций\n"
        + "\n".join(f"- {section}" for section in state.get("toc", []))
    )
    try:
        raw, tokens = await client.call(
            [
                {"role": "system", "content": _load_prompt("diagnosis_questions.txt")},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            response_model=QuestionSet,
            reasoning_effort="low",
            metadata={
                "node": "generate_questions",
                "card_guid": state.get("card_guid"),
                "dx_code": state.get("dx_code"),
            },
        )
        output = _validated_output(raw, QuestionSet)
        assert isinstance(output, QuestionSet)
        questions = _questions_from_output(output)
        if not questions:
            raise ValueError("model returned no questions")
        return {"questions": questions, "tokens": tokens}
    except Exception as exc:  # noqa: BLE001 - node-level degradation is deliberate
        return {
            "questions": _fallback_questions(state.get("diagnosis_block", "")),
            "errors": [_error("generate_questions: fallback templates", exc)],
            "tokens": tokens,
        }


async def extract_drugs(
    state: DiagnosisState,
    *,
    client: StructuredClient | None = None,
) -> dict[str, Any]:
    client = client or _default_client()
    tokens = 0
    try:
        raw, tokens = await client.call(
            [
                {"role": "system", "content": _load_prompt("diagnosis_drugs.txt")},
                {"role": "user", "content": state.get("visit_context", "—")},
            ],
            temperature=0.0,
            response_model=DrugList,
            reasoning_effort="low",
            metadata={
                "node": "extract_drugs",
                "card_guid": state.get("card_guid"),
                "dx_code": state.get("dx_code"),
            },
        )
        output = _validated_output(raw, DrugList)
        assert isinstance(output, DrugList)
        mentions = [
            {
                "as_written": item.as_written.strip(),
                "normalized": item.normalized.strip(),
            }
            for item in output.items
            if item.as_written.strip() and item.normalized.strip()
        ]
        return {"drug_mentions": mentions, "tokens": tokens}
    except Exception as exc:  # noqa: BLE001 - node-level degradation is deliberate
        return {
            "drug_mentions": [],
            "errors": [_error("extract_drugs", exc)],
            "tokens": tokens,
        }


async def _legacy_medicine_lookup(query: str, on: date | None = None) -> str:
    del on  # The legacy ESKLP table has no registration history.
    from storage.dietary_supplements_storage import DietarySupplementsStorage
    from storage.drugs_storage import DrugsStorage

    async with DrugsStorage() as storage:
        inn_matches = await storage.search_by_inn(query)
        if inn_matches:
            inn = inn_matches[0].inn_name or query
            return f"МНН: {inn} (реестр ЕСКЛП без статуса РУ)"
        drugs = await storage.search(query, threshold=0.85)
    if drugs:
        rendered = []
        for drug in drugs:
            details = [drug.trade_name]
            if drug.inn_name:
                details.append(f"МНН {drug.inn_name}")
            if drug.dosage_form:
                details.append(drug.dosage_form)
            rendered.append(", ".join(details))
        return "ЕСКЛП: " + "; ".join(rendered)

    async with DietarySupplementsStorage() as storage:
        supplements = await storage.search(query)
    if supplements:
        return "БАД: " + "; ".join(
            supplement.product_name
            + (
                f" ({supplement.registration_number})"
                if supplement.registration_number
                else ""
            )
            for supplement in supplements
        )
    return "не найден в реестрах"


async def _default_medicine_lookup(query: str, on: date | None = None) -> str:
    try:
        from grls.format import format_medicine_lookup
        from grls.lookup import lookup_medicine
    except ModuleNotFoundError:
        return await _legacy_medicine_lookup(query, on)
    try:
        return format_medicine_lookup(await lookup_medicine(query, on=on))
    except Exception as exc:
        # The graph can be deployed before migration 027 even when the GRLS
        # package is already present. Keep using the old registry during that
        # transition instead of losing the whole medicine context.
        from psycopg.errors import UndefinedTable

        if isinstance(exc, UndefinedTable):
            logger.info("[diagnosis_graph] GRLS tables are absent; using legacy drugs")
            return await _legacy_medicine_lookup(query, on)
        raise


async def lookup_drugs(
    state: DiagnosisState,
    *,
    lookup=None,
) -> dict[str, Any]:
    mentions = state.get("drug_mentions", [])
    if not mentions:
        return {"drug_context": ""}
    lookup = lookup or _default_medicine_lookup
    try:
        lines = []
        for mention in mentions:
            result = await lookup(mention["normalized"], on=state.get("visit_date"))
            lines.append(f"- {mention['as_written']} → {result}")
    except Exception as exc:  # noqa: BLE001 - registry failure must not drop the card
        return {
            "drug_context": "справка недоступна",
            "errors": [_error("lookup_drugs", exc)],
        }

    visit_date = state.get("visit_date")
    prefix = f"Дата визита: {visit_date.isoformat()}\n" if visit_date else ""
    return {"drug_context": prefix + "\n".join(lines)}


async def retrieve(
    state: DiagnosisState,
    *,
    search=None,
) -> dict[str, Any]:
    if search is None:
        from RAG.retrieval.searches import search_in_guideline

        search = search_in_guideline

    semaphore = asyncio.Semaphore(RETRIEVE_CONCURRENCY)

    async def _one(question: Question):
        async with semaphore:
            try:
                rows = await search(
                    question["text"],
                    state["file_id"],
                    candidates=CANDIDATES_PER_QUESTION,
                    top_k=TOP_K_PER_QUESTION,
                )
                return question, rows, None
            except Exception as exc:  # noqa: BLE001 - isolate one retrieval question
                return question, [], exc

    results = await asyncio.gather(
        *[_one(question) for question in state.get("questions", [])]
    )
    rows_by_aspect: dict[Aspect, list[tuple[str, list[dict[str, Any]]]]] = {
        "anamnesis": [],
        "inspection": [],
        "treatment": [],
        "criteria": [],
    }
    errors: list[str] = []
    for question, rows, exc in results:
        if exc is not None:
            errors.append(_error(f"retrieve_{question['aspect']}", exc))
            continue
        rows_by_aspect[question["aspect"]].append((question["text"], rows))

    pools = {
        aspect: build_aspect_pool(
            rows_by_aspect[aspect],
            file_id=state["file_id"],
            doc_title=state["doc_title"],
        )
        for aspect in ("anamnesis", "inspection", "treatment")
    }
    for aspect in ("anamnesis", "inspection", "treatment"):
        if not pools[aspect]:
            errors.append(f"retrieve_{aspect}: no chunks")
    return {"pools": pools, "errors": errors}


async def retrieve_criteria(
    state: DiagnosisState,
    *,
    get_chunks=None,
) -> dict[str, Any]:
    if get_chunks is None:
        from RAG.retrieval.searches import get_section_chunks_by_pattern

        get_chunks = get_section_chunks_by_pattern
    try:
        rows = await get_chunks(
            state["file_id"],
            CRITERIA_SECTION_PATTERN,
            CRITERIA_MAX_CHUNKS,
        )
    except Exception as exc:  # noqa: BLE001 - criteria is an optional graph branch
        return {
            "pools": {"criteria": []},
            "errors": [_error("retrieve_criteria", exc)],
        }
    if not rows:
        return {
            "pools": {"criteria": []},
            "errors": ["retrieve_criteria: section not found"],
        }
    pool = build_aspect_pool(
        [("", rows)],
        file_id=state["file_id"],
        doc_title=state["doc_title"],
        limit=CRITERIA_MAX_CHUNKS,
    )
    for chunk in pool:
        chunk["questions"] = []
    return {"pools": {"criteria": pool}}


def _render_pool(pool: list[Chunk]) -> str:
    parts: list[str] = []
    for chunk in pool:
        chunk_index = chunk["chunk_index"] if chunk["chunk_index"] is not None else "—"
        location = f"{chunk['section'] or 'раздел не указан'} | фрагмент {chunk_index}"
        if chunk["page"] is not None:
            location += f" | стр. {chunk['page']}"
        parts.append(f"[{chunk['ref']}] {location}\n{chunk['text']}")
    return "\n\n".join(parts)


async def judge_aspect(
    state: DiagnosisState,
    aspect: Aspect,
    *,
    client: StructuredClient | None = None,
    detector=None,
) -> dict[str, Any]:
    pool = state.get("pools", {}).get(aspect, [])
    if not pool:
        return {"issues": {aspect: []}}

    client = client or _default_client()
    prompt_name = {
        "anamnesis": "anamnesis_checker.txt",
        "inspection": "inspection_checker.txt",
        "treatment": "treatment_checker.txt",
        "criteria": "criteria_checker.txt",
    }[aspect]
    context_parts = [
        f"## Пациент\n{state.get('patient_block', '—')}",
        f"## Диагноз\n{state.get('diagnosis_block', '—')}",
        f"## Клинический контекст записи\n{state.get('visit_context', '—')}",
    ]
    if aspect == "treatment":
        context_parts.append(
            f"## Справка по препаратам\n{state.get('drug_context', '—')}"
        )
    heading = (
        "Критерии оценки качества"
        if aspect == "criteria"
        else "Фрагменты клинических рекомендаций"
    )
    context_parts.append(
        f"## {heading} «{state.get('doc_title', '')}»\n{_render_pool(pool)}"
    )

    tokens = 0
    try:
        raw, tokens = await client.call(
            [
                {"role": "system", "content": _load_prompt(prompt_name)},
                {"role": "user", "content": "\n\n".join(context_parts)},
            ],
            temperature=0.1,
            response_model=JudgeOutput,
            reasoning_effort="low",
            metadata={
                "node": f"judge_{aspect}",
                "card_guid": state.get("card_guid"),
                "dx_code": state.get("dx_code"),
            },
        )
        output = _validated_output(raw, JudgeOutput)
        assert isinstance(output, JudgeOutput)
        issues = resolve_judge_output(output, aspect=aspect, pool=pool)
    except Exception as exc:  # noqa: BLE001 - malformed structured output degrades one judge
        return {
            "issues": {aspect: []},
            "errors": [_error(f"judge_{aspect}", exc)],
            "tokens": tokens,
        }

    if detector is None:
        from LLM.chinese_detector import ChineseDetector

        detector = ChineseDetector()
    errors: list[str] = []
    for issue in issues:
        if not detector.check_str(issue["issue"]):
            continue
        try:
            repaired, repair_tokens = await detector.repair_issue(issue["issue"])
            issue["issue"] = repaired
            tokens += repair_tokens
        except Exception as exc:  # noqa: BLE001 - keep the original issue on repair failure
            errors.append(_error(f"judge_{aspect}: chinese repair", exc))
    return {"issues": {aspect: issues}, "errors": errors, "tokens": tokens}


async def judge_anamnesis(state: DiagnosisState, **kwargs) -> dict[str, Any]:
    return await judge_aspect(state, "anamnesis", **kwargs)


async def judge_inspection(state: DiagnosisState, **kwargs) -> dict[str, Any]:
    return await judge_aspect(state, "inspection", **kwargs)


async def judge_treatment(state: DiagnosisState, **kwargs) -> dict[str, Any]:
    return await judge_aspect(state, "treatment", **kwargs)


async def judge_criteria(state: DiagnosisState, **kwargs) -> dict[str, Any]:
    return await judge_aspect(state, "criteria", **kwargs)


async def collect_sources(state: DiagnosisState) -> dict[str, Any]:
    return {
        "sources": collect_guideline_sources(
            state.get("pools", {}),
            state.get("issues", {}),
        )
    }


def _metadata_dict(raw_metadata: object) -> dict[str, Any]:
    if isinstance(raw_metadata, dict):
        return raw_metadata
    if isinstance(raw_metadata, str):
        try:
            parsed = json.loads(raw_metadata)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def chunk_from_row(
    row: dict[str, Any],
    *,
    file_id: str,
    doc_title: str,
    question: str | None = None,
) -> Chunk:
    """Normalize a retrieval row into the graph's stable chunk contract."""
    metadata = _metadata_dict(row.get("metadata"))
    questions = [question] if question else []
    return {
        "ref": 0,
        "id": str(row.get("id") or ""),
        "file_id": str(row.get("file_id") or file_id),
        "doc_title": str(doc_title),
        "section": metadata.get("section"),
        "chunk_index": _optional_int(metadata.get("chunk_index")),
        "content_type": str(metadata.get("content_type") or "text"),
        "page": _optional_int(metadata.get("page")),
        "table_index": _optional_int(metadata.get("table_index")),
        "text": str(row.get("chunk") or ""),
        "rrf_score": _optional_float(row.get("rrf_score")) or 0.0,
        "rerank_score": _optional_float(row.get("rerank_score")),
        "questions": questions,
    }


def _chunk_score(chunk: Chunk) -> float:
    rerank_score = chunk.get("rerank_score")
    return rerank_score if rerank_score is not None else chunk.get("rrf_score", 0.0)


def build_aspect_pool(
    rows_by_question: Iterable[tuple[str, list[dict[str, Any]]]],
    *,
    file_id: str,
    doc_title: str,
    limit: int = ASPECT_POOL_MAX_CHUNKS,
) -> list[Chunk]:
    """Deduplicate retrieval rows, cap by relevance, then number in document order."""
    by_id: dict[str, Chunk] = {}
    for question, rows in rows_by_question:
        for row in rows:
            candidate = chunk_from_row(
                row,
                file_id=file_id,
                doc_title=doc_title,
                question=question,
            )
            chunk_id = candidate["id"]
            if not chunk_id:
                logger.warning("[diagnosis_graph] ignoring retrieved chunk without id")
                continue

            current = by_id.get(chunk_id)
            if current is None:
                by_id[chunk_id] = candidate
                continue

            if question not in current["questions"]:
                current["questions"].append(question)
            if _chunk_score(candidate) > _chunk_score(current):
                candidate["questions"] = current["questions"]
                by_id[chunk_id] = candidate

    selected = sorted(by_id.values(), key=_chunk_score, reverse=True)[:limit]
    selected.sort(
        key=lambda chunk: (
            chunk.get("section") or "",
            chunk.get("chunk_index") if chunk.get("chunk_index") is not None else 2**31,
            chunk["id"],
        )
    )
    for ref, chunk in enumerate(selected, start=1):
        chunk["ref"] = ref
    return selected


def resolve_judge_output(
    output: JudgeOutput,
    *,
    aspect: Aspect,
    pool: list[Chunk],
    cite_max_chars: int = CITE_MAX_CHARS,
) -> list[ResolvedIssue]:
    """Resolve model-provided numeric references against the exact shown pool."""
    chunks_by_ref = {chunk["ref"]: chunk for chunk in pool}
    resolved: list[ResolvedIssue] = []

    for item in output.issues:
        issue_text = item.issue.strip()
        if not issue_text:
            continue

        sources: list[ResolvedSource] = []
        seen_refs: set[int] = set()
        for ref in item.chunk_refs:
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            chunk = chunks_by_ref.get(ref)
            if chunk is None:
                logger.warning(
                    "[diagnosis_graph] judge_%s returned unknown chunk ref=%s",
                    aspect,
                    ref,
                )
                continue
            text = chunk["text"]
            if chunk["content_type"] == "table":
                text = text.splitlines()[0] if text.splitlines() else text
            sources.append(
                {
                    "doc_title": chunk["doc_title"],
                    "section": chunk["section"],
                    "cite": text[:cite_max_chars],
                    "chunk_id": chunk["id"],
                    "chunk_index": chunk["chunk_index"],
                }
            )

        if not sources:
            logger.warning(
                "[diagnosis_graph] judge_%s issue has no valid source refs", aspect
            )
        resolved.append({"aspect": aspect, "issue": issue_text, "sources": sources})

    return resolved


def collect_guideline_sources(
    pools: dict[Aspect, list[Chunk]],
    issues: dict[Aspect, list[ResolvedIssue]],
) -> list[GuidelineSource]:
    """Aggregate every chunk shown to judges and mark sections cited by an issue."""
    cited_ids = {
        source["chunk_id"]
        for aspect_issues in issues.values()
        for issue in aspect_issues
        for source in issue["sources"]
        if source.get("chunk_id")
    }
    grouped: dict[tuple[str, str], dict[str | None, dict[str, Any]]] = {}

    for pool in pools.values():
        for chunk in pool:
            guideline_key = (chunk["file_id"], chunk["doc_title"])
            sections = grouped.setdefault(guideline_key, {})
            section = sections.setdefault(
                chunk["section"],
                {"chunk_indices": set(), "cited": False},
            )
            if chunk["chunk_index"] is not None:
                section["chunk_indices"].add(chunk["chunk_index"])
            section["cited"] = section["cited"] or chunk["id"] in cited_ids

    result: list[GuidelineSource] = []
    for (file_id, doc_title), raw_sections in sorted(grouped.items()):
        sections: list[GuidelineSourceSection] = []
        for section_name, data in sorted(
            raw_sections.items(), key=lambda item: item[0] or ""
        ):
            sections.append(
                {
                    "section": section_name,
                    "chunk_indices": sorted(data["chunk_indices"]),
                    "cited": bool(data["cited"]),
                }
            )
        result.append(
            {"file_id": file_id, "doc_title": doc_title, "sections": sections}
        )
    return result
