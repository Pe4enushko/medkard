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

from audit.graph_trace import emit as trace_emit
from LLM.prompt_context import today_block
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

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class StructuredClient(Protocol):
    async def call(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        response_model: type[BaseModel] | None = None,
        reasoning_effort: str | None = None,
        enable_thinking: bool | None = None,
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
    diagnosis = diagnosis_block.strip()
    if not diagnosis:
        raise ValueError("cannot build fallback questions without diagnosis")
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
    trace_emit(
        "graph.node.started",
        node="generate_questions",
        dx_code=state.get("dx_code"),
        input={
            "patient_block": state.get("patient_block"),
            "diagnosis_block": state.get("diagnosis_block"),
            "visit_context": state.get("visit_context"),
            "toc": state.get("toc", []),
        },
    )
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
                "correlation_id": state.get("correlation_id"),
                "dx_code": state.get("dx_code"),
            },
        )
        output = _validated_output(raw, QuestionSet)
        assert isinstance(output, QuestionSet)
        questions = _questions_from_output(output)
        if not questions:
            raise ValueError("model returned no questions")
        update = {"questions": questions, "tokens": tokens}
        trace_emit(
            "graph.node.completed",
            node="generate_questions",
            dx_code=state.get("dx_code"),
            output=update,
            raw_output=output,
        )
        return update
    except Exception as exc:  # noqa: BLE001 - diagnosed fallback is deliberate
        try:
            fallback_questions = _fallback_questions(
                state.get("diagnosis_block", "")
            )
        except ValueError as fallback_exc:
            trace_emit(
                "graph.node.failed",
                node="generate_questions",
                dx_code=state.get("dx_code"),
                exception=fallback_exc,
                generation_exception=exc,
                tokens=tokens,
            )
            raise fallback_exc from exc
        update = {
            "questions": fallback_questions,
            "errors": [_error("generate_questions: fallback templates", exc)],
            "tokens": tokens,
        }
        trace_emit(
            "graph.node.degraded",
            node="generate_questions",
            dx_code=state.get("dx_code"),
            exception=exc,
            output=update,
        )
        return update


def _flatten(text: str) -> str:
    """Для сверки дословности: регистр и переносы строк расхождением не считаем."""
    return " ".join(text.lower().split())


_VERBATIM_REPROACH = (
    "Этих фрагментов нет в записи дословно: {bad}. Ты их изменил или придумал. "
    "Верни список заново, беря только те строки, которые есть в тексте символ "
    "в символ, вместе с опечатками врача."
)


def _verbatim(items: list[Any], context: str) -> tuple[list[str], list[str]]:
    """Разделить упоминания на найденные в тексте и выдуманные."""
    flat = _flatten(context)
    good: list[str] = []
    bad: list[str] = []
    for item in items:
        written = item.as_written.strip()
        if not written:
            continue
        (good if _flatten(written) in flat else bad).append(written)
    return good, bad


async def extract_drugs(
    state: DiagnosisState,
    *,
    client: StructuredClient | None = None,
) -> dict[str, Any]:
    trace_emit(
        "graph.node.started",
        node="extract_drugs",
        dx_code=state.get("dx_code"),
        input={"visit_context": state.get("visit_context")},
    )
    client = client or _default_client()
    tokens = 0
    try:
        context = state.get("visit_context", "—")
        messages = [
            {"role": "system", "content": _load_prompt("diagnosis_drugs.txt")},
            {"role": "user", "content": context},
        ]
        # Второй заход — не «ещё раз то же самое», а разбор с перечнем
        # непрошедших строк: без него модель повторяет ту же выдумку.
        for attempt in range(2):
            raw, spent = await client.call(
                messages,
                temperature=0.0,
                response_model=DrugList,
                reasoning_effort="low",
                metadata={
                    "node": "extract_drugs",
                    "card_guid": state.get("card_guid"),
                    "correlation_id": state.get("correlation_id"),
                    "dx_code": state.get("dx_code"),
                    "attempt": attempt + 1,
                },
            )
            tokens += spent
            output = _validated_output(raw, DrugList)
            assert isinstance(output, DrugList)
            good, bad = _verbatim(output.items, context)
            if not bad or attempt:
                break
            trace_emit(
                "medicine.extraction.not_verbatim",
                dx_code=state.get("dx_code"),
                invented=bad,
                kept=good,
            )
            messages = [*messages, {"role": "user",
                                    "content": _VERBATIM_REPROACH.format(bad=", ".join(bad))}]
        if bad:
            logger.warning("[extract_drugs] выдуманные упоминания отброшены: %s", bad)
        mentions = [{"as_written": written} for written in good]
        update = {"drug_mentions": mentions, "tokens": tokens}
        trace_emit(
            "graph.node.completed",
            node="extract_drugs",
            dx_code=state.get("dx_code"),
            output=update,
            raw_output=output,
        )
        trace_emit(
            "medicine.extracted",
            dx_code=state.get("dx_code"),
            mentions=mentions,
        )
        return update
    except Exception as exc:  # noqa: BLE001 - node-level degradation is deliberate
        update = {
            "drug_mentions": [],
            "errors": [_error("extract_drugs", exc)],
            "tokens": tokens,
        }
        trace_emit(
            "graph.node.degraded",
            node="extract_drugs",
            dx_code=state.get("dx_code"),
            exception=exc,
            output=update,
        )
        return update


# Реестр ищется по написанию из карты, а совпадение написаний — не опознание
# препарата. Судья обязан видеть это как посылку, а не догадываться по формулировке
# отдельной карточки: на прогоне 2026-08-23 карточка Но-шпы (дротаверин) уехала в
# контекст как справка о Нольпазе (пантопразол) — совпадение было точным, неверным
# был запрос.
_REGISTRY_CAVEAT = (
    "Справка из ГРЛС подобрана по написанию препарата в карте. Совпадение строк — "
    "не опознание препарата: уровень совпадения указан у каждой записи, и всё, что "
    "не помечено точным, считай предположительным.\n"
)


async def _default_medicine_lookup(query: str, on: date | None = None) -> str:
    from grls.format import format_medicine_lookup
    from grls.lookup import lookup_medicine

    raw_result = await lookup_medicine(query, on=on)
    formatted = format_medicine_lookup(raw_result)
    trace_emit(
        "medicine.registry.completed",
        registry="grls",
        query=query,
        visit_date=on,
        results=raw_result,
        formatted=formatted,
    )
    return formatted


async def lookup_drugs(
    state: DiagnosisState,
    *,
    lookup=None,
) -> dict[str, Any]:
    mentions = state.get("drug_mentions", [])
    trace_emit(
        "graph.node.started",
        node="lookup_drugs",
        dx_code=state.get("dx_code"),
        input={"mentions": mentions, "visit_date": state.get("visit_date")},
    )
    if not mentions:
        update = {"drug_context": ""}
        trace_emit(
            "graph.node.completed",
            node="lookup_drugs",
            dx_code=state.get("dx_code"),
            output=update,
        )
        return update
    lookup = lookup or _default_medicine_lookup
    try:
        lines = []
        for mention in mentions:
            result = await lookup(mention["as_written"], on=state.get("visit_date"))
            trace_emit(
                "medicine.retrieved",
                dx_code=state.get("dx_code"),
                mention=mention,
                visit_date=state.get("visit_date"),
                result=result,
            )
            lines.append(f"- {mention['as_written']} → {result}")
    except Exception as exc:  # noqa: BLE001 - registry failure must not drop the card
        update = {
            "drug_context": "справка недоступна",
            "errors": [_error("lookup_drugs", exc)],
        }
        trace_emit(
            "graph.node.degraded",
            node="lookup_drugs",
            dx_code=state.get("dx_code"),
            exception=exc,
            output=update,
        )
        return update

    visit_date = state.get("visit_date")
    prefix = f"Дата визита: {visit_date.isoformat()}\n" if visit_date else ""
    update = {"drug_context": prefix + _REGISTRY_CAVEAT + "\n".join(lines)}
    trace_emit(
        "graph.node.completed",
        node="lookup_drugs",
        dx_code=state.get("dx_code"),
        output=update,
    )
    return update


async def retrieve(
    state: DiagnosisState,
    *,
    search=None,
) -> dict[str, Any]:
    trace_emit(
        "graph.node.started",
        node="retrieve",
        dx_code=state.get("dx_code"),
        input={
            "file_id": state.get("file_id"),
            "questions": state.get("questions", []),
        },
    )
    if search is None:
        from RAG.retrieval.searches import search_in_guideline

        search = search_in_guideline

    semaphore = asyncio.Semaphore(RETRIEVE_CONCURRENCY)

    async def _one(question: Question):
        async with semaphore:
            trace_emit(
                "retrieval.query.started",
                dx_code=state.get("dx_code"),
                file_id=state.get("file_id"),
                aspect=question["aspect"],
                query=question["text"],
            )
            try:
                rows = await search(
                    question["text"],
                    state["file_id"],
                    candidates=CANDIDATES_PER_QUESTION,
                    top_k=TOP_K_PER_QUESTION,
                )
                trace_emit(
                    "retrieval.query.completed",
                    dx_code=state.get("dx_code"),
                    file_id=state.get("file_id"),
                    aspect=question["aspect"],
                    query=question["text"],
                    chunks=rows,
                )
                return question, rows, None
            except Exception as exc:  # noqa: BLE001 - isolate one retrieval question
                trace_emit(
                    "retrieval.query.failed",
                    dx_code=state.get("dx_code"),
                    file_id=state.get("file_id"),
                    aspect=question["aspect"],
                    query=question["text"],
                    exception=exc,
                )
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
    update = {"pools": pools, "errors": errors}
    trace_emit(
        "graph.node.completed",
        node="retrieve",
        dx_code=state.get("dx_code"),
        output=update,
    )
    return update


async def retrieve_criteria(
    state: DiagnosisState,
    *,
    get_chunks=None,
) -> dict[str, Any]:
    trace_emit(
        "graph.node.started",
        node="retrieve_criteria",
        dx_code=state.get("dx_code"),
        input={
            "file_id": state.get("file_id"),
            "pattern": CRITERIA_SECTION_PATTERN,
        },
    )
    if get_chunks is None:
        from RAG.retrieval.searches import get_section_chunks_by_pattern

        get_chunks = get_section_chunks_by_pattern
    try:
        rows = await get_chunks(
            state["file_id"],
            CRITERIA_SECTION_PATTERN,
        )
    except Exception as exc:  # noqa: BLE001 - criteria is an optional graph branch
        update = {
            "pools": {"criteria": []},
            "errors": [_error("retrieve_criteria", exc)],
        }
        trace_emit(
            "graph.node.degraded",
            node="retrieve_criteria",
            dx_code=state.get("dx_code"),
            exception=exc,
            output=update,
        )
        return update
    trace_emit(
        "retrieval.criteria.completed",
        dx_code=state.get("dx_code"),
        file_id=state.get("file_id"),
        pattern=CRITERIA_SECTION_PATTERN,
        chunks=rows,
    )
    if not rows:
        update = {
            "pools": {"criteria": []},
            "errors": ["retrieve_criteria: section not found"],
        }
        trace_emit(
            "graph.node.degraded",
            node="retrieve_criteria",
            dx_code=state.get("dx_code"),
            output=update,
        )
        return update
    pool = build_aspect_pool(
        [("", rows)],
        file_id=state["file_id"],
        doc_title=state["doc_title"],
        limit=None,
    )
    for chunk in pool:
        chunk["questions"] = []
    update = {"pools": {"criteria": pool}}
    trace_emit(
        "graph.node.completed",
        node="retrieve_criteria",
        dx_code=state.get("dx_code"),
        output=update,
    )
    return update


def _render_pool(pool: list[Chunk]) -> str:
    allowed_refs = ", ".join(str(chunk["ref"]) for chunk in pool)
    parts = [
        (
            f"Допустимые значения chunk_refs: {allowed_refs}. "
            "Используй только номер после chunk_ref=; "
            "номера разделов в него не входят."
        )
    ]
    for chunk in pool:
        parts.append(
            f"### Источник chunk_ref={chunk['ref']}\n"
            f"Раздел: {chunk['section'] or 'не указан'}\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def _markdown_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        rendered = json.dumps(value, ensure_ascii=False)
    else:
        rendered = str(value)
    return rendered.replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


def _render_criteria_pool(pool: list[Chunk]) -> str:
    """Reconstruct all ingested row batches as one criteria table."""
    columns: list[str] = []
    table_rows: list[tuple[int, dict[str, object]]] = []
    unparsed: list[Chunk] = []

    for chunk in pool:
        try:
            raw_rows = json.loads(chunk["text"])
        except (json.JSONDecodeError, TypeError):
            unparsed.append(chunk)
            continue
        if not isinstance(raw_rows, list) or not raw_rows:
            unparsed.append(chunk)
            continue

        parsed_any = False
        for raw_row in raw_rows:
            if not isinstance(raw_row, dict):
                continue
            row = {str(key): value for key, value in raw_row.items()}
            for column in row:
                if column not in columns:
                    columns.append(column)
            table_rows.append((chunk["ref"], row))
            parsed_any = True
        if not parsed_any:
            unparsed.append(chunk)

    allowed_refs = ", ".join(str(chunk["ref"]) for chunk in pool)
    parts = [
        (
            f"Допустимые значения chunk_refs: {allowed_refs}. "
            "Первый столбец — техническая ссылка на источник. "
            "Номер критерия из таблицы не является chunk_ref."
        )
    ]
    if table_rows:
        headers = ["chunk_ref (источник)", *columns]
        parts.extend(
            [
                "| " + " | ".join(_markdown_cell(header) for header in headers) + " |",
                "| " + " | ".join("---" for _header in headers) + " |",
            ]
        )
        for ref, row in table_rows:
            values = [ref, *(row.get(column, "") for column in columns)]
            parts.append(
                "| " + " | ".join(_markdown_cell(value) for value in values) + " |"
            )

    for chunk in unparsed:
        parts.append(
            f"### Нераспознанная часть таблицы chunk_ref={chunk['ref']}\n"
            f"Раздел: {chunk['section'] or 'не указан'}\n{chunk['text']}"
        )
    return "\n".join(parts)


async def judge_aspect(
    state: DiagnosisState,
    aspect: Aspect,
    *,
    client: StructuredClient | None = None,
    detector=None,
) -> dict[str, Any]:
    pool = state.get("pools", {}).get(aspect, [])
    trace_emit(
        "graph.node.started",
        node=f"judge_{aspect}",
        dx_code=state.get("dx_code"),
        input={"pool": pool},
    )
    if not pool:
        update = {"issues": {aspect: []}}
        trace_emit(
            "graph.node.completed",
            node=f"judge_{aspect}",
            dx_code=state.get("dx_code"),
            output=update,
            skipped_reason="empty pool",
        )
        return update

    client = client or _default_client()
    prompt_name = {
        "anamnesis": "anamnesis_checker.txt",
        "inspection": "inspection_checker.txt",
        "treatment": "treatment_checker.txt",
        "criteria": "criteria_checker.txt",
    }[aspect]
    context_parts = []
    today = today_block(state.get("visit_date"))
    if today:
        context_parts.append(today)
    context_parts += [
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
    rendered_pool = (
        _render_criteria_pool(pool) if aspect == "criteria" else _render_pool(pool)
    )
    context_parts.append(
        f"## {heading} «{state.get('doc_title', '')}»\n{rendered_pool}"
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
            enable_thinking=True,
            metadata={
                "node": f"judge_{aspect}",
                "card_guid": state.get("card_guid"),
                "correlation_id": state.get("correlation_id"),
                "dx_code": state.get("dx_code"),
            },
        )
        output = _validated_output(raw, JudgeOutput)
        assert isinstance(output, JudgeOutput)
        issues = resolve_judge_output(output, aspect=aspect, pool=pool)
    except Exception as exc:  # noqa: BLE001 - malformed structured output degrades one judge
        update = {
            "issues": {aspect: []},
            "errors": [_error(f"judge_{aspect}", exc)],
            "tokens": tokens,
        }
        trace_emit(
            "graph.node.degraded",
            node=f"judge_{aspect}",
            dx_code=state.get("dx_code"),
            exception=exc,
            checker_context=context_parts,
            output=update,
        )
        return update

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
    update = {"issues": {aspect: issues}, "errors": errors, "tokens": tokens}
    trace_emit(
        "graph.node.completed",
        node=f"judge_{aspect}",
        dx_code=state.get("dx_code"),
        checker_context=context_parts,
        raw_output=output,
        output=update,
    )
    return update


async def judge_anamnesis(state: DiagnosisState, **kwargs) -> dict[str, Any]:
    return await judge_aspect(state, "anamnesis", **kwargs)


async def judge_inspection(state: DiagnosisState, **kwargs) -> dict[str, Any]:
    return await judge_aspect(state, "inspection", **kwargs)


async def judge_treatment(state: DiagnosisState, **kwargs) -> dict[str, Any]:
    return await judge_aspect(state, "treatment", **kwargs)


async def judge_criteria(state: DiagnosisState, **kwargs) -> dict[str, Any]:
    return await judge_aspect(state, "criteria", **kwargs)


async def collect_sources(state: DiagnosisState) -> dict[str, Any]:
    trace_emit(
        "graph.node.started",
        node="collect_sources",
        dx_code=state.get("dx_code"),
        input={"pools": state.get("pools", {}), "issues": state.get("issues", {})},
    )
    update = {
        "sources": collect_guideline_sources(
            state.get("pools", {}),
            state.get("issues", {}),
        )
    }
    trace_emit(
        "graph.node.completed",
        node="collect_sources",
        dx_code=state.get("dx_code"),
        output=update,
    )
    return update


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
    limit: int | None = ASPECT_POOL_MAX_CHUNKS,
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

    selected = sorted(by_id.values(), key=_chunk_score, reverse=True)
    if limit is not None:
        selected = selected[:limit]
    selected.sort(
        key=lambda chunk: (
            chunk.get("section") or "",
            chunk.get("page") if chunk.get("page") is not None else -1,
            chunk.get("table_index") if chunk.get("table_index") is not None else -1,
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
