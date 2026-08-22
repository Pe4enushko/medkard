"""
audit/icd_check/validator.py — ICD-10 coding check for a full visit.

Два этапа с раздельным контекстом, без ReAct-цикла:

1. Отбор гипотез. Один вызов модели видит приём и перечень клинических
   рекомендаций целиком и называет не больше трёх рекомендаций, которые стоит
   прочитать. Перечень отправляется ровно один раз — в ReAct он переотправлялся
   на каждом шаге, и на карте `8b809667` прогона 21.08 это дало 182 532 токена.
2. Проверка гипотез. Разделы названных рекомендаций читаются кодом, каждая
   гипотеза судится отдельным вызовом и не видит остальных. Зациклиться здесь
   нечему: число вызовов известно до начала работы.

Usage::
    from audit.icd_check.validator import check_icd_codes

    issues, tokens = await check_icd_codes(patient, diagnoses, manifest_rows)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from LLM.client import LLMClient
from storage.models.guideline import Guideline
from storage.models.result import IcdCodingIssue, IssueSource

logger = logging.getLogger(__name__)

_client = LLMClient()

_PROMPTS = Path(__file__).parent.parent.parent / "LLM" / "prompts"
_PICKER_PROMPT: str = (_PROMPTS / "icd_candidate_picker.txt").read_text(encoding="utf-8")
_JUDGE_PROMPT: str = (_PROMPTS / "icd_code_judge.txt").read_text(encoding="utf-8")

# Сколько гипотез проверяем на визит. Не настройка: три чтения — это цена,
# которую контур платит на каждой карте, и её меняют осознанно, а не из .env.
_MAX_HYPOTHESES = 3
# Разделы одной рекомендации, уезжающие в контекст судьи.
_SECTION_CHARS_MAX = 8000
_MIN_CONFIDENCE = 8

# Разделы, по которым отличают одну нозологию от другой. Нумерация «1.x» —
# шаблон Минздрава (определение, кодирование по МКБ-10, классификация), но
# часть документов её не соблюдает, поэтому рядом идут ключевые слова.
# Каждая запись — набор слов, которые должны встретиться в названии разом.
# «Критерии» отдельно брать нельзя: «критерии оценки качества» — это раздел
# для аудита лечения, а не для различения нозологий.
_SECTION_KEYWORDS: tuple[tuple[str, ...], ...] = (
    ("определени",),
    ("кодирован",),
    ("классификац",),
    ("критери", "диагноз"),
    ("клиническая картина",),
)
_SECTIONS_PER_GUIDELINE = 5


class _CandidateChoice(BaseModel):
    dx_index: int
    file_id: str
    reason: str = Field(default="")


class _CandidateList(BaseModel):
    candidates: list[_CandidateChoice] = Field(default_factory=list)


class _Verdict(BaseModel):
    better: bool
    confidence: int = Field(default=0)
    suggested_code: str = Field(default="")
    comment: str = Field(default="")
    section: str = Field(default="")
    cite: str = Field(default="")


def _render_manifest_table(rows: list[Guideline]) -> str:
    """Перечень клинреков как таблица. Уходит модели один раз, на первом этапе."""
    header = "ID | Наименование | МКБ-10 | Возрастная категория"
    lines = [header, "-" * len(header)]
    for g in rows:
        lines.append(
            f"{g.file_id} | {g.name or ''} | "
            f"{', '.join(g.mkb)} | {', '.join(g.age_category)}"
        )
    return "\n".join(lines)


def _format_diagnoses(diagnoses: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, dx in enumerate(diagnoses):
        code = dx.get("КодМКБ", "—")
        name = dx.get("НаименованиеМКБ", "—")
        detail = dx.get("Детализация", "")
        line = f"{i}. Код МКБ: {code} — {name}"
        if detail:
            line += f" ({detail})"
        parts.append(line)
    return "\n".join(parts)


def _format_patient(patient: dict[str, Any]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in patient.items() if v is not None)


def _format_inspection(inspection_data: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in inspection_data:
        key = str(item.get("Параметр", "")).strip()
        value = str(item.get("Значение", "")).strip()
        if key and value:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _clinical_context(patient: dict[str, Any], inspection_data: list[dict[str, Any]]) -> str:
    text = f"## Пациент\n{_format_patient(patient)}\n\n"
    inspection = _format_inspection(inspection_data)
    if inspection:
        text += f"## Клинический контекст (данные осмотра)\n{inspection}\n\n"
    return text


def _code_compact(value: str) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _pick_sections(sections: list[str]) -> list[str]:
    """Разделы для чтения — в порядке оглавления, без обращения к модели."""
    picked = [
        section
        for section in sections
        if section.strip().startswith("1.")
        or any(
            all(word in section.lower() for word in words)
            for words in _SECTION_KEYWORDS
        )
    ]
    return picked[:_SECTIONS_PER_GUIDELINE]


async def _read_guideline(file_id: str, *, get_sections=None, get_chunks=None) -> tuple[str, list[str]]:
    """Прочитать различающие разделы одного клинрека. Возвращает (текст, разделы)."""
    if get_sections is None or get_chunks is None:
        from RAG.retrieval.searches import get_section_chunks, get_sections_for_file

        get_sections = get_sections or get_sections_for_file
        get_chunks = get_chunks or get_section_chunks

    sections = _pick_sections(await get_sections(file_id))
    if not sections:
        return "", []

    parts: list[str] = []
    read: list[str] = []
    size = 0
    for section in sections:
        chunks = await get_chunks(file_id, section)
        body = "\n".join(str(row.get("chunk", "")) for row in chunks if row.get("chunk"))
        if not body:
            continue
        block = f"### {section}\n{body}"
        if size + len(block) > _SECTION_CHARS_MAX:
            block = block[: max(0, _SECTION_CHARS_MAX - size)]
            if block:
                parts.append(block)
                read.append(section)
            break
        parts.append(block)
        read.append(section)
        size += len(block)
    return "\n\n".join(parts), read


async def _pick_candidates(
    context: str,
    diagnoses: list[dict[str, Any]],
    manifest_rows: list[Guideline],
    card_guid: str | None,
) -> tuple[list[_CandidateChoice] | None, int]:
    """Этап 1: какие клинреки читать. None — модель не ответила по контракту."""
    user = (
        f"{context}"
        f"## Диагнозы врача (все диагнозы визита)\n{_format_diagnoses(diagnoses)}\n\n"
        "## Клинические рекомендации (отфильтровано по возрасту пациента)\n"
        f"{_render_manifest_table(manifest_rows)}"
    )
    raw, tokens = await _client.call(
        messages=[
            {"role": "system", "content": _PICKER_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        response_model=_CandidateList,
        metadata={"card_guid": card_guid, "checker": "icd", "stage": "candidates"},
    )
    try:
        picked = _CandidateList.model_validate_json(raw)
    except ValidationError as exc:
        logger.error("[icd_check] этап отбора вернул не тот контракт: %s", exc)
        return None, tokens

    by_id = {g.file_id: g for g in manifest_rows}
    valid: list[_CandidateChoice] = []
    for candidate in picked.candidates:
        if candidate.file_id not in by_id:
            logger.warning("[icd_check] клинрек %r вне перечня — гипотеза отброшена", candidate.file_id)
            continue
        if not 0 <= candidate.dx_index < len(diagnoses):
            logger.warning("[icd_check] dx_index %d вне списка диагнозов", candidate.dx_index)
            continue
        valid.append(candidate)
    return valid[:_MAX_HYPOTHESES], tokens


async def _judge_candidate(
    context: str,
    diagnosis: dict[str, Any],
    guideline: Guideline,
    card_guid: str | None,
    *,
    read_guideline=None,
) -> tuple[_Verdict | None, int]:
    """Этап 2: подходит ли код этого клинрека лучше кода врача."""
    text, sections = await (read_guideline or _read_guideline)(guideline.file_id)
    if not text:
        logger.warning(
            "[icd_check] у клинрека %s нет различающих разделов — гипотеза не проверена",
            guideline.file_id,
        )
        return None, 0

    user = (
        f"{context}"
        f"## Диагноз врача\n"
        f"{diagnosis.get('КодМКБ', '—')} — {diagnosis.get('НаименованиеМКБ', '—')}\n\n"
        f"## Клиническая рекомендация\n"
        f"{guideline.name or guideline.file_id}\n"
        f"Коды МКБ-10 этой рекомендации: {', '.join(guideline.mkb)}\n\n"
        f"## Текст разделов\n{text}"
    )
    raw, tokens = await _client.call(
        messages=[
            {"role": "system", "content": _JUDGE_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        response_model=_Verdict,
        metadata={
            "card_guid": card_guid,
            "checker": "icd",
            "stage": "judge",
            "file_id": guideline.file_id,
            "sections": sections,
        },
    )
    try:
        verdict = _Verdict.model_validate_json(raw)
    except ValidationError as exc:
        logger.error("[icd_check] суждение по %s вернуло не тот контракт: %s", guideline.file_id, exc)
        return None, tokens

    allowed = {_code_compact(code) for code in guideline.mkb}
    if verdict.better and _code_compact(verdict.suggested_code) not in allowed:
        # Код вне рекомендации, по тексту которой судили: проверить его нашим
        # же контуром нечем, а в отчёте он выглядит как обоснованный.
        logger.warning(
            "[icd_check] %s предложен вне кодов клинрека %s (%s) — отброшено",
            verdict.suggested_code,
            guideline.file_id,
            ", ".join(guideline.mkb),
        )
        return None, tokens
    return verdict, tokens


async def check_icd_codes(
    patient: dict[str, Any],
    diagnoses: list[dict[str, Any]],
    manifest_rows: list[Guideline],
    inspection_data: list[dict[str, Any]] | None = None,
    card_guid: str | None = None,
) -> tuple[list[IcdCodingIssue] | None, int]:
    """Проверить кодирование всех диагнозов визита.

    Returns:
        (рекомендации, токены) — либо (None, токены), если чекер не отработал.
        None и пустой список — разные вещи: пустой означает «код сомнений не
        вызвал», None — «мнения нет», и карта не должна выдавать второе за первое.
    """
    if not diagnoses:
        return [], 0

    context = _clinical_context(patient, inspection_data or [])

    logger.info(
        "[icd_check] отбор гипотез: %d диагноз(ов), %d клинреков",
        len(diagnoses),
        len(manifest_rows),
    )
    candidates, tokens = await _pick_candidates(context, diagnoses, manifest_rows, card_guid)
    if candidates is None:
        return None, tokens
    if not candidates:
        logger.info("[icd_check] гипотез нет — кодирование сомнений не вызвало")
        return [], tokens

    by_id = {g.file_id: g for g in manifest_rows}
    logger.info(
        "[icd_check] проверяю %d гипотез(ы): %s",
        len(candidates),
        ", ".join(f"dx{c.dx_index}→{c.file_id}" for c in candidates),
    )
    judged = await asyncio.gather(
        *[
            _judge_candidate(context, diagnoses[c.dx_index], by_id[c.file_id], card_guid)
            for c in candidates
        ],
        return_exceptions=True,
    )

    issues: list[IcdCodingIssue] = []
    answered = 0
    for candidate, outcome in zip(candidates, judged):
        if isinstance(outcome, BaseException):
            logger.error(
                "[icd_check] гипотеза %s не проверена: %s", candidate.file_id, outcome
            )
            continue
        verdict, spent = outcome
        tokens += spent
        if verdict is None:
            continue
        answered += 1
        if not verdict.better or verdict.confidence < _MIN_CONFIDENCE:
            continue
        if not verdict.suggested_code or not verdict.comment:
            continue
        guideline = by_id[candidate.file_id]
        issues.append(
            IcdCodingIssue(
                dx_index=candidate.dx_index,
                initial_code=diagnoses[candidate.dx_index].get("КодМКБ", "?"),
                suggested_code=verdict.suggested_code,
                confidence=verdict.confidence,
                comment=verdict.comment,
                sources=[
                    IssueSource(
                        doc_title=guideline.name or guideline.file_id,
                        section=verdict.section or None,
                        cite=verdict.cite or None,
                    )
                ],
            )
        )
        logger.info(
            "[icd_check] рекомендация dx_index=%d: %s → %s (уверенность %d, %s)",
            candidate.dx_index,
            diagnoses[candidate.dx_index].get("КодМКБ", "?"),
            verdict.suggested_code,
            verdict.confidence,
            verdict.comment,
        )

    if answered == 0:
        # Гипотезы были, но ни одна не дошла до суждения: «замечаний нет»
        # сказать не о чем.
        logger.error("[icd_check] ни одна гипотеза не проверена — мнения нет")
        return None, tokens

    logger.info("[icd_check] готово — %d рекомендаци(й), токенов=%d", len(issues), tokens)
    return issues, tokens
