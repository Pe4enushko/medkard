"""
decider.py — LLM-based selection of the most relevant clinical-guideline
document when multiple candidates match an МКБ-10 code.

Given patient metadata, the diagnosis record, and the list of matching
manifest rows, the LLM picks the single most relevant ``file_id``.

Usage::
    from LLM.decider import decide_file_id

    file_id = await decide_file_id(patient, diagnosis, candidates)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        _client = AsyncOpenAI(base_url=base_url) if base_url else AsyncOpenAI()
    return _client


async def decide_file_id(
    patient: dict[str, Any],
    diagnosis: dict[str, Any],
    candidates: list[dict[str, str]],
) -> str | None:
    """Ask the LLM to pick the most relevant guideline file_id.

    Args:
        patient:    The «Пациент» dict from the raw visit JSON.
        diagnosis:  A single entry from «Диагнозы».
        candidates: List of manifest rows (dicts) that matched the МКБ code.

    Returns:
        The chosen ``ID`` string, or ``None`` if the LLM response is unusable.
    """
    candidate_json = json.dumps(candidates, ensure_ascii=False, indent=2)
    diagnosis_json = json.dumps(diagnosis, ensure_ascii=False, indent=2)
    patient_json = json.dumps(patient, ensure_ascii=False, indent=2)

    system = (
        "Ты — медицинский эксперт. Тебе даны данные о пациенте, диагноз и список "
        "клинических рекомендаций, подходящих по коду МКБ-10. "
        "Выбери ОДНУ наиболее подходящую рекомендацию для данного пациента и диагноза. "
        "Ответь ТОЛЬКО значением поля ID выбранной рекомендации, без пояснений."
    )

    user = (
        f"## Пациент\n{patient_json}\n\n"
        f"## Диагноз\n{diagnosis_json}\n\n"
        f"## Кандидаты (клинические рекомендации)\n{candidate_json}"
    )

    resp = await _get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
    )

    logger.debug("[decider] LLM response:\n%s", resp.model_dump_json(indent=2))
    raw_content = resp.choices[0].message.content
    finish_reason = resp.choices[0].finish_reason
    if finish_reason != "stop":
        logger.error(
            "[decider] unexpected finish_reason=%r; full response: %s",
            finish_reason,
            resp.model_dump_json(indent=2),
        )
    logger.debug("[decider] raw LLM answer: %s", raw_content)
    chosen = raw_content.strip().strip('"').strip("'")
    # Validate it is actually one of the candidate IDs
    valid_ids = {row.get("ID", "") for row in candidates}
    return chosen if chosen in valid_ids else None
