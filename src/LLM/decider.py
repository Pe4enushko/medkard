"""
decider.py — LLM-based selection of the most relevant clinical-guideline
document when multiple candidates match an МКБ-10 code.

Given patient metadata, the diagnosis record, and the list of matching
Guideline candidates, the LLM picks the single most relevant file_id.

Usage::
    from LLM.decider import decide_file_id

    file_id, tokens = await decide_file_id(patient, diagnosis, candidates)
"""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

from LLM.client import LLMClient
from storage.models.guideline import Guideline

_client = LLMClient()


async def decide_file_id(
    patient: dict[str, Any],
    diagnosis: dict[str, Any],
    candidates: list[Guideline],
) -> tuple[str | None, int]:
    """Ask the LLM to pick the most relevant guideline file_id.

    Args:
        patient:    The «Пациент» dict from the raw visit JSON.
        diagnosis:  A single entry from «Диагнозы».
        candidates: Guideline objects that matched the МКБ code.

    Returns:
        (chosen file_id or None, tokens spent).
    """
    candidate_json = json.dumps([asdict(c) for c in candidates], ensure_ascii=False, indent=2)
    diagnosis_json = json.dumps(diagnosis, ensure_ascii=False, indent=2)
    patient_json = json.dumps(patient, ensure_ascii=False, indent=2)

    system = (
        "Ты — медицинский эксперт. Тебе даны данные о пациенте, диагноз и список "
        "клинических рекомендаций, подходящих по коду МКБ-10. "
        "Выбери ОДНУ наиболее подходящую рекомендацию для данного пациента и диагноза. "
        "Ответь ТОЛЬКО значением поля file_id выбранной рекомендации, без пояснений."
    )

    user = (
        f"## Пациент\n{patient_json}\n\n"
        f"## Диагноз\n{diagnosis_json}\n\n"
        f"## Кандидаты (клинические рекомендации)\n{candidate_json}"
    )

    raw_content, tokens = await _client.call(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
    )

    logger.debug("[decider] raw LLM answer: %s", raw_content)
    chosen = raw_content.strip().strip('"').strip("'")
    valid_ids = {c.file_id for c in candidates}
    return (chosen if chosen in valid_ids else None), tokens
