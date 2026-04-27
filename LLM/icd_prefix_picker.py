"""
icd_prefix_picker.py — LLM-based selection of a clinical guideline when the
exact МКБ-10 code had no match and a prefix-only lookup returned candidates.

Usage::
    from LLM.icd_prefix_picker import IcdPrefixPicker

    picker = IcdPrefixPicker()
    file_id = await picker.pick(patient, diagnosis, candidates)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")

_PROMPT_PATH = Path(__file__).parent / "prompts" / "icd_prefix_picker.txt"

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        _client = AsyncOpenAI(base_url=base_url) if base_url else AsyncOpenAI()
    return _client


class IcdPrefixPicker:
    """Pick a guideline file_id from prefix-matched candidates using an LLM."""

    def __init__(self) -> None:
        self._system = _PROMPT_PATH.read_text(encoding="utf-8").strip()

    async def pick(
        self,
        patient: dict[str, Any],
        diagnosis: dict[str, Any],
        candidates: list[dict[str, str]],
    ) -> str | None:
        """Return the most relevant guideline ID among prefix-matched *candidates*.

        Args:
            patient:    Patient info dict.
            diagnosis:  Diagnosis dict with at least ``КодМКБ`` key.
            candidates: Manifest rows matched by the ICD prefix (e.g. ``J20``).

        Returns:
            The chosen ``ID`` string, or ``None`` if the response is unusable.
        """
        user = (
            f"## Пациент\n{json.dumps(patient, ensure_ascii=False, indent=2)}\n\n"
            f"## Диагноз\n{json.dumps(diagnosis, ensure_ascii=False, indent=2)}\n\n"
            f"## Кандидаты (клинические рекомендации)\n"
            f"{json.dumps(candidates, ensure_ascii=False, indent=2)}"
        )

        resp = await _get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": self._system},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
        )

        finish_reason = resp.choices[0].finish_reason
        if finish_reason != "stop":
            logger.error(
                "[icd_prefix_picker] unexpected finish_reason=%r; response: %s",
                finish_reason,
                resp.model_dump_json(indent=2),
            )

        raw = resp.choices[0].message.content
        logger.debug("[icd_prefix_picker] raw answer: %s", raw)
        chosen = raw.strip().strip('"').strip("'")
        if chosen.lower() == "none":
            return None
        valid_ids = {row.get("ID", "") for row in candidates}
        return chosen if chosen in valid_ids else None
