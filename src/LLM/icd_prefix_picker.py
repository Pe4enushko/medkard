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
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from LLM.base import MODEL, get_openai_client

_PROMPT_PATH = Path(__file__).parent / "prompts" / "icd_prefix_picker.txt"


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

        resp = await get_openai_client().chat.completions.create(
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

        tokens = resp.usage.total_tokens if resp.usage else 0
        raw = resp.choices[0].message.content
        logger.debug("[icd_prefix_picker] raw answer: %s", raw)
        chosen = raw.strip().strip('"').strip("'")
        if chosen.lower() == "none":
            return None, tokens
        valid_ids = {row.get("ID", "") for row in candidates}
        return (chosen if chosen in valid_ids else None), tokens
