"""
icd_prefix_picker.py — LLM-based selection of a clinical guideline when the
exact МКБ-10 code had no match and a prefix-only lookup returned candidates.

Usage::
    from LLM.icd_prefix_picker import IcdPrefixPicker

    picker = IcdPrefixPicker()
    file_id, tokens = await picker.pick(patient, diagnosis, candidates)
"""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from LLM.client import LLMClient
from storage.models.guideline import Guideline

_PROMPT_PATH = Path(__file__).parent / "prompts" / "icd_prefix_picker.txt"


class IcdPrefixPicker:
    """Pick a guideline file_id from prefix-matched candidates using an LLM."""

    def __init__(self) -> None:
        self._system = _PROMPT_PATH.read_text(encoding="utf-8").strip()
        self._client = LLMClient()

    async def pick(
        self,
        patient: dict[str, Any],
        diagnosis: dict[str, Any],
        candidates: list[Guideline],
    ) -> tuple[str | None, int]:
        """Return the most relevant guideline file_id among prefix-matched *candidates*.

        Args:
            patient:    Patient info dict.
            diagnosis:  Diagnosis dict with at least ``КодМКБ`` key.
            candidates: Guideline objects matched by the ICD prefix (e.g. ``J20``).

        Returns:
            (chosen file_id or None, tokens spent).
        """
        candidate_json = json.dumps([asdict(c) for c in candidates], ensure_ascii=False, indent=2)
        user = (
            f"## Пациент\n{json.dumps(patient, ensure_ascii=False, indent=2)}\n\n"
            f"## Диагноз\n{json.dumps(diagnosis, ensure_ascii=False, indent=2)}\n\n"
            f"## Кандидаты (клинические рекомендации)\n{candidate_json}"
        )

        raw, tokens = await self._client.call(
            messages=[
                {"role": "system", "content": self._system},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
        )

        logger.debug("[icd_prefix_picker] raw answer: %s", raw)
        chosen = raw.strip().strip('"').strip("'")
        if chosen.lower() == "none":
            return None, tokens
        valid_ids = {c.file_id for c in candidates}
        return (chosen if chosen in valid_ids else None), tokens
