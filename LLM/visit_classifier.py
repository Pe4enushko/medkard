"""
LLM/visit_classifier.py — classify the type of an ambulatory visit via LLM.

Usage::
    from LLM.visit_classifier import VisitClassifier

    classifier = VisitClassifier()
    label = await classifier.classify(visit)   # "primary" | "repeat" | "prophylactic" | None
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_PROMPT: str = (Path(__file__).parent / "prompts" / "visit_type_classifier.txt").read_text(encoding="utf-8")
_VALID_LABELS = {"primary", "repeat", "prophylactic"}


class VisitClassifier:
    """Ask the LLM to classify a visit as primary / repeat / prophylactic."""

    async def classify(self, visit: dict[str, Any]) -> str | None:
        """Return one of ``"primary"``, ``"repeat"``, ``"prophylactic"``, or
        ``None`` if the model response is not a recognised label.

        Args:
            visit: Raw visit dict (as parsed from the source JSON).
        """
        client = AsyncOpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)
        model = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")

        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _PROMPT},
                {"role": "user", "content": json.dumps(visit, ensure_ascii=False, indent=2)},
            ],
            temperature=0
        )

        answer = resp.choices[0].message.content.strip().lower()
        if answer not in _VALID_LABELS:
            logger.warning("[visit_classifier] unrecognised label %r", answer)
            return None

        logger.info("[visit_classifier] classified as %r", answer)
        return answer
