"""
chinese_detector.py — detect stray Chinese characters in LLM outputs.

LLMs occasionally hallucinate CJK characters in responses that should be
purely Russian/Latin. This module provides ChineseDetector with a core regex
check and typed adapters for every output model used in the LLM package.

Usage::
    from LLM.chinese_detector import ChineseDetector

    detector = ChineseDetector()

    detector.check_str("Нет нарушений")  # False
    detector.check_str("Нет 违规")        # True
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

from LLM.base import MODEL, get_openai_client

# CJK Unified Ideographs and common extensions — never valid in Russian medical text.
_CJK_RE = re.compile(
    r"[⺀-⻿"   # CJK Radicals Supplement
    r"⼀-⿟"    # Kangxi Radicals
    r"　-〿"    # CJK Symbols and Punctuation
    r"㐀-䶿"    # CJK Unified Ideographs Extension A
    r"一-鿿"    # CJK Unified Ideographs
    r"豈-﫿"    # CJK Compatibility Ideographs
    r"\U00020000-\U0002A6DF"  # CJK Unified Ideographs Extension B
    r"]"
)


class ChineseDetector:
    """Detect CJK characters in strings and structured LLM output models."""

    # ── Core ──────────────────────────────────────────────────────────────────

    def check_str(self, text: str) -> bool:
        """Return True if *text* contains any CJK character."""
        return bool(_CJK_RE.search(text))

    # ── Adapters ──────────────────────────────────────────────────────────────

    def check_findings(self, findings: list[Any]) -> bool:
        """Check findings from validations.py for CJK contamination.

        Accepts pydantic ``_Finding`` instances or plain dicts
        with ``flag`` / ``issue`` keys.
        """
        for finding in findings:
            if isinstance(finding, dict):
                flag, issue = finding.get("flag", ""), finding.get("issue", "")
            else:
                flag, issue = getattr(finding, "flag", ""), getattr(finding, "issue", "")
            if self.check_str(str(flag)) or self.check_str(str(issue)):
                return True
        return False

    def check_hypothetical_queries(self, queries: Any) -> bool:
        """Check a HypotheticalQueries object from query_generator.py.

        Accepts a pydantic ``HypotheticalQueries`` instance or a plain dict.
        """
        if isinstance(queries, dict):
            values = (
                queries.get("fact_query", ""),
                queries.get("procedural_query", ""),
                queries.get("constraint_query", ""),
            )
        else:
            values = (
                getattr(queries, "fact_query", ""),
                getattr(queries, "procedural_query", ""),
                getattr(queries, "constraint_query", ""),
            )
        return any(self.check_str(str(v)) for v in values)

    def check_visit_label(self, label: str | None) -> bool:
        """Check a visit classification label from visit_classifier.py."""
        return self.check_str(label or "")

    def check_file_id(self, file_id: str | None) -> bool:
        """Check a guideline file ID returned by decider.py."""
        return self.check_str(file_id or "")

    def check_raw_content(self, content: str | None) -> bool:
        """Check a raw ``message.content`` string before any parsing."""
        return self.check_str(content or "")

    async def repair_issue(self, issue: str) -> tuple[str, int]:
        """Ask the LLM to rewrite *issue*, replacing any CJK characters with Russian.

        Args:
            issue: An audit issue string that contains CJK characters.

        Returns:
            (repaired_text, tokens_used)
        """
        resp = await get_openai_client().chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты редактор медицинских текстов. "
                        "Тебе дан текст замечания, в котором есть китайские символы (артефакты генерации). "
                        "Замени все китайские символы на правильный русский текст по контексту. "
                        "Верни ТОЛЬКО исправленный текст, без пояснений."
                    ),
                },
                {"role": "user", "content": issue},
            ],
            temperature=0.3,
        )
        tokens = resp.usage.total_tokens if resp.usage else 0
        repaired = resp.choices[0].message.content.strip()
        logger.warning("[chinese_detector] repaired CJK in issue. original=%r repaired=%r", issue, repaired)
        return repaired, tokens
