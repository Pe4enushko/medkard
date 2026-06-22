"""
validations.py — LLM caller for audit validation checks.

Sends a visit record (as JSON) to the configured LLM together with a
pre-rendered system prompt (which already contains the applicable rules)
and returns a list of structured findings.

Each finding has the shape::
    {"flag": "<flag_code>", "issue": "<short Russian explanation>"}

Usage::
    from LLM.validations import validate_visit

    findings = await validate_visit(system_prompt, visit)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

from LLM.client import LLMClient

_client = LLMClient()


class _Finding(BaseModel):
    flag: str
    issue: str


class _Findings(BaseModel):
    root: list[_Finding]


async def validate_visit(
    system_prompt: str,
    visit: dict[str, Any],
    *,
    client: LLMClient | None = None,
) -> tuple[list[dict[str, str]], int]:
    """Call the LLM to validate a visit against a pre-rendered system prompt.

    Args:
        system_prompt: Fully rendered system prompt containing the applicable
                       rules (produced by FormalValidator).
        visit:         Raw visit dict (as parsed from the source JSON).

    Returns:
        (findings, tokens) — findings is a list of ``{"flag": ..., "issue": ...}``
        dicts; tokens is the total LLM token count for this call.
    """
    visit_text = json.dumps(visit, ensure_ascii=False, indent=2)

    resolved_client = client or _client
    raw_content, tokens = await resolved_client.call(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": visit_text},
        ],
        temperature=0.1,
        response_model=_Findings,
    )

    logger.debug("[validations] raw LLM answer: %s", raw_content)

    try:
        findings_obj = _Findings.model_validate_json(raw_content)
        return [{"flag": f.flag, "issue": f.issue} for f in findings_obj.root], tokens
    except Exception:
        logger.warning("[validations] failed to parse as _Findings, trying fallback: %r", raw_content)
        pass

    # Fallback: try parsing as a bare JSON array (some models ignore the wrapper)
    try:
        raw: list = json.loads(raw_content)
        if isinstance(raw, list):
            findings: list[dict[str, str]] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                try:
                    f = _Finding.model_validate(item)
                    findings.append({"flag": f.flag, "issue": f.issue})
                except Exception:
                    logger.warning("[validations] skipping malformed finding: %r", item)
            return findings, tokens
    except json.JSONDecodeError:
        pass

    logger.error("[validations] failed to parse JSON response: %r", raw_content)
    return [], tokens
