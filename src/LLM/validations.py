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
import re
from typing import Any

from pydantic import BaseModel, RootModel

logger = logging.getLogger(__name__)

from LLM.client import LLMClient

_client = LLMClient()


class _Finding(BaseModel):
    flag: str
    issue: str
    comment: str = ""


class _Findings(RootModel[list[_Finding]]):
    pass


def _finding_to_dict(finding: _Finding) -> dict[str, str]:
    return {"flag": finding.flag, "issue": finding.issue, "comment": finding.comment}


def _json_candidates(raw_content: str) -> list[str]:
    text = raw_content.strip()
    candidates = [text]

    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        candidates.insert(0, match.group(1).strip())

    for marker in ("[", "{"):
        idx = text.find(marker)
        if idx >= 0:
            candidates.append(text[idx:].strip())

    return [candidate for candidate in candidates if candidate]


def _parse_findings(raw_content: str) -> list[dict[str, str]]:
    """Parse LLM findings from a bare array, fenced JSON, or legacy root wrapper."""
    decoder = json.JSONDecoder()

    for candidate in _json_candidates(raw_content):
        try:
            findings_obj = _Findings.model_validate_json(candidate)
            return [_finding_to_dict(f) for f in findings_obj.root]
        except Exception:
            pass

        try:
            raw, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(raw, dict) and isinstance(raw.get("root"), list):
            raw = raw["root"]

        if isinstance(raw, list):
            findings: list[dict[str, str]] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                try:
                    findings.append(_finding_to_dict(_Finding.model_validate(item)))
                except Exception:
                    logger.warning("[validations] skipping malformed finding: %r", item)
            return findings

    logger.error("[validations] failed to parse JSON response: %r", raw_content)
    return []


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

    return _parse_findings(raw_content), tokens
