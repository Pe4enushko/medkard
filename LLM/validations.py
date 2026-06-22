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
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from LLM.base import MODEL, get_openai_client

SCHEMAS_DIR: Path = Path(__file__).parent / "schemas"

_JSON_SCHEMA: dict = json.loads(
    (SCHEMAS_DIR / "formal_structure_findings.json").read_text(encoding="utf-8")
)


class _Finding(BaseModel):
    flag: str
    issue: str


async def validate_visit(
    system_prompt: str,
    visit: dict[str, Any],
    *,
    client: AsyncOpenAI | None = None,
    model: str = MODEL,
) -> tuple[list[dict[str, str]], int]:
    """Call the LLM to validate a visit against a pre-rendered system prompt.

    Args:
        system_prompt: Fully rendered system prompt containing the applicable
                       rules (produced by FormalValidator).
        visit:         Raw visit dict (as parsed from the source JSON).
        client:        Optional AsyncOpenAI client (for testing / client reuse).
                       Falls back to the module-level singleton.
        model:         LLM model identifier.

    Returns:
        (findings, tokens) — findings is a list of ``{"flag": ..., "issue": ...}``
        dicts; tokens is the total LLM token count for this call.
    """
    resolved_client = client or get_openai_client()
    visit_text = json.dumps(visit, ensure_ascii=False, indent=2)

    completion = await resolved_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": visit_text},
        ],
        temperature=0.1,
        extra_body={"guided_json": _JSON_SCHEMA},
    )

    finish_reason = completion.choices[0].finish_reason
    if finish_reason != "stop":
        logger.error(
            "[validations] unexpected finish_reason=%r; full response: %s",
            finish_reason,
            completion.model_dump_json(indent=2),
        )

    raw_content = completion.choices[0].message.content or "[]"
    logger.debug("[validations] raw LLM answer: %s", raw_content)
    tokens = completion.usage.total_tokens if completion.usage else 0

    try:
        raw: list = json.loads(raw_content)
    except json.JSONDecodeError:
        logger.error("[validations] failed to parse JSON response: %r", raw_content)
        return [], tokens

    if not isinstance(raw, list):
        logger.error("[validations] expected list, got %s: %r", type(raw).__name__, raw_content)
        return [], tokens

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
