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

from pydantic import BaseModel

logger = logging.getLogger(__name__)

from LLM.base import MODEL, get_instructor_client

SCHEMAS_DIR: Path = Path(__file__).parent / "schemas"

_JSON_SCHEMA: dict = json.loads(
    (SCHEMAS_DIR / "formal_structure_findings.json").read_text(encoding="utf-8")
)


class _Finding(BaseModel):
    flag: str
    issue: str


class _Findings(BaseModel):
    findings: list[_Finding]


async def validate_visit(
    system_prompt: str,
    visit: dict[str, Any],
    *,
    client: instructor.AsyncInstructor | None = None,
    model: str = MODEL,
) -> list[dict[str, str]]:
    """Call the LLM to validate a visit against a pre-rendered system prompt.

    Args:
        system_prompt: Fully rendered system prompt containing the applicable
                       rules (produced by FormalValidator).
        visit:         Raw visit dict (as parsed from the source JSON).
        client:        Optional pre-built instructor client (for testing /
                       client reuse). Falls back to the module-level singleton.
        model:         LLM model identifier.

    Returns:
        List of finding dicts: [{"flag": ..., "issue": ...}, ...]
    """
    resolved_client = client or get_instructor_client()
    visit_text = json.dumps(visit, ensure_ascii=False, indent=2)

    result, completion = await resolved_client.chat.completions.create_with_completion(
        model=model,
        response_model=_Findings,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": visit_text},
        ],
        temperature=0.7,
        #extra_body={"guided_json": _JSON_SCHEMA},
    )

    finish_reason = completion.choices[0].finish_reason
    if finish_reason != "stop":
        logger.error(
            "[validations] unexpected finish_reason=%r; full response: %s",
            finish_reason,
            completion.model_dump_json(indent=2),
        )
    logger.debug("[validations] raw LLM answer: %s", result.model_dump_json(indent=2))
    tokens = completion.usage.total_tokens if completion.usage else 0
    return [{"flag": f.flag, "issue": f.issue} for f in result.findings], tokens
