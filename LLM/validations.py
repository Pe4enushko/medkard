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
import os
from pathlib import Path
from typing import Any

import instructor
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)

# ── Configurable ──────────────────────────────────────────────────────────────
MODEL: str = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
SCHEMAS_DIR: Path = Path(__file__).parent / "schemas"
# ─────────────────────────────────────────────────────────────────────────────

_JSON_SCHEMA: dict = json.loads(
    (SCHEMAS_DIR / "formal_structure_findings.json").read_text(encoding="utf-8")
)

_client: instructor.AsyncInstructor | None = None


def _get_client() -> instructor.AsyncInstructor:
    global _client
    if _client is None:
        base_url = os.environ.get("OPENAI_BASE_URL")
        _client = instructor.from_openai(
            AsyncOpenAI(base_url=base_url) if base_url else AsyncOpenAI(),
            mode=instructor.Mode.JSON,
        )
    return _client


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
    resolved_client = client or _get_client()
    visit_text = json.dumps(visit, ensure_ascii=False, indent=2)

    result, completion = await resolved_client.chat.completions.create_with_completion(
        model=model,
        response_model=_Findings,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": visit_text},
        ],
        extra_body={"guided_json": _JSON_SCHEMA},
        temperature=0.4,
    )

    finish_reason = completion.choices[0].finish_reason
    if finish_reason != "stop":
        logger.error(
            "[validations] unexpected finish_reason=%r; full response: %s",
            finish_reason,
            completion.model_dump_json(indent=2),
        )
    logger.debug("[validations] raw LLM answer: %s", result.model_dump_json(indent=2))
    return [{"flag": f.flag, "issue": f.issue} for f in result.findings]
