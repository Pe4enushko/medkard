"""
Hypothetical query generator for reverse HyDE.

For each content chunk produced by PDFContentReader.iter_chunks(), generates
3 hypothetical search queries (fact, procedural, constraint) that a medical
professional might use to retrieve this chunk.

Returns a tuple of (chunk, HypotheticalQueries) so callers can pair the
original chunk with its generated queries for downstream embedding.

Usage::
    from LLM.query_generator import generate_queries

    chunk, queries = await generate_queries(chunk)
    # queries.fact_query, queries.procedural_query, queries.constraint_query
"""

import json
import logging
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from src.LLM.base import MODEL, get_openai_client

PROMPTS_DIR: Path = Path(__file__).parent / "prompts"
SCHEMAS_DIR: Path = Path(__file__).parent / "schemas"

_PROMPT_TEMPLATE: str = (PROMPTS_DIR / "chunk_query_generator.txt").read_text(encoding="utf-8")
_JSON_SCHEMA: dict = json.loads((SCHEMAS_DIR / "hypothetical_queries.json").read_text(encoding="utf-8"))


class HypotheticalQueries(BaseModel):
    """Three hypothetical queries covering different retrieval intents for a chunk."""

    fact_query: str
    """Factual 'what' question answered by this chunk."""

    procedural_query: str
    """Procedural 'how' question answered by this chunk."""

    constraint_query: str
    """Constraint question about prohibitions or contraindications in this chunk."""


def _render_content(chunk: dict) -> str:
    """Serialise chunk content to a string suitable for prompt insertion."""
    if chunk["type"] == "table":
        return json.dumps(chunk["content"], ensure_ascii=False, indent=2)
    return chunk["content"]  # already a str for text chunks


async def generate_queries(
    chunk: dict,
    *,
    client: AsyncOpenAI | None = None,
    model: str = MODEL,
) -> tuple[dict, HypotheticalQueries]:
    """Generate 3 hypothetical queries for a single content chunk.

    Args:
        chunk:  A chunk dict from PDFContentReader.iter_chunks().
        client: Optional AsyncOpenAI client (useful for testing or
                when reusing a client across many calls). Falls back to the
                module-level singleton.
        model:  LLM model identifier. Defaults to MODULE-level MODEL constant.

    Returns:
        (chunk, HypotheticalQueries) — the original chunk paired with the
        three generated queries.
    """
    resolved_client = client or get_openai_client()
    prompt = _PROMPT_TEMPLATE.replace("{chunk}", _render_content(chunk))

    completion = await resolved_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"guided_json": _JSON_SCHEMA},
    )

    finish_reason = completion.choices[0].finish_reason
    if finish_reason != "stop":
        logger.error(
            "[query_generator] unexpected finish_reason=%r; full response: %s",
            finish_reason,
            completion.model_dump_json(indent=2),
        )

    raw_content = completion.choices[0].message.content or "{}"
    queries = HypotheticalQueries.model_validate_json(raw_content)
    return chunk, queries
