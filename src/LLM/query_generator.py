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

from pydantic import BaseModel

logger = logging.getLogger(__name__)

from LLM.client import LLMClient

PROMPTS_DIR: Path = Path(__file__).parent / "prompts"

_PROMPT_TEMPLATE: str = (PROMPTS_DIR / "chunk_query_generator.txt").read_text(encoding="utf-8")
_client = LLMClient()


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


async def generate_queries(chunk: dict) -> tuple[dict, HypotheticalQueries]:
    """Generate 3 hypothetical queries for a single content chunk.

    Args:
        chunk: A chunk dict from PDFContentReader.iter_chunks().

    Returns:
        (chunk, HypotheticalQueries) — the original chunk paired with the
        three generated queries.
    """
    prompt = _PROMPT_TEMPLATE.replace("{chunk}", _render_content(chunk))

    raw_content, _ = await _client.call(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_model=HypotheticalQueries,
        reasoning_effort="low",  # mechanical extraction — don't waste the token budget on CoT
    )

    try:
        queries = HypotheticalQueries.model_validate_json(raw_content)
    except Exception:
        meta = chunk.get("metadata", {})
        logger.error(
            "HypotheticalQueries JSON parse failed — %s #%s section=%r; "
            "raw LLM output (%d chars): %r",
            meta.get("content_type"), meta.get("chunk_index"), meta.get("section"),
            len(raw_content), raw_content[:500],
        )
        raise
    return chunk, queries
