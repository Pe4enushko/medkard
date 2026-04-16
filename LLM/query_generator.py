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
import os
from pathlib import Path

import instructor
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

# ── Configurable ──────────────────────────────────────────────────────────────
MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
PROMPTS_DIR: Path = Path(__file__).parent / "prompts"
SCHEMAS_DIR: Path = Path(__file__).parent / "schemas"
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE: str = (PROMPTS_DIR / "chunk_query_generator.txt").read_text(encoding="utf-8")
_JSON_SCHEMA: dict = json.loads((SCHEMAS_DIR / "hypothetical_queries.json").read_text(encoding="utf-8"))

# Lazily initialised module-level client; can be overridden in tests by passing
# a client explicitly to generate_queries().
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
    client: instructor.AsyncInstructor | None = None,
    model: str = MODEL,
) -> tuple[dict, HypotheticalQueries]:
    """Generate 3 hypothetical queries for a single content chunk.

    Args:
        chunk:  A chunk dict from PDFContentReader.iter_chunks().
        client: Optional pre-built instructor client (useful for testing or
                when reusing a client across many calls). Falls back to the
                module-level singleton backed by AsyncOpenAI().
        model:  LLM model identifier. Defaults to MODULE-level MODEL constant.

    Returns:
        (chunk, HypotheticalQueries) — the original chunk paired with the
        three generated queries.
    """
    resolved_client = client or _get_client()
    prompt = _PROMPT_TEMPLATE.replace("{chunk}", _render_content(chunk))

    queries: HypotheticalQueries = await resolved_client.chat.completions.create(
        model=model,
        response_model=HypotheticalQueries,
        messages=[{"role": "user", "content": prompt}],
        # extra_body activates guided JSON decoding on vLLM-compatible endpoints;
        # silently ignored by the standard OpenAI API.
        extra_body={"guided_json": _JSON_SCHEMA},
    )

    return chunk, queries
