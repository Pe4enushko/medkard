"""pipeline.py — shared per-chunk ingest pipeline: chunk → LLM queries → embeddings → Doc.

Extracted from scripts/ingest-pdfs.py so ingest-pdfs.py and reingest-pdfs.py share it.
"""
import asyncio
import json
import logging

from LLM.embed_queries import embed_queries
from LLM.query_generator import generate_queries
from storage.models import Doc

log = logging.getLogger(__name__)


def chunk_text(chunk: dict) -> str:
    content = chunk["content"]
    if isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    return content


async def process_chunk(chunk: dict, file_id: str) -> Doc | None:
    """Generate queries + embeddings for one chunk; return a ready-to-insert Doc (None on LLM error)."""
    text = chunk_text(chunk)
    try:
        _, queries = await generate_queries(chunk)
        embeddings = await embed_queries(queries)
    except Exception as exc:
        log.error(
            "Query/embedding generation failed for %s page %s: %s",
            file_id, chunk["metadata"].get("page"), exc,
        )
        return None

    return Doc(
        file_id=file_id,
        chunk=text,
        metadata=chunk["metadata"],
        fact_q=queries.fact_query,
        procedure_q=queries.procedural_query,
        constraint_q=queries.constraint_query,
        fact_q_embedding=embeddings.fact_embedding,
        procedure_q_embedding=embeddings.procedural_embedding,
        constraint_q_embedding=embeddings.constraint_embedding,
    )


async def process_batch(chunks: list[dict], file_id: str) -> list[Doc | None]:
    """Process a batch of chunks concurrently."""
    return list(await asyncio.gather(*[process_chunk(c, file_id) for c in chunks]))
