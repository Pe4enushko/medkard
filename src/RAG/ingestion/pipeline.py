"""pipeline.py — shared per-chunk ingest pipeline: chunk → contextual embedding → Doc.

Shared by scripts/knowledge/ingest-pdfs.py and scripts/knowledge/reingest-pdfs.py. Embeds the chunk's
contextual text ("[section]\\n<body>") — NOT hypothetical queries (reverse HyDE removed).
"""
import asyncio
import json
import logging

from RAG.retrieval.embeddings import embed
from storage.models import Doc

log = logging.getLogger(__name__)


def chunk_text(chunk: dict) -> str:
    content = chunk["content"]
    if isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    return content


def embed_text(chunk: dict) -> str:
    """Contextual text to embed: section header (if any) + chunk body."""
    section = (chunk.get("metadata") or {}).get("section")
    body = chunk_text(chunk)
    return f"[{section}]\n{body}" if section else body


async def process_chunk(chunk: dict, file_id: str) -> Doc | None:
    """Embed the chunk's contextual text; return a ready-to-insert Doc (None on embed error)."""
    body = chunk_text(chunk)
    try:
        vector = await embed(embed_text(chunk))
    except Exception as exc:
        meta = chunk.get("metadata", {})
        log.error(
            "Embedding failed for %s [%s #%s section=%r]: %s",
            file_id, meta.get("content_type"), meta.get("chunk_index"),
            meta.get("section"), exc,
        )
        return None

    return Doc(
        file_id=file_id,
        chunk=body,
        metadata=chunk["metadata"],
        embedding=vector,
    )


async def process_batch(chunks: list[dict], file_id: str) -> list[Doc | None]:
    """Process a batch of chunks concurrently."""
    return list(await asyncio.gather(*[process_chunk(c, file_id) for c in chunks]))
