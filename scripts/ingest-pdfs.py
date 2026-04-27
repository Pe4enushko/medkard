#!/usr/bin/env python3
"""
ingest-pdfs.py — ingest PDF chunks from manifest.csv into the docs table.

Run from the project root::
    python scripts/ingest-pdfs.py

For each chunk: generates hypothetical queries via the LLM, embeds them,
then stores everything in the docs table.
Already-ingested files (by file_id) are skipped automatically.
Progress and errors are written to both stdout and a timestamped log file.
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

# ── ensure project root is on sys.path ───────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from LLM.embed_queries import embed_queries
from LLM.query_generator import generate_queries
from RAG.ingestion.data_loader import load_documents
from storage import DocsStorage
from storage.models import Doc

# ── Configurable ──────────────────────────────────────────────────────────────
# Number of chunks processed concurrently (query generation + embedding per batch).
QUERY_GENERATION_BATCH_SIZE: int = 3
# ─────────────────────────────────────────────────────────────────────────────

# ── Logging setup ─────────────────────────────────────────────────────────────
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

log_filename = LOGS_DIR / f"ingest-pdfs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_text(chunk: dict) -> str:
    content = chunk["content"]
    if isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    return content


async def _process_chunk(chunk: dict) -> Doc | None:
    """Generate queries + embeddings for one chunk; return a ready-to-insert Doc."""
    chunk_text = _chunk_text(chunk)
    file_id: str = chunk["metadata"]["ID"]
    try:
        _, queries = await generate_queries(chunk)
        embeddings = await embed_queries(queries)
    except Exception as exc:
        log.error(
            "Query/embedding generation failed for %s page %s: %s",
            file_id,
            chunk["metadata"].get("page"),
            exc,
        )
        return None

    return Doc(
        file_id=file_id,
        chunk=chunk_text,
        metadata=chunk["metadata"],
        fact_q=queries.fact_query,
        procedure_q=queries.procedural_query,
        constraint_q=queries.constraint_query,
        fact_q_embedding=embeddings.fact_embedding,
        procedure_q_embedding=embeddings.procedural_embedding,
        constraint_q_embedding=embeddings.constraint_embedding,
    )


async def _process_batch(chunks: list[dict]) -> list[Doc | None]:
    """Process a batch of chunks concurrently."""
    return list(await asyncio.gather(*[_process_chunk(c) for c in chunks]))


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("Starting PDF ingestion (batch size: %d)", QUERY_GENERATION_BATCH_SIZE)

    async with DocsStorage() as storage:
        ingested = await storage.get_ingested_file_ids()
        if ingested:
            log.info(
                "Skipping %d already-ingested file(s): %s",
                len(ingested), sorted(ingested),
            )

        exceptions = ingested if ingested else None
        readers = load_documents(exceptions=exceptions)

        total_chunks = 0
        total_errors = 0
        current_file_id: str | None = None

        try:
            for reader in readers:
                current_file_id = reader.metadata["ID"]
                log.info("Ingesting file: %s", current_file_id)
                file_chunks = 0
                file_errors = 0

                # Collect all chunks for this file, then process in batches.
                all_chunks = list(reader.iter_chunks())
                log.info("  %s: %d chunk(s) to process", current_file_id, len(all_chunks))
                for ci, chunk in enumerate(all_chunks):
                    meta = chunk["metadata"]
                    content_preview = _chunk_text(chunk)[:120].replace("\n", " ")
                    log.debug(
                        "  [%s] chunk %d/%d — type=%s page=%s section=%r content=%r",
                        current_file_id,
                        ci + 1,
                        len(all_chunks),
                        chunk.get("type"),
                        meta.get("page"),
                        meta.get("section"),
                        content_preview,
                    )

                for batch_start in range(0, len(all_chunks), QUERY_GENERATION_BATCH_SIZE):
                    batch = all_chunks[batch_start: batch_start + QUERY_GENERATION_BATCH_SIZE]
                    docs = await _process_batch(batch)

                    for chunk, doc in zip(batch, docs):
                        if doc is None:
                            file_errors += 1
                            total_errors += 1
                            continue
                        try:
                            await storage.insert(doc)
                            file_chunks += 1
                        except Exception as exc:
                            file_errors += 1
                            total_errors += 1
                            log.error(
                                "DB insert failed for %s page %s: %s",
                                current_file_id,
                                chunk["metadata"].get("page"),
                                exc,
                            )

                total_chunks += file_chunks
                if file_errors:
                    log.warning(
                        "Finished %s — %d chunk(s) inserted, %d error(s)",
                        current_file_id, file_chunks, file_errors,
                    )
                else:
                    log.info("Finished %s — %d chunk(s) inserted", current_file_id, file_chunks)

                current_file_id = None  # mark as fully committed

        except (KeyboardInterrupt, asyncio.CancelledError):
            log.warning("Interrupted!")
            if current_file_id is not None:
                log.warning(
                    "Removing partial chunks for unfinished file: %s", current_file_id
                )
                deleted = await storage.delete_by_file_id(current_file_id)
                log.warning(
                    "Deleted %d partial chunk(s) for %s", deleted, current_file_id
                )
            log.info(
                "Ingestion interrupted — total chunks committed: %d, total errors: %d",
                total_chunks, total_errors,
            )
            log.info("Log written to %s", log_filename)
            sys.exit(1)

    log.info(
        "Ingestion complete — total chunks: %d, total errors: %d",
        total_chunks, total_errors,
    )
    log.info("Log written to %s", log_filename)


if __name__ == "__main__":
    asyncio.run(main())
