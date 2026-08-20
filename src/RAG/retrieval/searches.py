"""Guideline retrieval used by diagnosis graph and ICD checker."""

from __future__ import annotations

import json
import logging

from RAG.retrieval.embeddings import embed
from RAG.retrieval.vector_store import (
    RRF_K,
    _bm25_rank,
    _rrf,
    _vector_search_filtered,
    rerank_results,
)

logger = logging.getLogger(__name__)


def _log_retrieved_chunks(
    query: str,
    file_id: str,
    section_filter: str | None,
    results: list[dict],
) -> None:
    lines = [
        "[retrieval] hybrid_filtered retrieved chunks",
        f"file_id: {file_id}",
        f"section_filter: {section_filter or '—'}",
        f"query: {query}",
        f"count: {len(results)}",
    ]

    for idx, row in enumerate(results, start=1):
        metadata = _metadata_dict(row.get("metadata"))
        section = metadata.get("section") or "—"
        title = metadata.get("title") or metadata.get("doc_title") or "—"
        score = row.get("rrf_score")
        score_text = f"{score:.6f}" if isinstance(score, float) else str(score)
        lines.extend(
            [
                "",
                f"--- chunk {idx} ---",
                f"id: {row.get('id', '—')}",
                f"rrf_score: {score_text}",
                f"title: {title}",
                f"section: {section}",
                str(row.get("chunk") or ""),
            ]
        )

    logger.info("%s", "\n".join(lines))


def _metadata_dict(raw_metadata: object) -> dict:
    if isinstance(raw_metadata, dict):
        return raw_metadata
    if isinstance(raw_metadata, str):
        try:
            parsed = json.loads(raw_metadata)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


# ── Public functions ──────────────────────────────────────────────────────────


async def search_in_guideline(
    query: str,
    file_id: str,
    *,
    candidates: int,
    top_k: int,
) -> list[dict]:
    """Retrieve and rerank chunks from one guideline without a section filter."""
    embedding = await embed(query)
    rows = await _vector_search_filtered(embedding, file_id, candidates)
    if not rows:
        return []

    vector_ranking = [row["id"] for row in rows]
    bm25_ranking = _bm25_rank(query, rows)
    rrf_scores = _rrf([vector_ranking, bm25_ranking], k=RRF_K)
    by_id = {row["id"]: row for row in rows}

    ranked: list[dict] = []
    for doc_id, score in sorted(
        rrf_scores.items(), key=lambda item: item[1], reverse=True
    ):
        row = dict(by_id[doc_id])
        row.pop("distance", None)
        row["rrf_score"] = score
        ranked.append(row)

    results = await rerank_results(query, ranked, top_k)
    _log_retrieved_chunks(
        query=query,
        file_id=file_id,
        section_filter=None,
        results=results,
    )
    return results


async def get_sections_for_file(file_id: str) -> list[str]:
    """Return distinct section names for *file_id* ordered by first chunk_index.

    Used by the ICD checker agent to get the TOC of a guideline before
    deciding which sections to read.
    """
    from RAG.retrieval.vector_store import _get_pool

    pool = await _get_pool()
    rows = await pool.fetch(
        """
        SELECT metadata->>'section' AS section,
               MIN((metadata->>'chunk_index')::INT) AS min_idx
        FROM docs
        WHERE file_id = $1
          AND metadata->>'section' IS NOT NULL
        GROUP BY metadata->>'section'
        ORDER BY min_idx ASC
        """,
        file_id,
    )
    return [r["section"] for r in rows if r["section"]]


async def get_section_chunks(file_id: str, section: str) -> list[dict]:
    """Return all chunks for *file_id* in *section*, ordered by chunk_index.

    Used by the ICD checker agent to read a guideline section sequentially
    (e.g. sections 1.1, 1.2 — definition and classification).
    """
    from RAG.retrieval.vector_store import _get_pool

    pool = await _get_pool()
    rows = await pool.fetch(
        """
        SELECT id::text, chunk, metadata
        FROM docs
        WHERE file_id = $1
          AND metadata->>'section' = $2
        ORDER BY (metadata->>'chunk_index')::INT ASC
        """,
        file_id,
        section,
    )
    return [dict(r) for r in rows]


async def get_section_chunks_by_pattern(
    file_id: str,
    pattern: str,
) -> list[dict]:
    """Return every chunk whose section matches a pattern, in document order."""
    from RAG.retrieval.vector_store import _get_pool

    pool = await _get_pool()
    rows = await pool.fetch(
        """
        SELECT id::text, file_id, chunk, metadata
        FROM docs
        WHERE file_id = $1
          AND metadata->>'section' ILIKE $2
        ORDER BY
          COALESCE((metadata->>'page')::INT, -1) ASC,
          COALESCE((metadata->>'table_index')::INT, -1) ASC,
          COALESCE((metadata->>'chunk_index')::INT, -1) ASC,
          id ASC
        """,
        file_id,
        pattern,
    )
    return [dict(row) for row in rows]
