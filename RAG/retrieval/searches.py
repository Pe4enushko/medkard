"""
searches.py — targeted hybrid search helpers filtered by file_id and section.

All public functions return raw Doc-like dicts (same shape as hybrid_search).
Use Doc._format_chunk() to render them for LLM prompts.

Public API
----------
search_by_file_id(file_id, query)
    Hybrid search restricted to a single clinical-guideline document.

search_anamnesis(file_id, query)
    Anamnesis / complaints chunks (section contains «анамнез» or «жалоб»).

search_inspection(file_id, query)
    Investigations / lab-results chunks (section contains «исследов»).

search_treatment(file_id, query)
    Treatment chunks (section contains «лечен»).
"""

from __future__ import annotations

import json
import logging

from RAG.retrieval.embeddings import embed
from RAG.retrieval.vector_store import (
    CANDIDATES_FACTOR,
    RRF_K,
    _bm25_rank,
    _rrf,
    _vector_search_filtered,
)

logger = logging.getLogger(__name__)

TARGETED_TOP_K = 4


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _hybrid_filtered(
    query: str,
    file_id: str,
    section_filter: str | None = None,
) -> list[dict]:
    """Core hybrid search (vector + BM25 + RRF) with file_id and optional section filter."""
    logger.info(
        "[retrieval] hybrid_filtered START file_id=%s section_filter=%s top_k=%d query=%r",
        file_id,
        section_filter,
        TARGETED_TOP_K,
        query,
    )
    embedding = await embed(query)
    n_candidates = TARGETED_TOP_K * CANDIDATES_FACTOR

    candidates = await _vector_search_filtered(
        embedding, file_id, n_candidates, section_filter
    )
    if not candidates:
        logger.info(
            "[retrieval] hybrid_filtered returned no candidates file_id=%s section_filter=%s query=%r",
            file_id,
            section_filter,
            query,
        )
        return []

    vector_ranking = [c["id"] for c in candidates]
    bm25_ranking = _bm25_rank(query, candidates)
    rrf_scores = _rrf([vector_ranking, bm25_ranking], k=RRF_K)

    by_id = {c["id"]: c for c in candidates}
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:TARGETED_TOP_K]

    results = []
    for doc_id, score in ranked:
        row = dict(by_id[doc_id])
        row.pop("distance", None)
        row["rrf_score"] = score
        results.append(row)

    _log_retrieved_chunks(
        query=query,
        file_id=file_id,
        section_filter=section_filter,
        results=results,
    )
    return results


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

async def search_by_file_id(
    file_id: str,
    query: str,
) -> list[dict]:
    """Hybrid search restricted to a single document (by file_id).

    Args:
        file_id: ID of the clinical guideline document to search within.
        query:   Natural-language search query.

    Returns:
        List of result dicts with keys: id, chunk, metadata, fact_q,
        procedure_q, constraint_q, rrf_score.
    """
    return await _hybrid_filtered(query, file_id)


async def search_anamnesis(
    file_id: str,
    query: str,
) -> list[dict]:
    """Search anamnesis and complaints sections within a document.

    Filters chunks whose section title (case-insensitive) contains «жалоб».

    Returns raw result dicts (no formatting).
    """
    return await _hybrid_filtered(query, file_id, section_filter="жалоб")


async def search_inspection(
    file_id: str,
    query: str,
) -> list[dict]:
    """Search investigation / diagnostic criteria sections within a document.

    Filters chunks whose section title (case-insensitive) contains «исследов».

    Returns raw result dicts (no formatting).
    """
    return await _hybrid_filtered(query, file_id, section_filter="исследов")


async def search_treatment(
    file_id: str,
    query: str,
) -> list[dict]:
    """Search treatment sections within a document.

    Filters chunks whose section title (case-insensitive) contains «лечен».

    Returns raw result dicts (no formatting).
    """
    return await _hybrid_filtered(query, file_id, section_filter="лечен")
