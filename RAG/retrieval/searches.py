"""
searches.py — targeted hybrid search helpers filtered by file_id and section.

All public functions return raw Doc-like dicts (same shape as hybrid_search).
Use Doc._format_chunk() to render them for LLM prompts.

Public API
----------
search_by_file_id(file_id, query, top_k)
    Hybrid search restricted to a single clinical-guideline document.

search_anamnesis(file_id, query, top_k)
    Anamnesis / complaints chunks (section contains «анамнез» or «жалоб»).

search_inspection(file_id, query, top_k)
    Investigations / lab-results chunks (section contains «исследов»).

search_treatment(file_id, query, top_k)
    Treatment chunks (section contains «лечен»).
"""

from __future__ import annotations

import numpy as np

from RAG.retrieval.embeddings import embed
from RAG.retrieval.vector_store import (
    CANDIDATES_FACTOR,
    RRF_K,
    _bm25_rank,
    _get_pool,
    _rrf,
)

# ── SQL fragments ─────────────────────────────────────────────────────────────
_SELECT_COLS = """
    id::text,
    chunk,
    metadata,
    fact_q,
    procedure_q,
    constraint_q
"""

_VECTOR_COL = "fact_q_embedding"  # use fact embeddings for all targeted searches


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _vector_search_filtered(
    embedding: list[float],
    file_id: str,
    limit: int,
    section_filter: str | None = None,
) -> list[dict]:
    """Fetch rows by cosine distance with mandatory file_id filter.

    Args:
        embedding:      Query embedding vector.
        file_id:        Restrict to this file_id value.
        limit:          Maximum rows to return.
        section_filter: If given, adds ``ILIKE '%<section_filter>%'`` on
                        ``metadata->>'section'``.
    """
    pool = await _get_pool()
    vec = np.array(embedding, dtype=np.float32)

    where_clauses = [
        f"{_VECTOR_COL} IS NOT NULL",
        "file_id = $2",
    ]
    params: list = [vec, file_id]

    if section_filter:
        params.append(f"%{section_filter}%")
        where_clauses.append(
            f"lower(metadata->>'section') LIKE ${len(params)}"
        )

    where_sql = " AND ".join(where_clauses)

    rows = await pool.fetch(
        f"""
        SELECT {_SELECT_COLS},
               {_VECTOR_COL} <=> $1 AS distance
        FROM docs
        WHERE {where_sql}
        ORDER BY distance ASC
        LIMIT ${len(params) + 1}
        """,
        *params,
        limit,
    )
    return [dict(r) for r in rows]


async def _hybrid_filtered(
    query: str,
    file_id: str,
    top_k: int,
    section_filter: str | None = None,
) -> list[dict]:
    """Core hybrid search (vector + BM25 + RRF) with file_id and optional section filter."""
    embedding = await embed(query)
    n_candidates = top_k * CANDIDATES_FACTOR

    candidates = await _vector_search_filtered(
        embedding, file_id, n_candidates, section_filter
    )
    if not candidates:
        return []

    vector_ranking = [c["id"] for c in candidates]
    bm25_ranking = _bm25_rank(query, candidates)
    rrf_scores = _rrf([vector_ranking, bm25_ranking], k=RRF_K)

    by_id = {c["id"]: c for c in candidates}
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for doc_id, score in ranked:
        row = dict(by_id[doc_id])
        row.pop("distance", None)
        row["rrf_score"] = score
        results.append(row)

    return results


# ── Public functions ──────────────────────────────────────────────────────────

async def search_by_file_id(
    file_id: str,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Hybrid search restricted to a single document (by file_id).

    Args:
        file_id: ID of the clinical guideline document to search within.
        query:   Natural-language search query.
        top_k:   Number of results to return.

    Returns:
        List of result dicts with keys: id, chunk, metadata, fact_q,
        procedure_q, constraint_q, rrf_score.
    """
    return await _hybrid_filtered(query, file_id, top_k)


async def search_anamnesis(
    file_id: str,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Search anamnesis and complaints sections within a document.

    Filters chunks whose section title (case-insensitive) contains «жалоб».

    Returns raw result dicts (no formatting).
    """
    return await _hybrid_filtered(query, file_id, top_k, section_filter="жалоб")


async def search_inspection(
    file_id: str,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Search investigation / diagnostic criteria sections within a document.

    Filters chunks whose section title (case-insensitive) contains «исследов».

    Returns raw result dicts (no formatting).
    """
    return await _hybrid_filtered(query, file_id, top_k, section_filter="исследов")


async def search_treatment(
    file_id: str,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Search treatment sections within a document.

    Filters chunks whose section title (case-insensitive) contains «лечен».

    Returns raw result dicts (no formatting).
    """
    return await _hybrid_filtered(query, file_id, top_k, section_filter="лечен")
