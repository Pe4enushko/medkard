"""
Vector store interface for pgvector-backed doc retrieval.

Three pure vector-search helpers (search_fact / search_procedure / search_constraint)
plus a hybrid_search that fuses vector similarity (Postgres HNSW) with BM25 lexical
ranking (rank_bm25 in Python) via Reciprocal Rank Fusion (RRF).

Hybrid search result shape:
    {
        "id":        str,          # UUID of the docs row
        "chunk":     str,          # original text / serialised table rows
        "metadata":  dict,         # JSONB metadata from docs row
        "fact_q":    str | None,
        "procedure_q": str | None,
        "constraint_q": str | None,
        "rrf_score": float,        # fused rank score (higher = more relevant)
    }
"""

import json
import logging
import os
from typing import Literal
from urllib.parse import quote_plus

import asyncpg
import numpy as np
from dotenv import load_dotenv
from natasha import Doc, Segmenter
from pgvector.asyncpg import register_vector
from rank_bm25 import BM25Okapi

from RAG.retrieval.embeddings import EMBEDDING_DIM, EMBEDDING_MODEL, embed  # noqa: F401

load_dotenv()
logger = logging.getLogger(__name__)

# ── Configurable ──────────────────────────────────────────────────────────────
# How many vector-search candidates to fetch before BM25 reranking.
# Actual returned results = top_k;  candidates fetched = top_k * CANDIDATES_FACTOR.
CANDIDATES_FACTOR: int = 5
# RRF constant: higher = rankings are more stable; lower = more weight on top results.
RRF_K: int = 60
# ─────────────────────────────────────────────────────────────────────────────

QueryType = Literal["fact", "procedure", "constraint"]

_EMBEDDING_COL: dict[QueryType, str] = {
    "fact":       "fact_q_embedding",
    "procedure":  "procedure_q_embedding",
    "constraint": "constraint_q_embedding",
}

_pool: asyncpg.Pool | None = None
_segmenter: Segmenter = Segmenter()


# ── Connection ────────────────────────────────────────────────────────────────

def _dsn() -> str:
    """Build a properly URL-encoded DSN from individual .env variables.

    Storing the password as a plain string in POSTGRES_PASSWORD and encoding
    it here means special characters (@, :, /, ?, #, etc.) never break the URL.
    """
    user     = os.environ["POSTGRES_USER"]
    password = quote_plus(os.environ["POSTGRES_PASSWORD"])
    host     = os.environ["POSTGRES_HOST"]
    port     = os.environ.get("POSTGRES_PORT", "5432")
    db       = os.environ["POSTGRES_DB"]
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


async def _init_conn(conn: asyncpg.Connection) -> None:
    """Register the pgvector codec so asyncpg can encode/decode VECTOR columns."""
    await register_vector(conn)


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(_dsn(), init=_init_conn)
    return _pool


async def close_pool() -> None:
    """Gracefully close the connection pool (call on application shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# ── Vector search ─────────────────────────────────────────────────────────────

async def _vector_search(
    embedding: list[float],
    col: str,
    limit: int,
) -> list[dict]:
    """Fetch rows closest to *embedding* in *col* using cosine distance."""
    pool = await _get_pool()
    vec = np.array(embedding, dtype=np.float32)
    rows = await pool.fetch(
        f"""
        SELECT
            id::text,
            chunk,
            metadata,
            fact_q,
            procedure_q,
            constraint_q,
            {col} <=> $1 AS distance
        FROM docs
        WHERE {col} IS NOT NULL
        ORDER BY distance ASC
        LIMIT $2
        """,
        vec,
        limit,
    )
    return [dict(r) for r in rows]


async def search_fact(
    embedding: list[float],
    top_k: int = 20,
) -> list[dict]:
    """Return top_k docs ranked by cosine similarity to *embedding* on fact_q_embedding."""
    return await _vector_search(embedding, "fact_q_embedding", top_k)


async def search_procedure(
    embedding: list[float],
    top_k: int = 20,
) -> list[dict]:
    """Return top_k docs ranked by cosine similarity to *embedding* on procedure_q_embedding."""
    return await _vector_search(embedding, "procedure_q_embedding", top_k)


async def search_constraint(
    embedding: list[float],
    top_k: int = 20,
) -> list[dict]:
    """Return top_k docs ranked by cosine similarity to *embedding* on constraint_q_embedding."""
    return await _vector_search(embedding, "constraint_q_embedding", top_k)


# ── Hybrid search internals ───────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Natasha-based tokenisation for Russian medical text."""
    doc = Doc(text.lower())
    doc.segment(_segmenter)
    return [token.text for token in doc.tokens]


def _bm25_rank(query_text: str, candidates: list[dict]) -> list[str]:
    """Return candidate IDs sorted by BM25Okapi score descending."""
    corpus = [_tokenize(c["chunk"]) for c in candidates]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query_text))
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    return [candidates[i]["id"] for i in order]


def _rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion: merge multiple ranked lists into a single score map."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


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


def _log_hybrid_chunks(
    query_text: str,
    query_type: QueryType,
    top_k: int,
    results: list[dict],
) -> None:
    lines = [
        "🔎 [retrieval] hybrid_search raw chunks",
        f"query_type: {query_type}",
        f"top_k: {top_k}",
        f"query: {query_text}",
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


# ── Public hybrid search ──────────────────────────────────────────────────────

async def hybrid_search(
    query_text: str,
    embedding: list[float],
    query_type: QueryType,
    top_k: int = 10,
) -> list[dict]:
    """Hybrid retrieval: HNSW vector search → BM25 rerank → RRF fusion.

    Steps:
        1. Fetch top_k * CANDIDATES_FACTOR candidates from Postgres using HNSW
           cosine search on the column matching *query_type*.
        2. Re-rank the same candidate set with BM25 against *query_text*.
        3. Apply RRF to merge vector rank and BM25 rank.
        4. Return the top_k highest-scoring results.

    Args:
        query_text:  Raw query string used for BM25 lexical scoring.
        embedding:   Query embedding vector (must match EMBEDDING_DIM).
        query_type:  Which embedding column to search — "fact", "procedure",
                     or "constraint".
        top_k:       Number of results to return.

    Returns:
        List of dicts with keys: id, chunk, metadata, fact_q, procedure_q,
        constraint_q, rrf_score. Sorted by rrf_score descending.
    """
    if query_type not in _EMBEDDING_COL:
        raise ValueError(
            f"query_type must be one of {list(_EMBEDDING_COL)}, got {query_type!r}"
        )

    col = _EMBEDDING_COL[query_type]
    n_candidates = top_k * CANDIDATES_FACTOR

    candidates = await _vector_search(embedding, col, n_candidates)
    if not candidates:
        logger.info(
            "🔎 [retrieval] hybrid_search found no chunks query_type=%s top_k=%d query=%r",
            query_type,
            top_k,
            query_text,
        )
        return []

    # Rank by vector similarity (already ordered distance ASC = similarity DESC)
    vector_ranking = [c["id"] for c in candidates]

    # Rank by BM25
    bm25_ranking = _bm25_rank(query_text, candidates)

    # Fuse rankings with RRF
    rrf_scores = _rrf([vector_ranking, bm25_ranking])

    # Assemble results
    by_id = {c["id"]: c for c in candidates}
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for doc_id, score in ranked:
        row = dict(by_id[doc_id])
        row.pop("distance", None)
        row["rrf_score"] = score
        results.append(row)

    _log_hybrid_chunks(
        query_text=query_text,
        query_type=query_type,
        top_k=top_k,
        results=results,
    )
    return results
