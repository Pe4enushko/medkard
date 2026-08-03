"""
Vector store interface for pgvector-backed doc retrieval.

A hybrid_search that fuses vector similarity (Postgres HNSW) with BM25 lexical
ranking (rank_bm25 in Python) via Reciprocal Rank Fusion (RRF).

Hybrid search result shape:
    {
        "id":        str,          # UUID of the docs row
        "chunk":     str,          # original text / serialised table rows
        "metadata":  dict,         # JSONB metadata from docs row
        "rrf_score": float,        # fused rank score (higher = more relevant)
    }
"""

import json
import logging
import os
import re
from urllib.parse import quote_plus

import asyncpg
import httpx
import numpy as np
from dotenv import load_dotenv
from natasha import Doc, Segmenter
from pgvector.asyncpg import register_vector
from rank_bm25 import BM25Okapi

from RAG.retrieval.embeddings import EMBEDDING_DIM, EMBEDDING_MODEL, embed  # noqa: F401
from LLM.observability import emit

load_dotenv()
logger = logging.getLogger(__name__)

# ── Configurable ──────────────────────────────────────────────────────────────
# How many vector-search candidates to fetch before BM25 reranking.
# Actual returned results = top_k;  candidates fetched = top_k * CANDIDATES_FACTOR.
CANDIDATES_FACTOR: int = 6
# RRF constant: higher = rankings are more stable; lower = more weight on top results.
RRF_K: int = 50
RERANK_BASE_URL: str = os.environ.get("RERANK_BASE_URL", "").rstrip("/")
RERANK_MODEL: str = os.environ.get("RERANK_MODEL", "")
RERANK_CANDIDATE_LIMIT: int = int(os.environ.get("RERANK_CANDIDATE_LIMIT", "20"))
RERANK_TIMEOUT_SECONDS: float = float(os.environ.get("RERANK_TIMEOUT_SECONDS", "10"))
# ─────────────────────────────────────────────────────────────────────────────

_SELECT_COLS = """
    id::text,
    chunk,
    metadata
"""

_EXCLUDED_CHUNK_PHRASES = (
    "Список литературы",
)

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
        _pool = await asyncpg.create_pool(_dsn(), init=_init_conn, min_size=2, max_size=5)
    return _pool


async def close_pool() -> None:
    """Gracefully close the connection pool (call on application shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# ── Vector search ─────────────────────────────────────────────────────────────

def _chunk_text_exclusion_clauses() -> list[str]:
    return [
        f"chunk NOT LIKE '%{phrase}%'"
        for phrase in _EXCLUDED_CHUNK_PHRASES
    ]


async def _vector_search(embedding: list[float], limit: int) -> list[dict]:
    """Fetch rows closest to *embedding* in the embedding column (cosine distance)."""
    pool = await _get_pool()
    vec = np.array(embedding, dtype=np.float32)
    where_sql = " AND ".join(["embedding IS NOT NULL", *_chunk_text_exclusion_clauses()])
    rows = await pool.fetch(
        f"""
        SELECT {_SELECT_COLS},
            embedding <=> $1 AS distance
        FROM docs
        WHERE {where_sql}
        ORDER BY distance ASC
        LIMIT $2
        """,
        vec,
        limit,
    )
    return [dict(r) for r in rows]


_SECTION_NUM_RE = re.compile(r"^\d+(?:\.\d+)*")


def _extract_section_number(section: str | None) -> str | None:
    """Leading dotted number of a section title, or None.

    '3.1.2 Наружная терапия' -> '3.1.2'; '3 Лечение' -> '3'; 'Приложение А' -> None.
    """
    m = _SECTION_NUM_RE.match(section or "")
    return m.group(0) if m else None


def _section_like_patterns(anchor_sections: list[str]) -> list[str]:
    """SQL-LIKE patterns covering each anchor section itself and its numbered descendants.

    '3 Лечение'  -> ['3 %', '3.%']      '2.1 Жалобы' -> ['2.1 %', '2.1.%']

    '<num> %' matches the section itself (number + space + title); '<num>.%' matches
    numbered descendants. The dot is a LIKE literal, so '3.1 %'/'3.1.%' do NOT match
    '3.10 …' (a '0', not a space/dot, follows '3.1'). Non-numbered anchors are skipped;
    patterns are de-duplicated by number, order preserved.
    """
    patterns: list[str] = []
    seen: set[str] = set()
    for section in anchor_sections:
        num = _extract_section_number(section)
        if num and num not in seen:
            seen.add(num)
            patterns.append(f"{num} %")
            patterns.append(f"{num}.%")
    return patterns


async def _section_anchor_sections(pool, file_id: str, keyword_like: str) -> list[str]:
    """Distinct numbered section titles in *file_id* whose title matches the keyword.

    *keyword_like* is the already-wrapped LIKE argument, e.g. '%лечен%'.
    """
    rows = await pool.fetch(
        """
        SELECT DISTINCT metadata->>'section' AS section
        FROM docs
        WHERE file_id = $1
          AND lower(metadata->>'section') LIKE $2
          AND metadata->>'section' ~ '^[0-9]'
        """,
        file_id,
        keyword_like,
    )
    return [r["section"] for r in rows if r["section"]]


async def _vector_search_filtered(
    embedding: list[float],
    file_id: str,
    limit: int,
    section_filter: str | None = None,
) -> list[dict]:
    """Fetch rows by cosine distance with file_id, optional section, and text filters."""
    pool = await _get_pool()
    vec = np.array(embedding, dtype=np.float32)

    where_clauses = [
        "embedding IS NOT NULL",
        *_chunk_text_exclusion_clauses(),
        "file_id = $2",
    ]
    params: list = [vec, file_id]

    if section_filter:
        keyword_like = f"%{section_filter}%"
        anchors = await _section_anchor_sections(pool, file_id, keyword_like)
        patterns = _section_like_patterns(anchors)

        params.append(keyword_like)
        kw_idx = len(params)
        params.append(patterns)
        pat_idx = len(params)
        where_clauses.append(
            f"(lower(metadata->>'section') LIKE ${kw_idx} "
            f"OR metadata->>'section' LIKE ANY(${pat_idx}::text[]))"
        )

    where_sql = " AND ".join(where_clauses)

    rows = await pool.fetch(
        f"""
        SELECT {_SELECT_COLS},
               embedding <=> $1 AS distance
        FROM docs
        WHERE {where_sql}
        ORDER BY distance ASC
        LIMIT ${len(params) + 1}
        """,
        *params,
        limit,
    )
    return [dict(r) for r in rows]


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
    top_k: int,
    results: list[dict],
) -> None:
    lines = [
        "🔎 [retrieval] hybrid_search raw chunks",
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


async def rerank_results(query_text: str, results: list[dict], top_k: int) -> list[dict]:
    """Optionally rerank a bounded candidate set through a vLLM `/rerank` API."""
    if not RERANK_BASE_URL or not RERANK_MODEL or not results:
        return results[:top_k]

    candidates = results[:max(top_k, min(RERANK_CANDIDATE_LIMIT, len(results)))]
    payload = {
        "model": RERANK_MODEL,
        "query": query_text,
        "documents": [str(row.get("chunk") or "") for row in candidates],
        "top_n": min(top_k, len(candidates)),
    }
    try:
        async with httpx.AsyncClient(timeout=RERANK_TIMEOUT_SECONDS) as client:
            response = await client.post(f"{RERANK_BASE_URL}/rerank", json=payload)
            response.raise_for_status()
            body = response.json()
        ranked = body.get("results") or []
        reranked: list[dict] = []
        for item in ranked:
            index = item.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(candidates):
                continue
            row = dict(candidates[index])
            row["rerank_score"] = item.get("relevance_score")
            reranked.append(row)
        if reranked:
            logger.info(
                "[retrieval] rerank applied model=%s candidates=%d returned=%d",
                RERANK_MODEL,
                len(candidates),
                len(reranked),
            )
            emit(
                "retrieval_rerank",
                model=RERANK_MODEL,
                candidate_count=len(candidates),
                returned_count=len(reranked),
            )
            return reranked[:top_k]
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        logger.warning("[retrieval] rerank unavailable, using RRF order: %s", str(exc)[:200])
        emit("retrieval_rerank_error", model=RERANK_MODEL, exception_type=type(exc).__name__, exception=str(exc)[:200])
    return results[:top_k]


# ── Public hybrid search ──────────────────────────────────────────────────────

async def hybrid_search(
    query_text: str,
    embedding: list[float],
    top_k: int = 10,
) -> list[dict]:
    """Hybrid retrieval: HNSW vector search → BM25 rerank → RRF fusion.

    Steps:
        1. Fetch top_k * CANDIDATES_FACTOR candidates from Postgres using HNSW
           cosine search on the embedding column.
        2. Re-rank the same candidate set with BM25 against *query_text*.
        3. Apply RRF to merge vector rank and BM25 rank.
        4. Return the top_k highest-scoring results.

    Args:
        query_text:  Raw query string used for BM25 lexical scoring.
        embedding:   Query embedding vector (must match EMBEDDING_DIM).
        top_k:       Number of results to return.

    Returns:
        List of dicts with keys: id, chunk, metadata, rrf_score. Sorted by
        rrf_score descending.
    """
    n_candidates = top_k * CANDIDATES_FACTOR

    candidates = await _vector_search(embedding, n_candidates)
    if not candidates:
        logger.info(
            "🔎 [retrieval] hybrid_search found no chunks top_k=%d query=%r",
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
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, score in ranked:
        row = dict(by_id[doc_id])
        row.pop("distance", None)
        row["rrf_score"] = score
        results.append(row)

    results = await rerank_results(query_text, results, top_k)
    _log_hybrid_chunks(
        query_text=query_text,
        top_k=top_k,
        results=results,
    )
    return results
