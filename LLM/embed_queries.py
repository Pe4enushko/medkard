"""
embed_queries.py — embed HypotheticalQueries into vectors.

Calls the configured embedding model (via RAG.retrieval.vector_store.embed)
for each of the three query fields and returns a typed result object.

Usage::
    from LLM.embed_queries import embed_queries
    from LLM.query_generator import HypotheticalQueries

    embeddings = await embed_queries(queries)
    # embeddings.fact_embedding, .procedural_embedding, .constraint_embedding
"""

import asyncio
from dataclasses import dataclass

from RAG.retrieval.vector_store import embed
from LLM.query_generator import HypotheticalQueries


@dataclass
class QueryEmbeddings:
    """Embedding vectors for each of the three hypothetical query types."""

    fact_embedding: list[float]
    procedural_embedding: list[float]
    constraint_embedding: list[float]


async def embed_queries(queries: HypotheticalQueries) -> QueryEmbeddings:
    """Embed all three hypothetical queries concurrently.

    Args:
        queries: HypotheticalQueries produced by generate_queries().

    Returns:
        QueryEmbeddings with one vector per query field.
    """
    fact_emb, procedural_emb, constraint_emb = await asyncio.gather(
        embed(queries.fact_query),
        embed(queries.procedural_query),
        embed(queries.constraint_query),
    )
    return QueryEmbeddings(
        fact_embedding=fact_emb,
        procedural_embedding=procedural_emb,
        constraint_embedding=constraint_emb,
    )
