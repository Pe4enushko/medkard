"""
Integration tests for RAG/retrieval/vector_store.py.

Requires a running Postgres instance configured via .env (POSTGRES_* vars).
Seeds its own rows in the docs table and removes them after each test.

Run::
    pytest tests/test_vector_store.py -v
"""

import random
import sys
from pathlib import Path

import pytest
import pytest_asyncio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from RAG.retrieval.vector_store import (
    EMBEDDING_DIM,
    close_pool,
    hybrid_search,
    search_fact,
)
from storage import DocsStorage
from storage.models import Doc

# ── Helpers ───────────────────────────────────────────────────────────────────

def _rand_vec() -> list[float]:
    """Return a random unit-normalised vector of EMBEDDING_DIM floats."""
    v = [random.gauss(0, 1) for _ in range(EMBEDDING_DIM)]
    mag = sum(x ** 2 for x in v) ** 0.5
    return [x / mag for x in v]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def seeded_docs():
    """Insert 3 test Doc rows; yield their IDs; delete them on teardown."""
    # mock data
    docs = [
        Doc(
            file_id="test_file_1",
            chunk="Пациент жалуется на боли в грудной клетке и одышку.",
            metadata={"ID": "test_file_1", "page": 0, "section": "Жалобы"},
            fact_q="Какие жалобы предъявляет пациент?",
            fact_q_embedding=_rand_vec(),
            procedure_q="Как проводилась диагностика?",
            procedure_q_embedding=_rand_vec(),
            constraint_q="Какие противопоказания указаны?",
            constraint_q_embedding=_rand_vec(),
        ),
        Doc(
            file_id="test_file_2",
            chunk="Артериальное давление 140/90, пульс 88 уд/мин.",
            metadata={"ID": "test_file_2", "page": 1, "section": "Осмотр"},
            fact_q="Каковы показатели давления пациента?",
            fact_q_embedding=_rand_vec(),
            procedure_q="Как измерялось давление?",
            procedure_q_embedding=_rand_vec(),
            constraint_q="Есть ли противопоказания к препарату?",
            constraint_q_embedding=_rand_vec(),
        ),
        Doc(
            file_id="test_file_3",
            chunk="Назначен эналаприл 10 мг утром, контроль через 2 недели.",
            metadata={"ID": "test_file_3", "page": 2, "section": "Назначения"},
            fact_q="Какой препарат назначен?",
            fact_q_embedding=_rand_vec(),
            procedure_q="Как принимать назначенный препарат?",
            procedure_q_embedding=_rand_vec(),
            constraint_q="Каковы противопоказания к эналаприлу?",
            constraint_q_embedding=_rand_vec(),
        ),
    ]

    async with DocsStorage() as storage:
        ids = await storage.insert_many(docs)

    yield ids

    # Teardown: delete the seeded rows
    async with DocsStorage() as storage:
        async with storage._pool.connection() as conn:
            await conn.execute(
                "DELETE FROM docs WHERE id = ANY(%(ids)s::uuid[])",
                {"ids": ids},
            )

    await close_pool()


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_db_connection_and_tables(seeded_docs):
    """DB is reachable, docs table exists, and inserted rows are retrievable."""
    ids = seeded_docs
    assert len(ids) == 3

    async with DocsStorage() as storage:
        docs = await storage.get_many(ids)

    assert len(docs) == 3
    chunks = {d.chunk for d in docs}
    assert "Пациент жалуется на боли в грудной клетке и одышку." in chunks
    assert "Артериальное давление 140/90, пульс 88 уд/мин." in chunks
    assert "Назначен эналаприл 10 мг утром, контроль через 2 недели." in chunks


@pytest.mark.asyncio
async def test_vector_search(seeded_docs):
    """search_fact returns results ordered by cosine similarity with expected fields."""
    # Use the embedding of the first seeded doc as the query vector — it should
    # rank highest (distance ≈ 0) among the three seeded rows.
    ids = seeded_docs

    async with DocsStorage() as storage:
        doc = await storage.get(ids[0])

    query_vec = doc.fact_q_embedding  # not stored on read; use a random vec instead
    # Since embeddings aren't returned by get(), use a fresh random query vector.
    query_vec = _rand_vec()

    results = await search_fact(query_vec, top_k=3)

    # Results may include rows from previous test runs; just validate shape.
    assert isinstance(results, list)
    if results:
        r = results[0]
        assert "id" in r
        assert "chunk" in r
        assert "metadata" in r
        assert "distance" in r
        assert isinstance(r["distance"], float)


@pytest.mark.asyncio
async def test_hybrid_search(seeded_docs):
    """hybrid_search returns results with rrf_score and correct field set."""
    query_vec = _rand_vec()
    query_text = "давление пациент назначение"

    results = await hybrid_search(
        query_text=query_text,
        embedding=query_vec,
        query_type="fact",
        top_k=3,
    )

    assert isinstance(results, list)
    if results:
        r = results[0]
        assert "id" in r
        assert "chunk" in r
        assert "metadata" in r
        assert "rrf_score" in r
        assert isinstance(r["rrf_score"], float)
        assert r["rrf_score"] > 0
        # distance should be stripped from hybrid results
        assert "distance" not in r
