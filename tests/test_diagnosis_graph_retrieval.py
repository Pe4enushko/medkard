from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _load_searches(monkeypatch, *, rows: list[dict], pool=None):
    calls: dict[str, object] = {}

    fake_embeddings = types.ModuleType("RAG.retrieval.embeddings")

    async def embed(query: str) -> list[float]:
        calls["embed"] = query
        return [0.25]

    fake_embeddings.embed = embed
    monkeypatch.setitem(sys.modules, "RAG.retrieval.embeddings", fake_embeddings)

    fake_vector_store = types.ModuleType("RAG.retrieval.vector_store")
    fake_vector_store.CANDIDATES_FACTOR = 6
    fake_vector_store.RRF_K = 50

    async def vector_search(embedding, file_id, limit, section_filter=None):
        calls["vector"] = (embedding, file_id, limit, section_filter)
        return rows

    def bm25_rank(query, candidates):
        calls["bm25"] = (query, candidates)
        return [row["id"] for row in reversed(candidates)]

    def rrf(rankings, *, k):
        calls["rrf"] = (rankings, k)
        return {doc_id: 1.0 / (index + 1) for index, doc_id in enumerate(rankings[1])}

    async def rerank(query, candidates, top_k):
        calls["rerank"] = (query, candidates, top_k)
        return candidates[:top_k]

    async def get_pool():
        return pool

    fake_vector_store._vector_search_filtered = vector_search
    fake_vector_store._bm25_rank = bm25_rank
    fake_vector_store._rrf = rrf
    fake_vector_store.rerank_results = rerank
    fake_vector_store._get_pool = get_pool
    monkeypatch.setitem(sys.modules, "RAG.retrieval.vector_store", fake_vector_store)

    module_path = SRC / "RAG" / "retrieval" / "searches.py"
    spec = importlib.util.spec_from_file_location(
        "diagnosis_graph_test_searches", module_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


@pytest.mark.asyncio
async def test_search_in_guideline_uses_unfiltered_candidate_pool(monkeypatch) -> None:
    rows = [
        {"id": "a", "file_id": "file-1", "chunk": "A", "metadata": {}, "distance": 0.1},
        {"id": "b", "file_id": "file-1", "chunk": "B", "metadata": {}, "distance": 0.2},
    ]
    searches, calls = _load_searches(monkeypatch, rows=rows)

    result = await searches.search_in_guideline(
        "клинический вопрос",
        "file-1",
        candidates=40,
        top_k=5,
    )

    assert calls["vector"] == ([0.25], "file-1", 40, None)
    assert calls["rerank"][0] == "клинический вопрос"
    assert calls["rerank"][2] == 5
    assert [row["id"] for row in result] == ["b", "a"]
    assert all("distance" not in row for row in result)
    assert all("rrf_score" in row for row in result)


@pytest.mark.asyncio
async def test_search_in_guideline_skips_reranker_for_empty_pool(monkeypatch) -> None:
    searches, calls = _load_searches(monkeypatch, rows=[])

    result = await searches.search_in_guideline(
        "question", "file-1", candidates=12, top_k=3
    )

    assert result == []
    assert "rerank" not in calls


@pytest.mark.asyncio
async def test_get_section_chunks_by_pattern_reads_the_complete_table_in_order(
    monkeypatch,
) -> None:
    class Pool:
        async def fetch(self, query, *params):
            self.query = query
            self.params = params
            return [
                {
                    "id": "criteria-1",
                    "file_id": "file-1",
                    "chunk": "table",
                    "metadata": {},
                }
            ]

    pool = Pool()
    searches, _ = _load_searches(monkeypatch, rows=[], pool=pool)

    result = await searches.get_section_chunks_by_pattern("file-1", "%критерии%")

    assert pool.params == ("file-1", "%критерии%")
    assert "ILIKE $2" in pool.query
    assert "LIMIT" not in pool.query
    assert "metadata->>'page'" in pool.query
    assert "metadata->>'table_index'" in pool.query
    assert result[0]["file_id"] == "file-1"
