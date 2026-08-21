from __future__ import annotations

from RAG.retrieval import vector_store


async def test_qwen_reranker_templates_query_and_documents(monkeypatch) -> None:
    payloads = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{"index": 1, "relevance_score": 0.9}]}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, json):
            payloads.append((url, json))
            return Response()

    monkeypatch.setattr(vector_store, "RERANK_BASE_URL", "http://reranker")
    monkeypatch.setattr(vector_store, "RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
    monkeypatch.setattr(
        vector_store, "RERANK_QUERY_TEMPLATE", "<I>{instruction}\n<Q>{query}"
    )
    monkeypatch.setattr(vector_store, "RERANK_DOC_TEMPLATE", "<D>{doc}")
    monkeypatch.setattr(vector_store, "RERANK_INSTRUCTION", "Проверь релевантность")
    monkeypatch.setattr(vector_store.httpx, "AsyncClient", lambda **kwargs: Client())

    result = await vector_store.rerank_results(
        "Как лечить?",
        [{"id": "a", "chunk": "A"}, {"id": "b", "chunk": "B"}],
        top_k=1,
    )

    assert payloads == [
        (
            "http://reranker/rerank",
            {
                "model": "Qwen/Qwen3-Reranker-0.6B",
                "query": "<I>Проверь релевантность\n<Q>Как лечить?",
                "documents": ["<D>A", "<D>B"],
                "top_n": 1,
            },
        )
    ]
    assert result == [{"id": "b", "chunk": "B", "rerank_score": 0.9}]


async def test_malformed_reranker_template_falls_back_to_rrf(monkeypatch) -> None:
    monkeypatch.setattr(vector_store, "RERANK_BASE_URL", "http://reranker")
    monkeypatch.setattr(vector_store, "RERANK_MODEL", "model")
    monkeypatch.setattr(vector_store, "RERANK_QUERY_TEMPLATE", "{unknown}")

    rows = [{"id": "a", "chunk": "A"}]

    assert await vector_store.rerank_results("q", rows, top_k=1) == rows
