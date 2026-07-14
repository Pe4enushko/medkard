from RAG.ingestion import pipeline


class _Q:
    fact_query = "f"; procedural_query = "p"; constraint_query = "c"


class _E:
    fact_embedding = [0.1]; procedural_embedding = [0.2]; constraint_embedding = [0.3]


def test_chunk_text_passthrough_str():
    assert pipeline.chunk_text({"content": "hello"}) == "hello"


def test_chunk_text_serializes_list_as_json():
    assert pipeline.chunk_text({"content": [{"a": "1"}]}) == '[{"a": "1"}]'


async def test_process_chunk_builds_doc(monkeypatch):
    async def fake_gen(chunk):
        return (None, _Q())

    async def fake_embed(queries):
        return _E()

    monkeypatch.setattr(pipeline, "generate_queries", fake_gen)
    monkeypatch.setattr(pipeline, "embed_queries", fake_embed)

    chunk = {"content": "txt", "metadata": {"section": "1.1", "content_type": "text", "chunk_index": 0}}
    doc = await pipeline.process_chunk(chunk, "F1")

    assert doc.file_id == "F1"
    assert doc.chunk == "txt"
    assert (doc.fact_q, doc.procedure_q, doc.constraint_q) == ("f", "p", "c")
    assert doc.fact_q_embedding == [0.1]
    assert doc.metadata == {"section": "1.1", "content_type": "text", "chunk_index": 0}


async def test_process_chunk_returns_none_on_llm_error(monkeypatch):
    async def boom(chunk):
        raise RuntimeError("llm down")

    monkeypatch.setattr(pipeline, "generate_queries", boom)
    doc = await pipeline.process_chunk({"content": "t", "metadata": {"page": 3}}, "F1")
    assert doc is None
