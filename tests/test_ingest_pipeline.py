from RAG.ingestion import pipeline


def test_chunk_text_passthrough_str():
    assert pipeline.chunk_text({"content": "hello"}) == "hello"


def test_chunk_text_serializes_list_as_json():
    assert pipeline.chunk_text({"content": [{"a": "1"}]}) == '[{"a": "1"}]'


def test_embed_text_prepends_section():
    chunk = {"content": "body", "metadata": {"section": "3.1.1 Диагностика"}}
    assert pipeline.embed_text(chunk) == "[3.1.1 Диагностика]\nbody"


def test_embed_text_without_section_is_body_only():
    assert pipeline.embed_text({"content": "body", "metadata": {}}) == "body"


def test_embed_text_table_serialized():
    chunk = {"content": [{"a": "1"}], "metadata": {"section": "S"}}
    assert pipeline.embed_text(chunk) == '[S]\n[{"a": "1"}]'


async def test_process_chunk_builds_doc_with_embedding(monkeypatch):
    captured = {}

    async def fake_embed(text):
        captured["text"] = text
        return [0.5, 0.6]

    monkeypatch.setattr(pipeline, "embed", fake_embed)

    chunk = {"content": "txt", "metadata": {"section": "1.1", "content_type": "text", "chunk_index": 0}}
    doc = await pipeline.process_chunk(chunk, "F1")

    assert doc.file_id == "F1"
    assert doc.chunk == "txt"                       # stored body is raw chunk text
    assert doc.embedding == [0.5, 0.6]
    assert captured["text"] == "[1.1]\ntxt"          # embedded text is contextual
    assert doc.metadata == {"section": "1.1", "content_type": "text", "chunk_index": 0}


async def test_process_chunk_returns_none_on_embed_error(monkeypatch):
    async def boom(text):
        raise RuntimeError("embed down")

    monkeypatch.setattr(pipeline, "embed", boom)
    doc = await pipeline.process_chunk({"content": "t", "metadata": {"section": "1.1"}}, "F1")
    assert doc is None
