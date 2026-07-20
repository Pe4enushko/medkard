# tests/test_guidelines_upsert_embeds_name.py
import pytest
from storage.models.guideline import Guideline, name_embed_input


@pytest.mark.asyncio
async def test_upsert_computes_embedding_from_name_and_age(monkeypatch):
    seen_texts = []

    async def fake_embed(text: str) -> list[float]:
        seen_texts.append(text)
        return [0.5] * 1024

    import storage.guidelines_storage as gs
    monkeypatch.setattr(gs, "embed", fake_embed, raising=False)

    # Capture what gets written without a DB: fake the connection/pool.
    written = []

    class FakeConn:
        async def execute(self, sql, params):
            written.append(params)
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class FakePool:
        def connection(self): return FakeConn()

    s = gs.GuidelinesStorage.__new__(gs.GuidelinesStorage)
    s._pool = FakePool()

    g = Guideline(file_id="X1", name="Бронхит", age_category=["Дети"])
    n = await s.upsert_many([g])

    assert n == 1
    assert seen_texts == [name_embed_input("Бронхит", ["Дети"])]  # "Название: Бронхит\nВозрастная группа: [Дети]"
    assert written[0]["name_embedding"] == [0.5] * 1024


@pytest.mark.asyncio
async def test_upsert_respects_preset_embedding(monkeypatch):
    called = False

    async def fake_embed(text: str) -> list[float]:
        nonlocal called
        called = True
        return [0.0] * 1024

    import storage.guidelines_storage as gs
    monkeypatch.setattr(gs, "embed", fake_embed, raising=False)

    written = []

    class FakeConn:
        async def execute(self, sql, params): written.append(params)
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class FakePool:
        def connection(self): return FakeConn()

    s = gs.GuidelinesStorage.__new__(gs.GuidelinesStorage)
    s._pool = FakePool()

    preset = [0.9] * 1024
    g = Guideline(file_id="X2", name="Готовый", name_embedding=preset)
    await s.upsert_many([g])

    assert called is False  # preset embedding must not be recomputed
    assert written[0]["name_embedding"] == preset
