import os
from contextlib import asynccontextmanager

from storage import DocsStorage
from storage.guidelines_storage import GuidelinesStorage
from storage.models import Doc
from storage.models.guideline import Guideline

_DIM = int(os.environ["EMBEDDING_DIM"])  # docs.embedding is vector(EMBEDDING_DIM)


def _doc(file_id: str, chunk: str) -> Doc:
    return Doc(
        file_id=file_id, chunk=chunk,
        metadata={"section": "1.1", "content_type": "text", "chunk_index": 0},
        embedding=[0.0] * _DIM,
    )


@asynccontextmanager
async def _seeded_file_id(file_id: str, docs: list[Doc]):
    """docs.file_id FK-references guidelines.file_id, so seed a guideline row first
    and tear down docs before guidelines (mirrors test_vector_store.py's fixture)."""
    async with GuidelinesStorage() as guidelines_storage:
        await guidelines_storage.upsert_many([Guideline(file_id=file_id, name="Test guideline")])
    async with DocsStorage() as s:
        await s.replace_by_file_id(file_id, docs)
    try:
        async with DocsStorage() as s:
            yield s
    finally:
        async with DocsStorage() as s:
            await s.delete_by_file_id(file_id)
        async with GuidelinesStorage() as guidelines_storage:
            async with guidelines_storage._pool.connection() as conn:
                await conn.execute(
                    "DELETE FROM guidelines WHERE file_id = %(file_id)s", {"file_id": file_id}
                )


async def test_get_chunk_counts_reflects_inserted_rows():
    async with _seeded_file_id("CC1", [_doc("CC1", "a"), _doc("CC1", "b"), _doc("CC1", "c")]) as s:
        counts = await s.get_chunk_counts()
        assert counts["CC1"] == 3


async def test_get_duplicate_chunk_counts_flags_repeated_text():
    docs = [_doc("CC2", "same"), _doc("CC2", "same"), _doc("CC2", "same"), _doc("CC2", "unique")]
    async with _seeded_file_id("CC2", docs) as s:
        dup = await s.get_duplicate_chunk_counts()
        assert dup["CC2"] == 2  # 3 copies of "same" -> 2 extras


async def test_get_duplicate_chunk_counts_omits_files_without_duplicates():
    async with _seeded_file_id("CC3", [_doc("CC3", "a"), _doc("CC3", "b")]) as s:
        dup = await s.get_duplicate_chunk_counts()
        assert "CC3" not in dup
