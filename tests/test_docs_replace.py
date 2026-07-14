import os

from storage import DocsStorage
from storage.models import Doc

_DIM = int(os.environ["EMBEDDING_DIM"])  # docs.*_embedding columns are vector(EMBEDDING_DIM)


def _doc(file_id: str, chunk: str) -> Doc:
    return Doc(
        file_id=file_id, chunk=chunk, metadata={"section": "1.1", "content_type": "text", "chunk_index": 0},
        fact_q="f", procedure_q="p", constraint_q="c",
        fact_q_embedding=[0.0] * _DIM, procedure_q_embedding=[0.0] * _DIM, constraint_q_embedding=[0.0] * _DIM,
    )


async def test_replace_by_file_id_swaps_rows():
    async with DocsStorage() as s:
        await s.replace_by_file_id("RP1", [_doc("RP1", "old-a"), _doc("RP1", "old-b")])
        new_ids = await s.replace_by_file_id("RP1", [_doc("RP1", "new-only")])
        assert len(new_ids) == 1
        rows = await s.get_many(new_ids)
        assert [r.chunk for r in rows] == ["new-only"]
