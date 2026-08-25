import importlib.util
import os
from contextlib import asynccontextmanager
from pathlib import Path

from storage import DocsStorage, IngestRunsStorage
from storage.guidelines_storage import GuidelinesStorage
from storage.models import Doc
from storage.models.guideline import Guideline

_spec = importlib.util.spec_from_file_location(
    "reingest_pdfs", Path(__file__).resolve().parent.parent / "scripts" / "knowledge" / "reingest-pdfs.py")
reingest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reingest)

_DIM = int(os.environ["EMBEDDING_DIM"])


@asynccontextmanager
async def _seeded_revision(file_id: str, with_docs: bool = True):
    async with GuidelinesStorage() as guidelines_storage:
        await guidelines_storage.upsert_many([Guideline(file_id=file_id, name="Stale test guideline")])
    if with_docs:
        async with DocsStorage() as docs_storage:
            await docs_storage.replace_by_file_id(
                file_id, [Doc(file_id=file_id, chunk="c", metadata={}, embedding=[0.0] * _DIM)]
            )
    async with IngestRunsStorage() as runs_storage:
        await runs_storage.mark_done(file_id, "deadbeef")
    try:
        yield
    finally:
        async with DocsStorage() as docs_storage:
            await docs_storage.delete_by_file_id(file_id)
        async with GuidelinesStorage() as guidelines_storage:
            await guidelines_storage.delete(file_id)
        async with IngestRunsStorage() as runs_storage:
            await runs_storage.delete(file_id)


_OLD_ID = "9999901_2"
_NEW_ID = "9999901_3"


async def test_purge_stale_revisions_removes_superseded_revision_only():
    async with _seeded_revision(_OLD_ID), _seeded_revision(_NEW_ID):
        manifest_rows = {_NEW_ID: {"ID": _NEW_ID}}
        async with DocsStorage() as docs_storage, \
                GuidelinesStorage() as guidelines_storage, \
                IngestRunsStorage() as runs_storage:
            guidelines_by_id = {g.file_id: g for g in await guidelines_storage.all()}
            stale = await reingest._purge_stale_revisions(
                manifest_rows, guidelines_by_id, docs_storage, guidelines_storage, runs_storage)
            assert stale == [_OLD_ID]

            assert await guidelines_storage.get(_OLD_ID) is None
            assert await guidelines_storage.get(_NEW_ID) is not None
            assert _OLD_ID not in await docs_storage.get_ingested_file_ids()
            assert _NEW_ID in await docs_storage.get_ingested_file_ids()

        # Re-seed the old id's guidelines row so the fixture teardown can clean it up
        # (it was deleted by the purge above).
        async with GuidelinesStorage() as guidelines_storage:
            await guidelines_storage.upsert_many([Guideline(file_id=_OLD_ID, name="x")])
