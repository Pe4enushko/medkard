"""Regression test for a prod FK violation: _full_reingest used to insert docs
rows before the guidelines row existed. docs.file_id FK-references
guidelines.file_id, so a brand-new file_id (never seen before, e.g. a fresh
manifest entry with no prior ingest) failed with:

    ForeignKeyViolation: insert or update on table "docs" violates foreign
    key constraint "docs_file_id_fkey"
    DETAIL: Key (file_id)=(...) is not present in table "guidelines".

The fix in scripts/knowledge/reingest-pdfs.py's _full_reingest is to upsert guidelines
before writing docs. This test exercises the storage layer directly (not the
LLM-backed chunking pipeline) to pin the required ordering.
"""
import os

import psycopg
import pytest

from storage import DocsStorage
from storage.guidelines_storage import GuidelinesStorage
from storage.models import Doc
from storage.models.guideline import Guideline

_DIM = int(os.environ["EMBEDDING_DIM"])
_NEW_FILE_ID = "9999902_1"  # never seen before -> no prior guidelines row


def _doc() -> Doc:
    return Doc(file_id=_NEW_FILE_ID, chunk="c", metadata={}, embedding=[0.0] * _DIM)


async def test_docs_insert_before_guidelines_violates_fk():
    async with DocsStorage() as docs_storage:
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            await docs_storage.replace_by_file_id(_NEW_FILE_ID, [_doc()])


async def test_guidelines_then_docs_succeeds_for_brand_new_file_id():
    async with GuidelinesStorage() as guidelines_storage:
        await guidelines_storage.upsert_many([Guideline(file_id=_NEW_FILE_ID, name="New guideline")])
    try:
        async with DocsStorage() as docs_storage:
            ids = await docs_storage.replace_by_file_id(_NEW_FILE_ID, [_doc()])
            assert len(ids) == 1
    finally:
        async with DocsStorage() as docs_storage:
            await docs_storage.delete_by_file_id(_NEW_FILE_ID)
        async with GuidelinesStorage() as guidelines_storage:
            await guidelines_storage.delete(_NEW_FILE_ID)
