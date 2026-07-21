import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from storage.guidelines_storage import GuidelinesStorage
from storage.models.guideline import Guideline

pytestmark = pytest.mark.skipif(
    not os.environ.get("POSTGRES_HOST"),
    reason="requires stand DB with migration 025 applied",
)


@pytest.mark.asyncio
async def test_name_embedding_is_written():
    # Preset the vector so upsert writes it verbatim (Task 4 adds auto-embed).
    vec = [0.1] * 1024
    g = Guideline(file_id="TEST_NAME_EMB_1", name="Тестовая река",
                  age_category=["Взрослые"], name_embedding=vec)
    async with GuidelinesStorage() as s:
        await s.upsert_many([g])
        async with s._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT name_embedding IS NOT NULL AS has_vec, "
                "vector_dims(name_embedding) AS dims "
                "FROM guidelines WHERE file_id = %(fid)s",
                {"fid": "TEST_NAME_EMB_1"},
            )
            row = await cur.fetchone()
        await s.delete("TEST_NAME_EMB_1")
    assert row["has_vec"] is True
    assert row["dims"] == 1024
