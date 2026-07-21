"""Static assertions on the guidelines name-embedding migration SQL (no DB required)."""
from pathlib import Path

SQL = (Path(__file__).resolve().parent.parent
       / "migrations" / "025_guidelines_name_embedding.sql").read_text()


def test_adds_name_embedding_column():
    assert "ADD COLUMN IF NOT EXISTS name_embedding VECTOR(1024)" in SQL


def test_creates_hnsw_index():
    assert "CREATE INDEX IF NOT EXISTS guidelines_name_embedding_idx" in SQL
    assert "hnsw (name_embedding vector_cosine_ops)" in SQL
    assert "m = 16, ef_construction = 64" in SQL


def test_index_is_partial_on_not_null():
    assert "WHERE name_embedding IS NOT NULL" in SQL
