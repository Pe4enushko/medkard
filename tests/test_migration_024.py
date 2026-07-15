"""Static assertions on the reconcile migration SQL (no DB required)."""
from pathlib import Path

SQL = (Path(__file__).resolve().parent.parent
       / "migrations" / "024_docs_reconcile.sql").read_text()


def test_old_single_embedding_file_is_gone():
    old = (Path(__file__).resolve().parent.parent
           / "migrations" / "024_docs_single_embedding.sql")
    assert not old.exists()


def test_drops_live_stand_hyde_columns():
    assert "DROP COLUMN IF EXISTS chunk_embedding" in SQL
    assert "DROP COLUMN IF EXISTS hyde_reembedded" in SQL
    assert "DROP INDEX IF EXISTS docs_chunk_embedding_hnsw" in SQL


def test_drops_fresh_db_hyde_columns():
    for col in ("fact_q", "procedure_q", "constraint_q",
                "fact_q_embedding", "procedure_q_embedding", "constraint_q_embedding"):
        assert f"DROP COLUMN IF EXISTS {col}" in SQL


def test_keeps_single_embedding_and_index():
    assert "ADD COLUMN IF NOT EXISTS embedding VECTOR(1024)" in SQL
    assert "CREATE INDEX IF NOT EXISTS docs_embedding_idx" in SQL
    assert "hnsw (embedding vector_cosine_ops)" in SQL


def test_has_data_loss_guard():
    assert "RAISE EXCEPTION" in SQL
    assert "embedding IS NULL AND chunk_embedding IS NOT NULL" in SQL
