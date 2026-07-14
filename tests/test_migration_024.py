from pathlib import Path

_SQL = (Path(__file__).resolve().parent.parent
        / "migrations" / "024_docs_single_embedding.sql").read_text(encoding="utf-8")


def test_drops_all_six_hyde_columns():
    for col in ("fact_q", "procedure_q", "constraint_q",
                "fact_q_embedding", "procedure_q_embedding", "constraint_q_embedding"):
        assert f"DROP COLUMN IF EXISTS {col}" in _SQL


def test_adds_single_embedding_column_and_index():
    assert "ADD COLUMN IF NOT EXISTS embedding VECTOR(1024)" in _SQL
    assert "docs_embedding_idx" in _SQL
    assert "hnsw (embedding vector_cosine_ops)" in _SQL


def test_drops_old_indexes():
    for idx in ("docs_fact_q_embedding_idx",
                "docs_procedure_q_embedding_idx",
                "docs_constraint_q_embedding_idx"):
        assert f"DROP INDEX IF EXISTS {idx}" in _SQL
