import inspect

from RAG.retrieval import vector_store


def test_hybrid_search_has_no_query_type():
    params = inspect.signature(vector_store.hybrid_search).parameters
    assert "query_type" not in params
    assert set(params) >= {"query_text", "embedding", "top_k"}


def test_query_type_machinery_removed():
    assert not hasattr(vector_store, "search_fact")
    assert not hasattr(vector_store, "search_procedure")
    assert not hasattr(vector_store, "search_constraint")
    assert not hasattr(vector_store, "_EMBEDDING_COL")


def test_select_cols_have_no_hyde():
    for gone in ("fact_q", "procedure_q", "constraint_q"):
        assert gone not in vector_store._SELECT_COLS
