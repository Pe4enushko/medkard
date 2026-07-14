import inspect


def test_rag_agent_retrieve_has_no_query_type_literal():
    from LLM import rag_agent
    assert "query_type" not in inspect.getsource(rag_agent)


def test_tools_module_constructs_doc_without_hyde():
    from LLM import tools
    src = inspect.getsource(tools)
    for gone in ("fact_q=", "procedure_q=", "constraint_q="):
        assert gone not in src


def test_get_section_chunks_select_has_no_hyde():
    from RAG.retrieval import searches
    src = inspect.getsource(searches.get_section_chunks)
    for gone in ("fact_q", "procedure_q", "constraint_q"):
        assert gone not in src
