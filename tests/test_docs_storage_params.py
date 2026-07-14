from storage import docs_storage
from storage.models import Doc


def test_doc_params_keys_are_minimal():
    doc = Doc(file_id="F1", chunk="txt", metadata={"section": "1.1"}, embedding=[0.1, 0.2])
    params = docs_storage._doc_params(doc)
    assert set(params) == {"file_id", "chunk", "metadata", "embedding"}
    assert params["embedding"] == [0.1, 0.2]


def test_insert_sql_mentions_embedding_not_hyde():
    sql = docs_storage._INSERT_DOC_SQL
    assert "embedding" in sql
    for gone in ("fact_q", "procedure_q", "constraint_q"):
        assert gone not in sql


def test_row_to_doc_reads_embedding():
    row = {"id": "u1", "file_id": "F1", "chunk": "c", "metadata": {},
           "embedding": [0.3], "g_name": "N", "g_mkb": ["I63"], "g_age_category": []}
    doc = docs_storage._row_to_doc(row)
    assert doc.embedding == [0.3]
    assert doc.name == "N" and doc.mkb == ["I63"]
    assert not hasattr(doc, "fact_q")
