from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"


def test_hyde_modules_deleted():
    assert not (_SRC / "LLM" / "query_generator.py").exists()
    assert not (_SRC / "LLM" / "embed_queries.py").exists()
    assert not (_SRC / "LLM" / "prompts" / "chunk_query_generator.txt").exists()


def test_no_source_imports_hyde_modules():
    offenders = []
    for py in _SRC.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if "query_generator" in text or "embed_queries" in text or "HypotheticalQueries" in text:
            offenders.append(py.name)
    assert offenders == [], f"HyDE references remain in: {offenders}"
