from __future__ import annotations

import sys
import types

from LLM.graphs.diagnosis import build_diagnosis_graph


def test_graph_is_acyclic_and_joins_treatment_and_final_collection(monkeypatch) -> None:
    fake_graph = types.ModuleType("langgraph.graph")
    fake_graph.START = "__start__"
    fake_graph.END = "__end__"

    class StateGraph:
        latest = None

        def __init__(self, state_type):
            self.state_type = state_type
            self.nodes = {}
            self.edges = []
            StateGraph.latest = self

        def add_node(self, name, node):
            self.nodes[name] = node

        def add_edge(self, source, target):
            self.edges.append((source, target))

        def compile(self, **kwargs):
            self.compile_kwargs = kwargs
            return self

    fake_graph.StateGraph = StateGraph
    fake_package = types.ModuleType("langgraph")
    fake_package.__path__ = []
    monkeypatch.setitem(sys.modules, "langgraph", fake_package)
    monkeypatch.setitem(sys.modules, "langgraph.graph", fake_graph)

    compiled = build_diagnosis_graph(debug=True)

    assert compiled.compile_kwargs == {"debug": True}
    assert (["retrieve", "lookup_drugs"], "judge_treatment") in compiled.edges
    assert (
        ["judge_anamnesis", "judge_inspection", "judge_treatment", "judge_criteria"],
        "collect_sources",
    ) in compiled.edges
    assert ("collect_sources", "__end__") in compiled.edges

    adjacency = {node: [] for node in compiled.nodes}
    for source, target in compiled.edges:
        for one_source in source if isinstance(source, list) else [source]:
            if one_source in adjacency and target in adjacency:
                adjacency[one_source].append(target)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        assert node not in visiting, f"cycle through {node}"
        if node in visited:
            return
        visiting.add(node)
        for child in adjacency[node]:
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in adjacency:
        visit(node)


async def test_real_graph_merges_parallel_updates() -> None:
    async def generate(state):
        return {"questions": [], "tokens": 2, "errors": ["questions fallback"]}

    async def extract(state):
        return {"drug_mentions": [], "tokens": 3}

    async def lookup(state):
        return {"drug_context": "none"}

    async def retrieve(state):
        return {
            "pools": {"anamnesis": [], "inspection": [], "treatment": []},
            "errors": ["empty retrieval"],
        }

    async def criteria(state):
        return {"pools": {"criteria": []}}

    def judge(aspect):
        async def run(state):
            return {"issues": {aspect: []}, "tokens": 1}

        return run

    async def collect(state):
        assert set(state["pools"]) == {
            "anamnesis",
            "inspection",
            "treatment",
            "criteria",
        }
        assert set(state["issues"]) == {
            "anamnesis",
            "inspection",
            "treatment",
            "criteria",
        }
        return {"sources": []}

    graph = build_diagnosis_graph(
        node_overrides={
            "generate_questions": generate,
            "extract_drugs": extract,
            "lookup_drugs": lookup,
            "retrieve": retrieve,
            "retrieve_criteria": criteria,
            "judge_anamnesis": judge("anamnesis"),
            "judge_inspection": judge("inspection"),
            "judge_treatment": judge("treatment"),
            "judge_criteria": judge("criteria"),
            "collect_sources": collect,
        }
    )

    result = await graph.ainvoke({"pools": {}, "issues": {}, "errors": [], "tokens": 0})

    assert result["tokens"] == 9
    assert result["errors"] == ["questions fallback", "empty retrieval"]
    assert result["sources"] == []
