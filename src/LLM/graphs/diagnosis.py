from __future__ import annotations

from collections.abc import Callable
from typing import Any

from LLM.graphs.diagnosis_nodes import (
    collect_sources,
    extract_drugs,
    generate_questions,
    judge_anamnesis,
    judge_criteria,
    judge_inspection,
    judge_treatment,
    lookup_drugs,
    retrieve,
    retrieve_criteria,
)
from LLM.graphs.diagnosis_state import DiagnosisState


def build_diagnosis_graph(
    *,
    node_overrides: dict[str, Callable[..., Any]] | None = None,
    **compile_kwargs: Any,
):
    """Build the acyclic diagnosis audit graph."""
    from langgraph.graph import END, START, StateGraph

    nodes = {
        "generate_questions": generate_questions,
        "extract_drugs": extract_drugs,
        "lookup_drugs": lookup_drugs,
        "retrieve": retrieve,
        "judge_anamnesis": judge_anamnesis,
        "judge_inspection": judge_inspection,
        "judge_treatment": judge_treatment,
        "retrieve_criteria": retrieve_criteria,
        "judge_criteria": judge_criteria,
        "collect_sources": collect_sources,
    }
    nodes.update(node_overrides or {})
    graph = StateGraph(DiagnosisState)
    for name, node in nodes.items():
        graph.add_node(name, node)

    graph.add_edge(START, "generate_questions")
    graph.add_edge(START, "extract_drugs")
    graph.add_edge(START, "retrieve_criteria")
    graph.add_edge("generate_questions", "retrieve")
    graph.add_edge("extract_drugs", "lookup_drugs")
    graph.add_edge("retrieve", "judge_anamnesis")
    graph.add_edge("retrieve", "judge_inspection")
    graph.add_edge(["retrieve", "lookup_drugs"], "judge_treatment")
    graph.add_edge("retrieve_criteria", "judge_criteria")
    graph.add_edge(
        ["judge_anamnesis", "judge_inspection", "judge_treatment", "judge_criteria"],
        "collect_sources",
    )
    graph.add_edge("collect_sources", END)
    return graph.compile(**compile_kwargs)
