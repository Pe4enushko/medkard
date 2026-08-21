from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from typing import Any

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.graph_trace import trace_context
from LLM.client import LLMClient


class _Output(BaseModel):
    value: str


class _FakeAgent:
    async def ainvoke(
        self, payload: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "messages": [object()],
            "structured_response": _Output(value="ok"),
        }


class _RecursingThenSuccessfulAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.configs: list[dict[str, Any]] = []

    async def ainvoke(
        self, payload: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls += 1
        self.configs.append(config)
        if self.calls == 1:
            from langgraph.errors import GraphRecursionError

            raise GraphRecursionError("recursion limit")
        return {
            "messages": [object()],
            "structured_response": _Output(value="ok after compact retry"),
        }


def _install_fake_rag_agent(monkeypatch, agent: Any) -> None:
    fake_rag_agent = types.ModuleType("LLM.rag_agent")
    fake_rag_agent._sum_agent_tokens = lambda result: 17
    fake_rag_agent.ToolCallGuard = lambda max_calls, max_result_chars: (
        types.SimpleNamespace(events=[])
    )
    fake_rag_agent.create_checker_agent = (
        lambda system_prompt, tools, response_format=None, tool_guard=None: agent
    )
    monkeypatch.setitem(sys.modules, "LLM.rag_agent", fake_rag_agent)


def test_call_agent_returns_langchain_structured_response(
    monkeypatch, tmp_path
) -> None:
    _install_fake_rag_agent(monkeypatch, _FakeAgent())
    trace_path = tmp_path / "graphtraces.jsonl"
    monkeypatch.setenv("GRAPH_TRACE_PATH", str(trace_path))

    client = LLMClient(max_retries=0)

    with trace_context("correlation-1", "card-1"):
        output, tokens = asyncio.run(
            client.call_agent("system", [], "human", response_format=_Output)
        )

    assert output == _Output(value="ok")
    assert tokens == 17
    records = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["correlation_id"] == "correlation-1" for row in records)
    assert all(row["card_guid"] == "card-1" for row in records)
    assert records[0]["event"] == "llm.agent.started"
    completed = next(row for row in records if row["event"] == "llm.agent.completed")
    assert completed["output"] == {"value": "ok"}


def test_call_agent_retries_recursion_in_compact_mode(monkeypatch) -> None:
    agent = _RecursingThenSuccessfulAgent()
    _install_fake_rag_agent(monkeypatch, agent)

    client = LLMClient(max_retries=1)

    output, tokens = asyncio.run(
        client.call_agent("system", [], "human", response_format=_Output)
    )

    assert output == _Output(value="ok after compact retry")
    assert tokens == 17
    assert agent.calls == 2
    assert agent.configs[0]["recursion_limit"] == 12
    assert agent.configs[1]["recursion_limit"] == 8
