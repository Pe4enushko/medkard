from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from LLM.client import LLMClient


class _Output(BaseModel):
    value: str


class _FakeAgent:
    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [object()],
            "structured_response": _Output(value="ok"),
        }


def _install_fake_rag_agent(monkeypatch, agent: Any) -> None:
    fake_rag_agent = types.ModuleType("LLM.rag_agent")
    fake_rag_agent._sum_agent_tokens = lambda result: 17
    fake_rag_agent.create_checker_agent = (
        lambda system_prompt, tools, response_format=None: agent
    )
    monkeypatch.setitem(sys.modules, "LLM.rag_agent", fake_rag_agent)


def test_call_agent_returns_langchain_structured_response(monkeypatch) -> None:
    _install_fake_rag_agent(monkeypatch, _FakeAgent())

    client = LLMClient(max_retries=0)

    output, tokens = asyncio.run(
        client.call_agent("system", [], "human", response_format=_Output)
    )

    assert output == _Output(value="ok")
    assert tokens == 17
