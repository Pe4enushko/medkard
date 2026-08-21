"""
rag_agent.py — LangGraph ReAct agent with a RAG retrieval tool.

The agent is equipped with a single tool — ``retrieve`` — that performs
hybrid search (HNSW vector + BM25 via RRF) against the docs table and
returns the top matching chunks as plain text.

The agent reasons over the retrieved context and the clinical input it
receives as the human message, then returns a final JSON answer.

Usage::
    from LLM.rag_agent import create_rag_agent

    agent = await create_rag_agent(system_prompt)
    result = await agent.ainvoke({"messages": [("user", clinical_text)]})
    # result["messages"][-1].content is the agent's final answer string
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool, tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from RAG.retrieval.embeddings import embed
from RAG.retrieval.vector_store import hybrid_search

load_dotenv()

# ── Configurable ──────────────────────────────────────────────────────────────
MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
RAG_TOP_K: int = int(os.environ.get("RAG_AGENT_TOP_K", "5"))
AGENT_TEMPERATURE: float = float(os.environ.get("LLM_AGENT_TEMPERATURE", "0.2"))
AGENT_MAX_OUTPUT_TOKENS: int = int(
    os.environ.get("LLM_AGENT_MAX_OUTPUT_TOKENS", "2048")
)
# ─────────────────────────────────────────────────────────────────────────────


@tool
async def retrieve(query: str) -> str:
    """Search the medical knowledge base for clinical guidelines, ICD coding rules,
    and diagnostic criteria relevant to the query.

    Returns the most relevant text chunks from the knowledge base.
    Use this to look up clinical criteria, coding standards, or treatment guidelines.

    Args:
        query: A natural-language search query in Russian or English.
    """
    embedding = await embed(query)
    results = await hybrid_search(
        query_text=query,
        embedding=embedding,
        top_k=RAG_TOP_K,
    )
    if not results:
        return "По данному запросу ничего не найдено в базе знаний."

    parts: list[str] = []
    for i, doc in enumerate(results, start=1):
        chunk: str = doc.get("chunk", "")
        meta: dict = doc.get("metadata", {})
        section: str = meta.get("section") or ""
        header = f"[{i}]" + (f" {section}" if section else "")
        parts.append(f"{header}\n{chunk}")

    return "\n\n---\n\n".join(parts)


def _sum_agent_tokens(result: dict) -> int:
    """Sum total_tokens across all AIMessage objects in an agent.ainvoke result."""
    total = 0
    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage) and msg.usage_metadata:
            total += msg.usage_metadata.get("total_tokens", 0)
    return total


class ToolCallGuard:
    def __init__(self, max_calls: int, max_result_chars: int) -> None:
        self.max_calls = max_calls
        self.max_result_chars = max_result_chars
        self.calls = 0
        self.seen: set[str] = set()
        self.events: list[dict[str, Any]] = []

    def wrap(self, tools: list) -> list:
        wrapped = []
        for original in tools:

            async def guarded(*, _original=original, **kwargs: Any) -> str:
                args_text = json.dumps(
                    kwargs, ensure_ascii=False, sort_keys=True, default=str
                )
                args_hash = hashlib.sha256(args_text.encode("utf-8")).hexdigest()[:16]
                key = f"{_original.name}:{args_hash}"
                if key in self.seen:
                    self.events.append(
                        {
                            "tool": _original.name,
                            "args_hash": args_hash,
                            "input": kwargs,
                            "duplicate": True,
                        }
                    )
                    return "Остановка: этот инструмент с такими аргументами уже вызывался. Перейди к итоговому ответу."
                if self.calls >= self.max_calls:
                    self.events.append(
                        {
                            "tool": _original.name,
                            "args_hash": args_hash,
                            "input": kwargs,
                            "budget_exhausted": True,
                        }
                    )
                    return "Остановка: исчерпан лимит вызовов инструментов. Верни итоговый ответ с имеющимися данными."

                self.seen.add(key)
                self.calls += 1
                result = await _original.ainvoke(kwargs)
                result_text = str(result)
                truncated = len(result_text) > self.max_result_chars
                self.events.append(
                    {
                        "tool": _original.name,
                        "args_hash": args_hash,
                        "input": kwargs,
                        "output": result_text,
                        "input_chars": len(args_text),
                        "output_chars": len(result_text),
                        "truncated": truncated,
                    }
                )
                if truncated:
                    result_text = (
                        result_text[: self.max_result_chars]
                        + "\n[tool-result truncated]"
                    )
                return result_text

            wrapped.append(
                StructuredTool.from_function(
                    coroutine=guarded,
                    name=original.name,
                    description=original.description,
                    args_schema=original.args_schema,
                )
            )
        return wrapped


def create_checker_agent(
    system_prompt: str,
    tools: list,
    response_format: type | None = None,
    tool_guard: ToolCallGuard | None = None,
) -> Any:
    """Create a checker agent with custom file-id-bound tools.

    Args:
        system_prompt:   Fully rendered system prompt for the checker.
        tools:           List of tool instances (already file-id bound).
        response_format: Optional Pydantic model passed to LangGraph
                         create_react_agent as response_format.

    Returns:
        A compiled agent graph (``CompiledStateGraph``) ready to invoke via
        ``await agent.ainvoke({"messages": [("user", clinical_text)]})``.
    """
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    from LLM.vllm_config import build_vllm_extra_body

    llm = ChatOpenAI(
        model=MODEL,
        base_url=base_url,
        temperature=AGENT_TEMPERATURE,
        max_completion_tokens=AGENT_MAX_OUTPUT_TOKENS,
        extra_body=build_vllm_extra_body(base_url) or None,
    )

    kwargs: dict[str, Any] = {
        "model": llm,
        "tools": tool_guard.wrap(tools) if tool_guard is not None else tools,
        "prompt": system_prompt,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    return create_react_agent(**kwargs)
