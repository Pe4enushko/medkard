"""
rag_agent.py — LangChain ReAct agent with a RAG retrieval tool.

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

import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage, RemoveMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from RAG.retrieval.embeddings import embed
from RAG.retrieval.vector_store import hybrid_search

load_dotenv()

# ── Configurable ──────────────────────────────────────────────────────────────
MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
RAG_TOP_K: int = int(os.environ.get("RAG_AGENT_TOP_K", "5"))
# ─────────────────────────────────────────────────────────────────────────────

BAD_TOKENS: tuple[str, ...] = (
    "<|channel|>",
    "<|start|>",
    "<|end|>",
    "<|constrain|>",
    "to=functions.",
)


def _cleanse_text(text: str) -> str:
    for token in BAD_TOKENS:
        text = text.replace(token, "")
    return text


def _cleanse_content(content: Any) -> Any:
    if isinstance(content, str):
        return _cleanse_text(content)
    if isinstance(content, list):
        return [_cleanse_content(item) for item in content]
    if isinstance(content, tuple):
        return tuple(_cleanse_content(item) for item in content)
    if isinstance(content, dict):
        return {key: _cleanse_content(value) for key, value in content.items()}
    return content


def _replace_message_content(message: Any, content: Any) -> Any:
    if isinstance(message, BaseMessage):
        if hasattr(message, "model_copy"):
            return message.model_copy(update={"content": content})
        return message.copy(update={"content": content})

    if isinstance(message, dict):
        return {**message, "content": content}

    if isinstance(message, tuple) and len(message) >= 2:
        return (message[0], content, *message[2:])

    return message


def _cleanse_message(message: Any) -> Any:
    content = (
        message.get("content")
        if isinstance(message, dict)
        else getattr(message, "content", None)
    )
    if isinstance(message, tuple) and len(message) >= 2:
        content = message[1]

    cleansed_content = _cleanse_content(content)
    if cleansed_content == content:
        return message

    return _replace_message_content(message, cleansed_content)


def _cleanse_messages(messages: list[Any]) -> tuple[list[Any], bool]:
    cleansed = [_cleanse_message(message) for message in messages]
    changed = any(
        new_message is not old_message
        for new_message, old_message in zip(cleansed, messages)
    )
    return cleansed, changed


class BadTokenCleansingMiddleware(AgentMiddleware):
    """Remove provider control-token artifacts from every agent message."""

    def before_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        return self._cleanse_state_messages(state)

    async def abefore_model(
        self,
        state: dict[str, Any],
        runtime: Any,
    ) -> dict[str, Any] | None:
        return self._cleanse_state_messages(state)

    def after_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        return self._cleanse_state_messages(state)

    async def aafter_model(
        self,
        state: dict[str, Any],
        runtime: Any,
    ) -> dict[str, Any] | None:
        return self._cleanse_state_messages(state)

    def before_agent(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        return self._cleanse_state_messages(state)

    async def abefore_agent(
        self,
        state: dict[str, Any],
        runtime: Any,
    ) -> dict[str, Any] | None:
        return self._cleanse_state_messages(state)

    def after_agent(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        return self._cleanse_state_messages(state)

    async def aafter_agent(
        self,
        state: dict[str, Any],
        runtime: Any,
    ) -> dict[str, Any] | None:
        return self._cleanse_state_messages(state)

    @staticmethod
    def _cleanse_state_messages(state: dict[str, Any]) -> dict[str, Any] | None:
        messages = state.get("messages")
        if not messages:
            return None

        cleansed_messages, changed = _cleanse_messages(list(messages))
        if not changed:
            return None

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *cleansed_messages,
            ],
        }


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
        query_type="fact",
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


def create_checker_agent(
    system_prompt: str,
    tools: list,
) -> Any:
    """Create a checker agent with custom file-id-bound tools.

    Args:
        system_prompt: Fully rendered system prompt for the checker.
        tools:         List of tool instances (already file-id bound).

    Returns:
        A compiled agent graph (``CompiledStateGraph``) ready to invoke via
        ``await agent.ainvoke({"messages": [("user", clinical_text)]})``.
    """
    llm = ChatOpenAI(
        model=MODEL,
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        temperature=0.4,
    )

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[BadTokenCleansingMiddleware()],
    )
