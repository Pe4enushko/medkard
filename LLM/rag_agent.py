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
    result = await agent.ainvoke({"input": clinical_text})
    # result["output"] is the agent's final answer string
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from RAG.retrieval.embeddings import embed
from RAG.retrieval.vector_store import hybrid_search

load_dotenv()

# ── Configurable ──────────────────────────────────────────────────────────────
MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
RAG_TOP_K: int = int(os.environ.get("RAG_AGENT_TOP_K", "5"))
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
    # Run fact, procedure, and constraint searches; merge by taking union over all types.
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

async def create_checker_agent(
    system_prompt: str,
    tools: list,
) -> AgentExecutor:
    """Create a checker agent with custom file-id-bound tools.

    Unlike ``create_rag_agent``, the tools are supplied by the caller (pre-bound
    to a specific guideline document via ``get_tools_for`` / ``get_*_tools_for``).

    Args:
        system_prompt: Fully rendered system prompt for the checker.
        tools:         List of :class:`BaseTool` instances (already file-id bound).

    Returns:
        A ready-to-invoke :class:`AgentExecutor` instance.
    """
    llm = ChatOpenAI(
        model=MODEL,
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=8,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )
