"""
base.py — shared LLM client factory for all modules in the LLM package.

All modules that need an AsyncOpenAI client should call get_openai_client()
rather than creating their own singletons.
Configuration is read once from the environment.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

MODEL: str = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")

_openai_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    """Return the shared AsyncOpenAI singleton."""
    global _openai_client
    if _openai_client is None:
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        _openai_client = AsyncOpenAI(base_url=base_url) if base_url else AsyncOpenAI()
    return _openai_client
