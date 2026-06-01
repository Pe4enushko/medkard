"""
base.py — shared LLM client factory for all modules in the LLM package.

All modules that need an AsyncOpenAI or instructor.AsyncInstructor client
should call the functions here rather than creating their own singletons.
Configuration is read once from the environment.
"""

from __future__ import annotations

import os

import instructor
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

MODEL: str = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")

_openai_client: AsyncOpenAI | None = None
_instructor_client: instructor.AsyncInstructor | None = None


def get_openai_client() -> AsyncOpenAI:
    """Return the shared AsyncOpenAI singleton."""
    global _openai_client
    if _openai_client is None:
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        _openai_client = AsyncOpenAI(base_url=base_url) if base_url else AsyncOpenAI()
    return _openai_client


def get_instructor_client() -> instructor.AsyncInstructor:
    """Return the shared instructor-wrapped AsyncInstructor singleton."""
    global _instructor_client
    if _instructor_client is None:
        _instructor_client = instructor.from_openai(
            get_openai_client(),
            mode=instructor.Mode.JSON,
        )
    return _instructor_client
