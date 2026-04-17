"""Embedding adapters for the RAG retrieval pipeline.

The provider is selected at startup via the EMBEDDING_PROVIDER env var:

  openai      — AsyncOpenAI-compatible REST API (default).
                Uses EMBEDDING_BASE_URL (falls back to the official endpoint) and
                OPENAI_API_KEY.  Set EMBEDDING_BASE_URL for self-hosted models
                (e.g. vLLM, Ollama).

  st          — sentence-transformers (local inference, no network call).
                Loads EMBEDDING_MODEL at first call and runs encode() in a thread
                executor to keep the async interface non-blocking.

  fastembed   — fastembed (local ONNX inference, no network call).
                Loads EMBEDDING_MODEL at first call and runs embed() in a thread
                executor to keep the async interface non-blocking.
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from functools import lru_cache

from dotenv import load_dotenv
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource

load_dotenv()

# ── Shared config ─────────────────────────────────────────────────────────────
EMBEDDING_DIM: int = int(os.environ.get("EMBEDDING_DIM", "1024"))
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
# ─────────────────────────────────────────────────────────────────────────────


class EmbeddingAdapter(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return a normalised embedding vector for *text*."""
        ...


class OpenAIEmbeddingAdapter(EmbeddingAdapter):
    """Calls an OpenAI-compatible embeddings endpoint (cloud or self-hosted)."""

    _client = None  # lazy-initialised AsyncOpenAI

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI  # imported lazily so ST users don't need it

            base_url = os.environ.get("EMBEDDING_BASE_URL") or None
            self._client = AsyncOpenAI(base_url=base_url) if base_url else AsyncOpenAI()
        return self._client

    async def embed(self, text: str) -> list[float]:
        response = await self._get_client().embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIM,
        )
        return response.data[0].embedding


class SentenceTransformersAdapter(EmbeddingAdapter):
    """Local inference via sentence-transformers (runs in a thread executor)."""

    _model = None  # lazy-initialised SentenceTransformer

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy import

            self._model = SentenceTransformer(EMBEDDING_MODEL, device="cpu", backend="onnx")
        return self._model

    async def embed(self, text: str) -> list[float]:
        model = self._get_model()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: model.encode(text, normalize_embeddings=True).tolist(),
        )


class FastEmbedAdapter(EmbeddingAdapter):
    """Local inference via fastembed (ONNX runtime, runs in a thread executor)."""

    _model = None  # lazy-initialised TextEmbedding
    

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding  # lazy import
            TextEmbedding.add_custom_model(
                model_id="onnx-community/Qwen3-Embedding-0.6B-ONNX",
                model_name="onnx-community/Qwen3-Embedding-0.6B-ONNX",
                model_source=ModelSource.hf,
                # Для Qwen3-0.6B размер эмбеддинга 1536 (проверьте в config.json на HF)
                dim=1024, 
                description="Qwen3 Embedding 0.6B ONNX version")

            self._model = TextEmbedding(
                model_name=EMBEDDING_MODEL,
                providers=["CPUExecutionProvider"]
            )
        return self._model

    async def embed(self, text: str) -> list[float]:
        model = self._get_model()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: next(iter(model.embed([text]))).tolist(),
        )


@lru_cache(maxsize=1)
def get_adapter() -> EmbeddingAdapter:
    """Return the singleton adapter chosen by EMBEDDING_PROVIDER env var."""
    provider = os.environ.get("EMBEDDING_PROVIDER", "openai").strip().lower()
    if provider == "st":
        return SentenceTransformersAdapter()
    if provider == "fastembed":
        return FastEmbedAdapter()
    return OpenAIEmbeddingAdapter()


async def embed(text: str) -> list[float]:
    """Embed *text* using the provider configured in EMBEDDING_PROVIDER."""
    return await get_adapter().embed(text)
