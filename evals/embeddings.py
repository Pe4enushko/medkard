"""Поход к эндпоинту эмбеддингов.

Переменные — как у medkard (src/RAG/retrieval/embeddings.py): база из
EMBEDDING_BASE_URL, иначе OPENAI_BASE_URL; ключ — OPENAI_API_KEY.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

EMBED_BATCH = 64


def load_env() -> Path | None:
    """.env репозитория. Ищем вверх от файла: evals/ → корень."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if candidate.is_file():
            try:
                from dotenv import load_dotenv
                load_dotenv(candidate)
            except ImportError:  # без python-dotenv разбираем сами: KEY=VALUE
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
            return candidate
    return None


def embed(texts: list[str], base_url: str, api_key: str, model: str,
          progress: bool = True) -> list[list[float]]:
    out: list[list[float]] = []
    url = base_url.rstrip("/") + "/embeddings"
    for start in range(0, len(texts), EMBED_BATCH):
        batch = [t or " " for t in texts[start:start + EMBED_BATCH]]
        body = json.dumps({"model": model, "input": batch}).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.load(resp)
        out.extend(item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"]))
        if progress:
            print(f"    эмбеддинг {min(start + EMBED_BATCH, len(texts))}/{len(texts)}", flush=True)
    return out


def to_pgvector(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"
