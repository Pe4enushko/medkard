"""Shared vLLM sampling and structured-output configuration."""

from __future__ import annotations

import os
from typing import Any


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def is_vllm_endpoint(base_url: str | None = None) -> bool:
    mode = os.environ.get("VLLM_PARAMS_ENABLED", "auto").strip().lower()
    if mode in {"1", "true", "yes", "on"}:
        return True
    if mode in {"0", "false", "no", "off"}:
        return False
    url = base_url or os.environ.get("OPENAI_BASE_URL", "")
    return bool(url) and "api.openai.com" not in url.lower()


def build_vllm_extra_body(base_url: str | None = None) -> dict[str, Any]:
    if not is_vllm_endpoint(base_url):
        return {}

    body: dict[str, Any] = {}
    if _bool("VLLM_DISABLE_THINKING", True):
        body["chat_template_kwargs"] = {"enable_thinking": False}
    body["repetition_penalty"] = _float("VLLM_REPETITION_PENALTY", 1.05)
    body["top_k"] = _int("VLLM_TOP_K", 50)
    body["min_p"] = _float("VLLM_MIN_P", 0.0)
    if _bool("VLLM_REPETITION_DETECTION_ENABLED", True):
        body["repetition_detection"] = {
            "max_pattern_size": _int("VLLM_REPETITION_MAX_PATTERN_SIZE", 20),
            "min_pattern_size": _int("VLLM_REPETITION_MIN_PATTERN_SIZE", 3),
            "min_count": _int("VLLM_REPETITION_MIN_COUNT", 4),
        }
    return body
