"""Append-only JSONL telemetry for LLM and retrieval diagnostics."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_lock = threading.Lock()


def emit(event: str, **fields: Any) -> None:
    path_value = os.environ.get("LLM_OBSERVABILITY_PATH", "logs/llm_observability.jsonl")
    if not path_value:
        return

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    path = Path(path_value)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with _lock:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(line)
    except OSError:
        return
