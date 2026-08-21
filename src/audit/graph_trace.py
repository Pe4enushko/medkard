"""Append-only structured trace for one card audit across async boundaries."""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import fcntl
import json
import os
import threading
import uuid
from collections.abc import Iterator
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "audit_correlation_id", default=None
)
_card_guid: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "audit_card_guid", default=None
)
_thread_lock = threading.Lock()


def new_correlation_id() -> str:
    """Return a new correlation id for one card audit."""
    return str(uuid.uuid4())


def current_correlation_id() -> str | None:
    return _correlation_id.get()


@contextlib.contextmanager
def trace_context(correlation_id: str, card_guid: str | None) -> Iterator[None]:
    """Bind trace identity to the current async task and its child calls."""
    correlation_token = _correlation_id.set(correlation_id)
    card_token = _card_guid.set(card_guid)
    try:
        yield
    finally:
        _card_guid.reset(card_token)
        _correlation_id.reset(correlation_token)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": str(value),
            "repr": repr(value),
        }
    return repr(value)


def emit(
    event: str,
    *,
    correlation_id: str | None = None,
    card_guid: str | None = None,
    **fields: Any,
) -> None:
    """Append one JSON object; tracing must never interrupt the audit itself."""
    path_value = os.environ.get("GRAPH_TRACE_PATH", "logs/graphtraces.jsonl")
    if not path_value:
        return

    resolved_correlation_id = correlation_id or _correlation_id.get()
    if resolved_correlation_id is None:
        return

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "correlation_id": resolved_correlation_id,
        "card_guid": card_guid if card_guid is not None else _card_guid.get(),
        **fields,
    }
    try:
        payload = (
            json.dumps(
                _jsonable(record),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _thread_lock:
            descriptor = os.open(
                path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written == 0:
                        raise OSError("zero-byte graph trace write")
                    view = view[written:]
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
    except Exception:  # noqa: BLE001 - tracing must never break a card audit
        return
