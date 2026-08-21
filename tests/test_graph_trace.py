from __future__ import annotations

import asyncio
import json
import stat
from datetime import datetime, timezone

from audit.graph_trace import emit, trace_context


def _records(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_emit_writes_structured_jsonl_with_bound_identity(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "graphtraces.jsonl"
    monkeypatch.setenv("GRAPH_TRACE_PATH", str(path))

    emit("orphan.event")
    with trace_context("correlation-1", "card-1"):
        emit(
            "test.event",
            payload={"when": datetime(2026, 8, 21, 10, 30, tzinfo=timezone.utc)},
        )

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert _records(path) == [
        {
            "timestamp": _records(path)[0]["timestamp"],
            "event": "test.event",
            "correlation_id": "correlation-1",
            "card_guid": "card-1",
            "payload": {"when": "2026-08-21T10:30:00+00:00"},
        }
    ]
    assert datetime.fromisoformat(_records(path)[0]["timestamp"]).tzinfo is not None


def test_async_tasks_keep_correlation_ids_isolated(tmp_path, monkeypatch) -> None:
    path = tmp_path / "graphtraces.jsonl"
    monkeypatch.setenv("GRAPH_TRACE_PATH", str(path))

    async def worker(correlation_id: str, card_guid: str) -> None:
        with trace_context(correlation_id, card_guid):
            emit("worker.started")
            await asyncio.sleep(0)
            emit("worker.completed")

    async def run() -> None:
        await asyncio.gather(
            worker("correlation-a", "card-a"),
            worker("correlation-b", "card-b"),
        )

    asyncio.run(run())

    records = _records(path)
    by_correlation = {
        correlation_id: [
            row for row in records if row["correlation_id"] == correlation_id
        ]
        for correlation_id in {row["correlation_id"] for row in records}
    }
    assert set(by_correlation) == {"correlation-a", "correlation-b"}
    assert {row["card_guid"] for row in by_correlation["correlation-a"]} == {"card-a"}
    assert {row["card_guid"] for row in by_correlation["correlation-b"]} == {"card-b"}
    assert [row["event"] for row in by_correlation["correlation-a"]] == [
        "worker.started",
        "worker.completed",
    ]
