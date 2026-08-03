#!/usr/bin/env python3
"""Summarize LLM JSONL telemetry without exposing prompt contents."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--card-guid")
    args = parser.parse_args()

    events: list[dict] = []
    for line in args.path.open(encoding="utf-8"):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if args.card_guid and event.get("card_guid") != args.card_guid:
            continue
        events.append(event)

    summary = {
        "events": len(events),
        "traces": sorted({event.get("trace_id") for event in events if event.get("trace_id")}),
        "by_event": dict(collections.Counter(event.get("event", "unknown") for event in events)),
        "by_exception": dict(collections.Counter(event.get("exception_type", "") for event in events if event.get("exception_type"))),
        "by_checker": dict(collections.Counter(event.get("checker", "") for event in events if event.get("checker"))),
        "tool_calls": [
            {
                "trace_id": event.get("trace_id"),
                "attempt": event.get("attempt"),
                "tool": event.get("tool"),
                "args_hash": event.get("args_hash"),
                "input_chars": event.get("input_chars"),
                "output_chars": event.get("output_chars"),
                "duplicate": event.get("duplicate", False),
                "budget_exhausted": event.get("budget_exhausted", False),
                "truncated": event.get("truncated", False),
            }
            for event in events
            if event.get("event") == "agent_tool"
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
