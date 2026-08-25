#!/usr/bin/env python3
"""
Filter/dump logs/graphtraces.jsonl for one correlation_id or card_guid.

By default strips the two fields that carry the verbatim patient visit
text — card_data / card_data_priem (the raw "Прием" block, written once by
audit.pipeline._traced_card_audit) and human_message (the same visit,
re-serialised into the LLM prompt by LLM.client.LLMClient.call_agent). Every
other field (messages, output, system_prompt, query, ...) is model-generated
or structural, not the patient's own record, and stays in by default.

Run from project root:
    python scripts/checks/dump-graphtraces.py --correlation-id ID
    python scripts/checks/dump-graphtraces.py --card-guid GUID
    python scripts/checks/dump-graphtraces.py --card-guid GUID --keep-card-data
    python scripts/checks/dump-graphtraces.py --card-guid GUID --path e2e/logs/graphtraces.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

_CARD_DATA_FIELDS = ("card_data", "card_data_priem", "human_message")
_REDACTED = "«card data removed — pass --keep-card-data to see it»"


def _strip_card_data(record: dict) -> dict:
    for field in _CARD_DATA_FIELDS:
        if field in record:
            record[field] = _REDACTED
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correlation-id", help="filter to one correlation_id")
    parser.add_argument("--card-guid", help="filter to one card_guid")
    parser.add_argument(
        "--path",
        type=Path,
        default=ROOT / "logs" / "graphtraces.jsonl",
        help="trace file to read (default: logs/graphtraces.jsonl)",
    )
    parser.add_argument(
        "--keep-card-data",
        action="store_true",
        help="do not redact card_data/card_data_priem/human_message",
    )
    args = parser.parse_args()

    if not args.correlation_id and not args.card_guid:
        parser.error("pass --correlation-id and/or --card-guid")

    if not args.path.exists():
        print(f"нет файла: {args.path}", file=sys.stderr)
        sys.exit(1)

    matched = 0
    with args.path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if args.correlation_id and record.get("correlation_id") != args.correlation_id:
                continue
            if args.card_guid and record.get("card_guid") != args.card_guid:
                continue
            if not args.keep_card_data:
                record = _strip_card_data(record)
            print(json.dumps(record, ensure_ascii=False))
            matched += 1

    print(f"{matched} запись(ей)", file=sys.stderr)


if __name__ == "__main__":
    main()
