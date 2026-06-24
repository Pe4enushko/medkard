#!/usr/bin/env python3
"""
Show done_cards_metrics for a given organization.

Run from project root:
    python scripts/metrics.py ORG [--csv PATH]

Options:
    ORG      Organization name (e.g. Alenka, MDS)
    --csv    Also write results to this CSV file
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from storage.base import BaseStorage
from RAG.retrieval.vector_store import close_pool

_parser = argparse.ArgumentParser()
_parser.add_argument("org", help="Organization name (e.g. Alenka, MDS)")
_parser.add_argument("--csv", dest="csv_path", default=None, metavar="PATH")
_args = _parser.parse_args()

_COLUMNS = [
    "visit_date", "total_cards", "audited_cards", "ignored_cards", "broken_cards",
    "total_tokens", "avg_tokens", "wall_clock_sec", "avg_time_sec",
    "organization_name",
]

_COL_WIDTHS = {
    "visit_date":        12,
    "total_cards":        6,
    "audited_cards":      7,
    "ignored_cards":      7,
    "broken_cards":       7,
    "total_tokens":      12,
    "avg_tokens":        10,
    "wall_clock_sec":    14,
    "avg_time_sec":      12,
    "organization_name": 20,
}

_COL_LABELS = {
    "visit_date":        "Date",
    "total_cards":       "Total",
    "audited_cards":     "Audited",
    "ignored_cards":     "Ignored",
    "broken_cards":      "Broken",
    "total_tokens":      "Total tokens",
    "avg_tokens":        "Avg tokens",
    "wall_clock_sec":    "Wall clock (s)",
    "avg_time_sec":      "Avg time (s)",
    "organization_name": "Organization",
}


class _MetricsReader(BaseStorage):
    async def fetch(self, org_name: str) -> list[dict]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM public.done_cards_metrics "
                "WHERE organization_name = %(org)s "
                "ORDER BY visit_date",
                {"org": org_name},
            )
            return await cur.fetchall()


def _cell(value: object, width: int) -> str:
    s = "—" if value is None else str(value)
    return s.ljust(width) if len(s) <= width else s[: width - 1] + "…"


def _print_table(rows: list[dict]) -> None:
    header = "  ".join(_cell(_COL_LABELS[c], _COL_WIDTHS[c]) for c in _COLUMNS)
    sep    = "  ".join("─" * _COL_WIDTHS[c] for c in _COLUMNS)
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(_cell(row[c], _COL_WIDTHS[c]) for c in _COLUMNS))
    print(sep)
    print(f"  {len(rows)} row(s)")


def _write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written: {path}  ({len(rows)} rows)")


async def main() -> None:
    try:
        async with _MetricsReader() as reader:
            rows = await reader.fetch(_args.org)

        if not rows:
            print(f"No data found for organization: {_args.org!r}")
            return

        _print_table(rows)

        if _args.csv_path:
            _write_csv(rows, _args.csv_path)
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
