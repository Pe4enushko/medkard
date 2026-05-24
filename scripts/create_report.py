#!/usr/bin/env python3
"""
Export done_cards rows for a date range to an Excel report.

Run from project root:
    python scripts/create_report.py --from 2024-01-01 --to 2024-01-31
    python scripts/create_report.py --from 2024-01-01 --to 2024-01-31 --excel my_report.xlsx

Dates are inclusive on both ends (time range: 00:00:00 of date_from
to 23:59:59.999… of date_to).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audit.excel_formatter import ExcelFormatter  # noqa: E402

# ── Args ──────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Create Excel report from done_cards for a date range")
_parser.add_argument("--from", dest="date_from", required=True, metavar="YYYY-MM-DD", help="Start date (inclusive)")
_parser.add_argument("--to",   dest="date_to",   required=True, metavar="YYYY-MM-DD", help="End date (inclusive)")
_parser.add_argument("--excel", default=None, metavar="PATH", help="Output xlsx file (default: report_<from>_<to>.xlsx)")
_args = _parser.parse_args()


def _parse_date(value: str, label: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        _parser.error(f"--{label}: expected YYYY-MM-DD, got {value!r}")


date_from = _parse_date(_args.date_from, "from")
date_to   = _parse_date(_args.date_to,   "to") + timedelta(days=1)  # make end-inclusive

excel_path = Path(
    _args.excel
    if _args.excel
    else ROOT / f"report_{_args.date_from}_to_{_args.date_to}.xlsx"
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


async def main() -> None:
    log.info("Creating report for %s — %s → %s", _args.date_from, _args.date_to, excel_path)
    async with ExcelFormatter(excel_path) as fmt:
        written = await fmt.export_period(date_from, date_to)
    if written:
        log.info("Done: wrote %d row(s) to %s", written, excel_path)
    else:
        log.info("No records found for the given period.")


if __name__ == "__main__":
    asyncio.run(main())
