#!/usr/bin/env python3
"""
Export done_cards rows for a date range to an Excel report.

Run from project root:
    python scripts/operator/create_report.py --from 2024-01-01 --to 2024-01-31
    python scripts/operator/create_report.py --from 2024-01-01 --to 2024-01-31 --output /some/dir
    python scripts/operator/create_report.py --from 2024-01-01 --to 2024-01-31 --org MDS

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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.excel_formatter import ExcelFormatter  # noqa: E402
from parsers.inspection_order import load_inspection_format  # noqa: E402
from storage.organizations_storage import OrganizationsStorage  # noqa: E402

# ── Args ──────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Create Excel report from done_cards for a date range")
_parser.add_argument("--from", dest="date_from", required=True, metavar="YYYY-MM-DD", help="Start date (inclusive)")
_parser.add_argument("--to",   dest="date_to",   required=True, metavar="YYYY-MM-DD", help="End date (inclusive)")
_parser.add_argument("--output", default=None, metavar="DIR", help="Directory for the report file (default: project root)")
_parser.add_argument("--org", required=True, choices=("Alenka", "MDS"), help="Organization to export (required: a report is always scoped to one org)")
_parser.add_argument("--legacy-report", action="store_true", help="Use legacy 3-column Excel layout (visits, formal, diagnosis)")
_parser.add_argument(
    "--format",
    default=None,
    metavar="NAME",
    help="Reorder ДанныеОсмотра fields using resources/inspection_formats.json "
         "[<org>][<NAME>]. Omit to leave field order unchanged.",
)
_args = _parser.parse_args()

INSPECTION_ORDER = (
    load_inspection_format(_args.org, _args.format) if _args.format else None
)


def _parse_date(value: str, label: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        _parser.error(f"--{label}: expected YYYY-MM-DD, got {value!r}")


date_from = _parse_date(_args.date_from, "from")
date_to   = _parse_date(_args.date_to,   "to") + timedelta(days=1)  # make end-inclusive

_auto_name = f"report_{_args.org}_{_args.date_from}_to_{_args.date_to}.xlsx"
excel_path = Path(_args.output) / _auto_name if _args.output else ROOT / _auto_name

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


async def main() -> None:
    async with OrganizationsStorage() as organizations:
        org_id = await organizations.get_id_by_name(_args.org)

    log.info(
        "Creating report for %s — %s (org=%s) → %s",
        _args.date_from, _args.date_to, _args.org, excel_path,
    )
    async with ExcelFormatter(
        excel_path, legacy=_args.legacy_report, order_tokens=INSPECTION_ORDER
    ) as fmt:
        written = await fmt.export_period(date_from, date_to, org_id)
    if written:
        log.info("Done: wrote %d row(s) to %s", written, excel_path)
    else:
        log.info("No records found for the given period.")


if __name__ == "__main__":
    asyncio.run(main())
