#!/usr/bin/env python3
"""
Fetch appointments from 1C for a configured period, save the raw JSON
snapshot, run the full audit pipeline, then export results to Excel.

Run from project root:
    python scripts/audit-one-c-period.py ORG [--days N | --date DD.MM.YYYY] [--ignore-icd CODE ...] [--excel PATH] [--num-batches N] [--ftpcreds FILE]

Options:
    ORG            1C organization: Alenka or MDS
    --days         Shift datebegin N days back from today (default: 0)
    --date         Process the cached single-day search for this exact date
                   (DD.MM.YYYY); used as both datebegin and dateend, selecting
                   the data_snapshots/one_c_<ORG>_<date>_to_<date>.json file
    --ignore-icd   ICD codes to ignore, e.g. Z00.0 J06.9
    --excel        Output xlsx file (default: audit_results.xlsx)
    --num-batches  Max concurrent visits processed at a time (default: 5)
    --ftpcreds     Credentials file for FTP upload (ip=, port=, username=, password=)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.excel_formatter import ExcelFormatter
from audit.pipeline import AuditPipeline
from integrations.ftp import load_creds, upload
from integrations.one_c import AlenkaOneCClient, MdsOneCClient, OneCClient
from audit.filters import CardFilter
from parsers.filter_config import load_card_filter
from parsers.inspection_order import load_inspection_format
from RAG.retrieval.vector_store import close_pool
from storage.organizations_storage import OrganizationsStorage

# ── Args ──────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("org", choices=("Alenka", "MDS"), help="1C organization")
_parser.add_argument("--days", type=int, default=0, help="Shift datebegin N days back from today")
_parser.add_argument("--date", default=None, metavar="DD.MM.YYYY", help="Process the cached single-day search for this exact date (used as both datebegin and dateend)")
_parser.add_argument("-y", action="store_true", help="Skip confirmation prompt")
_parser.add_argument("--excel", default=None, metavar="PATH", help="Output xlsx file (default: report_<datebegin>_to_<dateend>.xlsx)")
_parser.add_argument("--num-batches", type=int, default=5, metavar="N", help="Max concurrent visits processed at a time (default: 5)")
_parser.add_argument("--ftpcreds", default=None, metavar="FILE", help="Credentials file for FTP upload (ip=, port=, username=, password=)")
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

ONE_C_CLIENTS: dict[str, type[OneCClient]] = {
    "Alenka": AlenkaOneCClient,
    "MDS": MdsOneCClient,
}

# ── Config ────────────────────────────────────────────────────────────────────
if _args.date:
    # Process the cached single-day search for the given date.
    try:
        _date = datetime.strptime(_args.date, "%d.%m.%Y")
    except ValueError:
        _parser.error(f"--date must be DD.MM.YYYY, got {_args.date!r}")
    DATEBEGIN = DATEEND = _date.strftime("%d.%m.%Y")
else:
    DATEBEGIN = (datetime.now() - timedelta(days=_args.days)).strftime("%d.%m.%Y")
    DATEEND   = datetime.now().strftime("%d.%m.%Y")

_safe = lambda s: "".join(c if c.isalnum() else "-" for c in s)
EXCEL_PATH = Path(_args.excel) if _args.excel else ROOT / f"report_{_args.org}_{_safe(DATEBEGIN)}_to_{_safe(DATEEND)}.xlsx"

# Period bounds for the DB-backed Excel export (end-inclusive).
PERIOD_FROM = datetime.strptime(DATEBEGIN, "%d.%m.%Y").replace(tzinfo=timezone.utc)
PERIOD_TO   = datetime.strptime(DATEEND,   "%d.%m.%Y").replace(tzinfo=timezone.utc) + timedelta(days=1)
DATA_SNAPSHOTS_DIR = ROOT / "data_snapshots"
LOGS_DIR           = ROOT / "logs"

# ── Logging ───────────────────────────────────────────────────────────────────
DATA_SNAPSHOTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"audit-one-c-period_{_ts}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def _safe_period_part(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value).strip("-")


def _cache_path_for_period(org: str, datebegin: str, dateend: str) -> Path:
    safe_datebegin = _safe_period_part(datebegin)
    safe_dateend = _safe_period_part(dateend)
    return DATA_SNAPSHOTS_DIR / f"one_c_{org}_{safe_datebegin}_to_{safe_dateend}.json"


def _load_or_fetch_one_c_payload(org: str, datebegin: str, dateend: str) -> Any:
    cache_path = _cache_path_for_period(org, datebegin, dateend)
    if cache_path.exists():
        log.info("Using cached 1C response: %s", cache_path)
        return json.loads(cache_path.read_text(encoding="utf-8"))

    log.info("No cached 1C response found for period; fetching from 1C")
    client = ONE_C_CLIENTS[org].from_env()
    payload = client.fetch_json_for_period(datebegin=datebegin, dateend=dateend)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Cached 1C response: %s", cache_path)
    return payload


def _confirm_period(org: str, datebegin: str, dateend: str, card_filter: CardFilter) -> None:
    print(f"Organization: {org}")
    print(f"Period: {datebegin} — {dateend}")
    print(f"Filters:\n{card_filter}")
    if _args.y:
        return
    answer = input("Proceed? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)


async def main() -> None:
    card_filter = load_card_filter(_args.org)
    _confirm_period(_args.org, DATEBEGIN, DATEEND, card_filter)
    log.info("🩺 Starting period audit: org=%s datebegin=%s dateend=%s", _args.org, DATEBEGIN, DATEEND)

    try:
        async with OrganizationsStorage() as organizations:
            org_id = await organizations.get_id_by_name(_args.org)

        # ── 1. Load raw JSON from cache or fetch it from 1C ───────────────────
        payload = _load_or_fetch_one_c_payload(org=_args.org, datebegin=DATEBEGIN, dateend=DATEEND)

        # ── 2. Run pipeline — each card is persisted to DB on completion ──────
        async with AuditPipeline(org_id=org_id, card_filter=card_filter) as pipeline:
            pairs = await pipeline.run_batched(payload, num_batches=_args.num_batches)
        log.info("Pipeline done: %d result(s)", len(pairs))
        if not pairs:
            log.info("Nothing new processed this run; all visits already in DB")

        # ── 3. Export the full period for this org from DB to Excel ───────────
        #     Independent of stage 2: runs whether or not new cards were processed,
        #     so the report always reflects every card in the period.
        async with ExcelFormatter(
            EXCEL_PATH, legacy=_args.legacy_report, order_tokens=INSPECTION_ORDER
        ) as fmt:
            written = await fmt.export_period(PERIOD_FROM, PERIOD_TO, org_id)
        log.info("📊 Exported %d row(s) to %s", written, EXCEL_PATH)

        # ── 4. Upload the report to FTP ───────────────────────────────────────
        #     Independent of stages 2 and 3: runs whenever the report file exists.
        if _args.ftpcreds:
            if EXCEL_PATH.exists():
                try:
                    creds = load_creds(_args.ftpcreds)
                    upload(EXCEL_PATH, EXCEL_PATH.name, creds)
                    log.info("📤 Uploaded %s to FTP", EXCEL_PATH.name)
                except (FileNotFoundError, ValueError) as e:
                    log.error("FTP upload failed: %s", e)
            else:
                log.warning("📤 No report file at %s; skipping FTP upload", EXCEL_PATH)

        log.info("Audit complete. Log: %s", LOG_FILE)
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
