#!/usr/bin/env python3
"""
Fetch appointments from 1C for a configured period, save the raw JSON
snapshot, run the full audit pipeline, then export results to Excel.

Run from project root:
    python scripts/audit-one-c-period.py [--days N] [--ignore-icd CODE ...] [--excel PATH] [--num-batches N]

Options:
    --days         Shift datebegin N days back from today (default: 0)
    --ignore-icd   ICD codes to ignore, e.g. Z00.0 J06.9
    --excel        Output xlsx file (default: audit_results.xlsx)
    --num-batches  Max concurrent visits processed at a time (default: 5)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audit.excel_formatter import ExcelFormatter
from audit.pipeline import AuditPipeline
from integrations.one_c import OneCClient
from RAG.retrieval.vector_store import close_pool

# ── Args ──────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--days", type=int, default=0, help="Shift datebegin N days back from today")
_parser.add_argument("-y", action="store_true", help="Skip confirmation prompt")
_parser.add_argument("--ignore-icd", nargs="*", default=[], metavar="CODE", help="ICD codes to ignore (e.g. Z00.0 J06.9)")
_parser.add_argument("--excel", default=str(ROOT / "audit_results.xlsx"), metavar="PATH", help="Output xlsx file (default: audit_results.xlsx)")
_parser.add_argument("--num-batches", type=int, default=5, metavar="N", help="Max concurrent visits processed at a time (default: 5)")
_args = _parser.parse_args()

IGNORE_ICD: list[str] = _args.ignore_icd

# ── Config ────────────────────────────────────────────────────────────────────
DATEBEGIN = (datetime.now() - timedelta(days=_args.days)).strftime("%d.%m.%Y")
DATEEND   = datetime.now().strftime("%d.%m.%Y")

EXCEL_PATH         = Path(_args.excel)
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


def _cache_path_for_period(datebegin: str, dateend: str) -> Path:
    safe_datebegin = _safe_period_part(datebegin)
    safe_dateend = _safe_period_part(dateend)
    return DATA_SNAPSHOTS_DIR / f"one_c_{safe_datebegin}_to_{safe_dateend}.json"


def _load_or_fetch_one_c_payload(datebegin: str, dateend: str) -> Any:
    cache_path = _cache_path_for_period(datebegin, dateend)
    if cache_path.exists():
        log.info("Using cached 1C response: %s", cache_path)
        return json.loads(cache_path.read_text(encoding="utf-8"))

    log.info("No cached 1C response found for period; fetching from 1C")
    client = OneCClient.from_env()
    payload = client.fetch_json_for_period(datebegin=datebegin, dateend=dateend)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Cached 1C response: %s", cache_path)
    return payload


def _confirm_period(datebegin: str, dateend: str) -> None:
    print(f"Period: {datebegin} — {dateend}")
    if _args.y:
        return
    answer = input("Proceed? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)


async def main() -> None:
    _confirm_period(DATEBEGIN, DATEEND)
    log.info("🩺 Starting period audit: datebegin=%s dateend=%s", DATEBEGIN, DATEEND)

    try:
        # ── 1. Load raw JSON from cache or fetch it from 1C ───────────────────
        payload = _load_or_fetch_one_c_payload(datebegin=DATEBEGIN, dateend=DATEEND)

        # ── 2. Run pipeline — each card is persisted to DB on completion ──────
        async with AuditPipeline() as pipeline:
            pairs = await pipeline.run_batched(payload, num_batches=_args.num_batches, ignore_icd=IGNORE_ICD or None)
        log.info("Pipeline done: %d result(s)", len(pairs))

        if not pairs:
            log.info("Nothing new to export; all visits already in DB")
            return

        # ── 3. Export new cards from DB to Excel ──────────────────────────────
        new_guids = {
            str((result.input.get("Прием") or {}).get("GUID") or "").lower()
            for result, _ in pairs
        }
        new_guids.discard("")

        if new_guids:
            async with ExcelFormatter(EXCEL_PATH) as fmt:
                written = await fmt.export_guids(new_guids)
            log.info("📊 Exported %d row(s) to %s", written, EXCEL_PATH)
        else:
            log.info("📊 No guid-bearing cards to export; skipping Excel write")

        log.info("Audit complete. Log: %s", LOG_FILE)
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
