#!/usr/bin/env python3
"""
Load appointments from a JSON file in the project root, run the full audit
pipeline, then export results to Excel.

Run from project root:
    python scripts/audit-file.py [--file PATH] [--excel PATH] [--num-batches N]

Options:
    --file         Input JSON file (default: data.json)
    --excel        Output xlsx file (default: audit_results.xlsx)
    --num-batches  Max concurrent visits processed at a time (default: 5)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.excel_formatter import ExcelFormatter
from audit.pipeline import AuditPipeline
from RAG.retrieval.vector_store import close_pool

# ── Args ──────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--file", default=str(ROOT / "data.json"), metavar="PATH", help="Input JSON file (default: data.json)")
_parser.add_argument("--excel", default=str(ROOT / "audit_results.xlsx"), metavar="PATH", help="Output xlsx file (default: audit_results.xlsx)")
_parser.add_argument("--num-batches", type=int, default=5, metavar="N", help="Max concurrent visits processed at a time (default: 5)")
_args = _parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE  = Path(_args.file)
EXCEL_PATH = Path(_args.excel)
LOGS_DIR   = ROOT / "logs"

# ── Logging ───────────────────────────────────────────────────────────────────
LOGS_DIR.mkdir(exist_ok=True)

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"audit-file_{_ts}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


async def main() -> None:
    if not DATA_FILE.exists():
        log.error("Data file not found: %s", DATA_FILE)
        sys.exit(1)

    log.info("Loading appointments from %s", DATA_FILE)
    raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    # MDSdata format: [{"appointments": [...]}] — unwrap the outer list
    payload = raw[0] if isinstance(raw, list) else raw

    # ── 1. Run pipeline — each card is persisted to DB on completion ──────────
    async with AuditPipeline() as pipeline:
        pairs = await pipeline.run_batched(payload, num_batches=_args.num_batches)
    log.info("Pipeline done: %d result(s)", len(pairs))

    if not pairs:
        log.info("Nothing new to export; all visits already in DB")
        return

    # ── 2. Export new cards from DB to Excel ──────────────────────────────────
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


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        asyncio.run(close_pool())
