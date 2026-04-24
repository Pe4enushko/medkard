#!/usr/bin/env python3
"""
Fetch appointments from 1C for a configured period, save the raw JSON
snapshot, run the full audit pipeline, then persist every result to DB
and Excel.

Run from project root:
    python scripts/audit-one-c-period.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audit.pipeline import AuditPipeline
from integrations.one_c import OneCClient
from storage import ResultsStorage

# ── Config ────────────────────────────────────────────────────────────────────
DATEBEGIN = "17.04.2026"
DATEEND   = "24.04.2026"

EXCEL_PATH         = ROOT / "audit_results.xlsx"
DATA_SNAPSHOTS_DIR = ROOT / "data_snapshots"
LOGS_DIR           = ROOT / "logs"

# ── Logging ───────────────────────────────────────────────────────────────────
DATA_SNAPSHOTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"audit-one-c-period_{_ts}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


async def main() -> None:
    log.info("Starting period audit: datebegin=%s dateend=%s", DATEBEGIN, DATEEND)

    # ── 1. Fetch raw JSON from 1C ─────────────────────────────────────────────
    client = OneCClient.from_env()
    payload = client.fetch_json_for_period(datebegin=DATEBEGIN, dateend=DATEEND)

    snapshot_path = DATA_SNAPSHOTS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Snapshot saved: %s", snapshot_path)

    # ── 2. Run full pipeline with raw payload ─────────────────────────────────
    pipeline = AuditPipeline(excel_path=EXCEL_PATH)
    results = await pipeline.run(payload)
    log.info("Pipeline done: %d result(s)", len(results))

    # ── 3. Persist results to DB ──────────────────────────────────────────────
    async with ResultsStorage() as storage:
        for idx, result in enumerate(results, start=1):
            result_id = await storage.insert(result)
            log.info("Persisted result %d/%d id=%s", idx, len(results), result_id)

    log.info("Audit complete. Log: %s", LOG_FILE)


if __name__ == "__main__":
    asyncio.run(main())
