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
from typing import Any

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


async def main() -> None:
    log.info("🩺 Starting period audit: datebegin=%s dateend=%s", DATEBEGIN, DATEEND)

    # ── 1. Load raw JSON from cache or fetch it from 1C ───────────────────────
    payload = _load_or_fetch_one_c_payload(datebegin=DATEBEGIN, dateend=DATEEND)

    # ── 2. Run full pipeline with raw payload ─────────────────────────────────
    pipeline = AuditPipeline(excel_path=EXCEL_PATH)
    results = await pipeline.run(payload)
    log.info("Pipeline done: %d result(s)", len(results))

    # ── 3. Persist results to DB ──────────────────────────────────────────────
    async with ResultsStorage() as storage:
        for idx, result in enumerate(results, start=1):
            log.info("💾 Persisting result %d/%d", idx, len(results))
            result_id = await storage.insert(result)
            log.info("💾 Persisted result %d/%d id=%s", idx, len(results), result_id)

    log.info("Audit complete. Log: %s", LOG_FILE)


if __name__ == "__main__":
    asyncio.run(main())
