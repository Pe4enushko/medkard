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
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import openpyxl

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

GUID_RE = re.compile(
    r"\bGUID:\s*([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b"
)

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


def _extract_guid_from_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = GUID_RE.search(value)
    return match.group(1).lower() if match else None


def _load_done_guids_from_excel(path: Path) -> set[str]:
    """Read GUID values from column A of an existing audit workbook."""
    if not path.exists():
        log.info("No existing Excel output found at %s", path)
        return set()

    done: set[str] = set()
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb.active
        for row_idx, (column_a,) in enumerate(
            ws.iter_rows(min_row=2, min_col=1, max_col=1, values_only=True),
            start=2,
        ):
            guid = _extract_guid_from_text(column_a)
            if guid:
                done.add(guid)
            elif column_a:
                log.debug("No GUID found in Excel row %d column A", row_idx)
    finally:
        wb.close()

    log.info("Loaded %d already audited GUID(s) from %s", len(done), path)
    return done


async def main() -> None:
    log.info("🩺 Starting period audit: datebegin=%s dateend=%s", DATEBEGIN, DATEEND)

    # ── 1. Load raw JSON from cache or fetch it from 1C ───────────────────────
    payload = _load_or_fetch_one_c_payload(datebegin=DATEBEGIN, dateend=DATEEND)
    done_guids = _load_done_guids_from_excel(EXCEL_PATH)

    # ── 2. Run full pipeline with raw payload ─────────────────────────────────
    pipeline = AuditPipeline(excel_path=EXCEL_PATH)
    results = await pipeline.run(payload, done_guids=done_guids)
    log.info("Pipeline done: %d result(s)", len(results))

    if not results:
        log.info("Nothing to persist; all visits may already be present in %s", EXCEL_PATH)
        return

    # ── 3. Persist results to DB ──────────────────────────────────────────────
    async with ResultsStorage() as storage:
        for idx, result in enumerate(results, start=1):
            log.info("💾 Persisting result %d/%d", idx, len(results))
            result_id = await storage.insert(result)
            log.info("💾 Persisted result %d/%d id=%s", idx, len(results), result_id)

    log.info("Audit complete. Log: %s", LOG_FILE)


if __name__ == "__main__":
    asyncio.run(main())
