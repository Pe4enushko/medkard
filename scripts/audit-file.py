#!/usr/bin/env python3
"""
Load appointments from a JSON file in the project root, run the full audit
pipeline, then persist every result to DB and Excel.

Run from project root:
    python scripts/audit-file.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audit.pipeline import AuditPipeline
from storage import ResultsStorage

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE  = ROOT / "data.json"

EXCEL_PATH = ROOT / "audit_results.xlsx"
LOGS_DIR   = ROOT / "logs"

GUID_RE = re.compile(
    r"\bGUID:\s*([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b"
)

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


def _extract_guid_from_text(value: object) -> str | None:
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
    if not DATA_FILE.exists():
        log.error("Data file not found: %s", DATA_FILE)
        sys.exit(1)

    log.info("Loading appointments from %s", DATA_FILE)
    raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    # MDSdata format: [{"appointments": [...]}] — unwrap the outer list
    payload = raw[0] if isinstance(raw, list) else raw
    done_guids = _load_done_guids_from_excel(EXCEL_PATH)

    # ── Run full pipeline ─────────────────────────────────────────────────────
    pipeline = AuditPipeline(excel_path=EXCEL_PATH)
    results = await pipeline.run(payload, done_guids=done_guids)
    log.info("Pipeline done: %d result(s)", len(results))

    if not results:
        log.info("Nothing to persist; all visits may already be present in %s", EXCEL_PATH)
        return

    # ── Persist results to DB ─────────────────────────────────────────────────
    async with ResultsStorage() as storage:
        for idx, result in enumerate(results, start=1):
            log.info("Persisting result %d/%d", idx, len(results))
            result_id = await storage.insert(result)
            log.info("Persisted result %d/%d id=%s", idx, len(results), result_id)

    log.info("Audit complete. Log: %s", LOG_FILE)


if __name__ == "__main__":
    asyncio.run(main())
