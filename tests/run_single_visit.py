"""
Run _audit_visit on a single visit parsed from a JSON file.

Change APPOINTMENTS_JSON_PATH to point at the JSON file containing an
appointments payload. The first appointment in the array will be audited.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from audit.pipeline import AuditPipeline
from parsers.json_parser import AppointmentsParser

APPOINTMENTS_JSON_PATH = r"visit.json"
LOG_DIR = Path("logs")

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"run_{timestamp}.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logger.info("Logging to %s", log_file)


async def main() -> None:
    _setup_logging()

    path = Path(APPOINTMENTS_JSON_PATH)
    visits = AppointmentsParser.split_file(path)
    visit = visits[0]

    logger.info("Input visit to be processed:\n%s", json.dumps(visit, ensure_ascii=False, indent=2))

    pipeline = AuditPipeline(excel_path="audit_results.xlsx")
    results = await pipeline._audit_visit(visit)

    for i, result in enumerate(results, 1):
        logger.info("--- Result %d ---\n%s", i, result.pretty_format())


if __name__ == "__main__":
    asyncio.run(main())
