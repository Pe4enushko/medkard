#!/usr/bin/env python3
"""
Backfill the "Прием" block in done_cards.card_data from 1C.

1C now returns more fields inside "Прием" than older stored cards carry.
This script walks the period day by day — one sequential 1C request per
date — matches every fetched visit to its done_cards row by Прием.GUID
and merges the fresh "Прием" values into the stored card_data (fresh
keys overwrite stored ones, keys missing from the fresh block are kept).

Run from project root:
    python scripts/backfill-priem.py ORG --since DD.MM.YYYY [--until DD.MM.YYYY] [--dry-run] [-y]

Options:
    ORG        1C organization: Alenka or MDS
    --since    First date of the period (DD.MM.YYYY), inclusive
    --until    Last date of the period (DD.MM.YYYY), inclusive (default: today)
    --dry-run  Fetch and match only — report what would change, write nothing
    -y         Skip confirmation prompt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from integrations.one_c import AlenkaOneCClient, MdsOneCClient, OneCClient
from parsers.json_parser import AppointmentParser
from storage.done_cards_storage import DoneCardsStorage

ONE_C_CLIENTS: dict[str, type[OneCClient]] = {
    "Alenka": AlenkaOneCClient,
    "MDS": MdsOneCClient,
}

DATE_FMT = "%d.%m.%Y"
LOGS_DIR = ROOT / "logs"

log = logging.getLogger(__name__)


def _parse_date(value: str, option: str) -> datetime:
    try:
        return datetime.strptime(value, DATE_FMT)
    except ValueError:
        raise SystemExit(f"{option} must be DD.MM.YYYY, got {value!r}")


def _date_range(since: datetime, until: datetime) -> Iterator[str]:
    """Yield every date from *since* to *until* inclusive as DD.MM.YYYY."""
    day = since
    while day <= until:
        yield day.strftime(DATE_FMT)
        day += timedelta(days=1)


def _visit_priem(visit: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    priem = visit.get("Прием") or {}
    guid = priem.get("GUID")
    return (str(guid) if guid else None), priem


def _setup_logging() -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    log_file = LOGS_DIR / f"backfill-priem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return log_file


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("org", choices=tuple(ONE_C_CLIENTS), help="1C organization")
    parser.add_argument("--since", required=True, metavar="DD.MM.YYYY", help="First date of the period, inclusive")
    parser.add_argument("--until", default=None, metavar="DD.MM.YYYY", help="Last date of the period, inclusive (default: today)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and match only — write nothing")
    parser.add_argument("-y", action="store_true", help="Skip confirmation prompt")
    return parser.parse_args(argv)


def _changed_keys(stored: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    return sorted(k for k, v in fresh.items() if stored.get(k) != v)


async def _process_day(
    day: str,
    client: OneCClient,
    storage: DoneCardsStorage,
    dry_run: bool,
    totals: dict[str, int],
) -> None:
    payload = client.fetch_json_for_period(datebegin=day, dateend=day)
    visits = AppointmentParser.split(payload)
    day_updated = day_missing = 0

    for visit in visits:
        guid, priem = _visit_priem(visit)
        totals["visits"] += 1
        if not guid:
            totals["no_guid"] += 1
            log.warning("📅 %s: visit without Прием.GUID skipped", day)
            continue

        if dry_run:
            stored = await storage.get_priem(guid)
            if stored is None:
                day_missing += 1
            else:
                day_updated += 1
                changed = _changed_keys(stored, priem)
                if changed:
                    log.info("📅 %s: would update guid=%s keys=%s", day, guid, ", ".join(changed))
        elif await storage.merge_priem(card_guid=guid, priem=json.dumps(priem, ensure_ascii=False)):
            day_updated += 1
        else:
            day_missing += 1

    totals["updated"] += day_updated
    totals["not_found"] += day_missing
    log.info(
        "📅 %s: %d visit(s), %d %s, %d without a done_cards row",
        day, len(visits), day_updated, "would update" if dry_run else "updated", day_missing,
    )


async def main() -> None:
    args = _parse_args()
    since = _parse_date(args.since, "--since")
    until = _parse_date(args.until, "--until") if args.until else datetime.now()
    if since > until:
        raise SystemExit(f"--since {args.since} is after --until {until.strftime(DATE_FMT)}")

    log_file = _setup_logging()
    days = list(_date_range(since, until))

    print(f"Organization: {args.org}")
    print(f"Period: {days[0]} — {days[-1]} ({len(days)} day(s), one 1C request per day)")
    if args.dry_run:
        print("Mode: dry-run (no DB writes)")
    if not args.y:
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    client = ONE_C_CLIENTS[args.org].from_env()
    totals = {"visits": 0, "updated": 0, "not_found": 0, "no_guid": 0}
    failed_days: list[str] = []

    async with DoneCardsStorage() as storage:
        for day in days:
            try:
                await _process_day(day, client, storage, args.dry_run, totals)
            except RuntimeError as exc:
                failed_days.append(day)
                log.error("📅 %s: 1C fetch failed, skipping day: %s", day, exc)

    log.info(
        "Backfill %s: %d visit(s) over %d day(s) — %d %s, %d without a done_cards row, %d without GUID",
        "dry-run complete" if args.dry_run else "complete",
        totals["visits"], len(days),
        totals["updated"], "would update" if args.dry_run else "updated",
        totals["not_found"], totals["no_guid"],
    )
    if failed_days:
        log.error("1C fetch failed for %d day(s): %s", len(failed_days), ", ".join(failed_days))
    log.info("Log: %s", log_file)
    if failed_days:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
