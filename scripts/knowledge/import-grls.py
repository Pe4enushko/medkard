#!/usr/bin/env python3
"""Import a GRLS xlsx export (zip or directory) into grls_registry.

Usage:
    python scripts/knowledge/import-grls.py <archive.zip | dir-with-xlsx> [--dry-run] [--make-dump FILE]

--dry-run        parse, dedup, print counts; do not touch the DB
--make-dump FILE write JSONL(.gz) dump for engine sync (spec §7); works with --dry-run
Full replacement of grls_registry in one transaction; idempotent.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from grls.dump import write_dump  # noqa: E402
from grls.parser import SheetResult, read_archive  # noqa: E402
from storage.models.grls_record import GrlsImport, GrlsRecord  # noqa: E402

logger = logging.getLogger("import-grls")


@dataclass
class ImportPlan:
    registry_date: date
    records: list[GrlsRecord]
    status_counts: dict[str, int]
    skipped_files: list[str] = field(default_factory=list)
    duplicates_dropped: int = 0


def plan_import(results: list[SheetResult]) -> ImportPlan:
    """Merge sheets, drop exact duplicates by row_hash, count per status."""
    if not results:
        raise SystemExit("no xlsx sheets found")
    dates = {r.registry_date for r in results}
    if len(dates) > 1:
        raise SystemExit(f"sheets carry different registry dates: {sorted(d.isoformat() for d in dates)}")
    seen: set[str] = set()
    records: list[GrlsRecord] = []
    dropped = 0
    skipped: list[str] = []
    for res in results:
        if res.skipped:
            skipped.append(res.source_name)
            continue
        for rec in res.records:
            if rec.row_hash in seen:
                dropped += 1
                continue
            seen.add(rec.row_hash)
            records.append(rec)
    counts = Counter(r.status for r in records)
    return ImportPlan(registry_date=dates.pop(), records=records, status_counts=dict(counts),
                      skipped_files=skipped, duplicates_dropped=dropped)


def _print_summary(plan: ImportPlan, archive_name: str, dry_run: bool) -> None:
    print(f"archive: {archive_name}")
    print(f"registry_date: {plan.registry_date.isoformat()}")
    for status, n in sorted(plan.status_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {status}: {n}")
    print(f"rows: {len(plan.records)} (exact duplicates dropped: {plan.duplicates_dropped})")
    print(f"skipped files: {plan.skipped_files}")
    if dry_run:
        print("dry-run: database not touched")


async def _write(plan: ImportPlan, archive_name: str) -> int:
    from storage.grls_storage import GrlsStorage  # DB deps only when writing

    imp = GrlsImport(archive_name=archive_name, registry_date=plan.registry_date,
                     status_counts=plan.status_counts, skipped_files=plan.skipped_files)
    async with GrlsStorage() as storage:
        return await storage.replace_all(plan.records, imp)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="zip archive or directory with xlsx")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--make-dump", type=Path, metavar="FILE")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    plan = plan_import(read_archive(args.source))
    archive_name = args.source.name
    _print_summary(plan, archive_name, args.dry_run)
    if args.make_dump:
        n = write_dump(args.make_dump, plan.records, registry_date=plan.registry_date, archive_name=archive_name)
        print(f"dump: {args.make_dump} ({n} rows)")
    if args.dry_run:
        return 0
    inserted = asyncio.run(_write(plan, archive_name))
    # registry row count after ON CONFLICT(row_hash) dedup, not an "inserted" count —
    # cross-sheet duplicates in the real export make this <= len(plan.records).
    print(f"registry rows after import: {inserted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
