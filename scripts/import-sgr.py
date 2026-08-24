#!/usr/bin/env python3
"""Import an SGR (EAEU dietary supplements) CSV export into dietary_supplements.

Usage:
    python scripts/import-sgr.py <export.csv> [--dry-run] [--make-dump FILE]

--dry-run        parse, dedup, print counts; do not touch the DB
--make-dump FILE write JSONL(.gz) dump for engine sync; works with --dry-run
Full replacement of dietary_supplements in one transaction; idempotent.

Replaces the supplements half of scripts/seed-reference-lists.sh: that one pins
the staging width in SQL (col01 … col39) and the encoding to UTF-8, so it breaks
on the 2026-08-24 export (43 columns, Windows-1251). Here the columns are taken
by name from the header, and the encoding is detected.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from sgr.dump import write_dump  # noqa: E402
from sgr.parser import ParseResult, SgrFormatError, read_export  # noqa: E402

logger = logging.getLogger("import-sgr")


def _print_summary(result: ParseResult, source: str, dry_run: bool) -> None:
    print(f"export: {source}")
    print(f"encoding: {result.encoding}; columns in header: {result.columns}")
    counts = Counter(r.status or "—" for r in result.rows)
    for status, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {status}: {n}")
    print(f"rows: {len(result.rows)} "
          f"(exact duplicates dropped: {result.duplicates_dropped}, "
          f"skipped without a product name: {result.skipped_no_name})")
    if dry_run:
        print("dry-run: database not touched")


async def _write(result: ParseResult) -> int:
    # Зависимости БД тянем только когда пишем: --dry-run и --make-dump обязаны
    # работать там, где до Postgres не дотянуться.
    from storage.dietary_supplements_storage import DietarySupplementsStorage
    from storage.models.dietary_supplement import DietarySupplement

    records = [
        DietarySupplement(
            product_name=r.product_name,
            registration_number=r.registration_number,
            status=r.status,
            manufacturer_name=r.manufacturer_name,
            country_of_manufacture=r.country_of_manufacture,
            scope_of_application=r.scope_of_application,
            label_info=r.label_info,
            registered_at=r.registered_at.isoformat() if r.registered_at else None,
        )
        for r in result.rows
    ]
    async with DietarySupplementsStorage() as storage:
        return await storage.replace_all(records)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="CSV export of the SGR registry")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--make-dump", type=Path, metavar="FILE")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        result = read_export(args.source)
    except (SgrFormatError, OSError) as e:
        # Отказы разбора — сообщение человеку, а не трейсбек: каждый означает
        # «файл не тот», и продолжать нечем.
        raise SystemExit(str(e))

    _print_summary(result, args.source.name, args.dry_run)
    if args.make_dump:
        n = write_dump(args.make_dump, result.rows)
        print(f"dump: {args.make_dump} ({n} rows)")
    if args.dry_run:
        return 0
    rows = asyncio.run(_write(result))
    print(f"registry rows after import: {rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
