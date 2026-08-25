"""seed-guidelines.py — залить resources/manifest.csv в таблицу guidelines.

Запускать после миграции 019 и ДО FK-миграции 021 (см. spec §4).
Идемпотентно: upsert по file_id.

    python scripts/knowledge/seed-guidelines.py
"""
from __future__ import annotations

import asyncio
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from storage.guidelines_storage import GuidelinesStorage
from storage.models.guideline import Guideline

_MANIFEST = ROOT / "resources" / "manifest.csv"


async def main() -> None:
    with open(_MANIFEST, newline="", encoding="utf-8") as fh:
        rows = [Guideline.from_manifest_row(r) for r in csv.DictReader(fh) if (r.get("ID") or "").strip()]
    async with GuidelinesStorage() as storage:
        written = await storage.upsert_many(rows)
    print(f"seeded {written} guideline(s) from {_MANIFEST.name}")


if __name__ == "__main__":
    asyncio.run(main())
