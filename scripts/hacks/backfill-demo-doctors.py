#!/usr/bin/env python3
"""
Give every visit of one clinic a made-up doctor from resources/demo_doctors.json.

WHY THIS EXISTS: the personal-report mailing and the "talk about this doctor's
cards" flow both hang on `Прием.Врач_код` and `Прием.Врач`, and 1C sends
neither for Alenka. api/demo_doctors.py stamps a random doctor onto each card
AS IT ARRIVES; this script does the same to the cards that arrived before the
crutch existed, and takes the stamp back off.

THIS IS A CRUTCH AND IS BUILT TO BE REMOVED, together with the module and the
three lines it occupies in api/routes/visits.py.

Nothing is generated: patients, visits and audit results stay exactly as the
pipeline produced them — only the doctor is ours. A card that already carries
a doctor is never touched, so the day 1C starts sending them the backfill has
nothing left to do; --revert removes only the codes listed in the file, so a
real doctor from 1C survives it.

Run from project root (dry-run unless -y):

    python scripts/hacks/backfill-demo-doctors.py Alenka [--limit 100] [-y]
    python scripts/hacks/backfill-demo-doctors.py Alenka --revert [-y]

Options:
    ORG        Organization as named in the organizations table: Alenka or MDS
    --limit    Stamp (or revert) at most this many cards; 0 — all of them
    --revert   Remove the made-up doctors instead of writing them
    -y         Actually write. Without it the script only reports what it would do.

The replica of the engine picks the change up by itself: done_cards_set_updated_at
(migration 022) fires on UPDATE, so a stamped card comes back in the incremental
sync just like a freshly audited one.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from api.demo_doctors import doctors_path, load_doctors
from storage.done_cards_storage import DoneCardsStorage
from storage.organizations_storage import OrganizationsStorage

ORGS = ("Alenka", "MDS")


def assign(card_guids: list[str], doctors: list[dict]) -> dict[str, list[str]]:
    """Split the cards between the doctors at random: {code: [guid, ...]}.

    Grouped by doctor rather than card by card, so the whole clinic is written
    in one statement per doctor instead of one per card.
    """
    if card_guids and not doctors:
        raise SystemExit(
            f"в файле {doctors_path()} нет ни одного врача с code и name — "
            "штамповать нечем")
    batches: dict[str, list[str]] = defaultdict(list)
    for guid in card_guids:
        batches[random.choice(doctors)["code"]].append(guid)
    return dict(batches)


async def _stamp(storage, *, org_id: str, doctors: list[dict], limit: int,
                 apply: bool) -> int:
    guids = await storage.list_cards_without_doctor(
        organization_id=org_id, limit=limit)
    print(f"Карт без врача: {len(guids)}")
    if not guids:
        return 0

    by_name = {d["code"]: d["name"] for d in doctors}
    batches = assign(guids, doctors)
    for code, batch in sorted(batches.items()):
        print(f"  {code} {by_name[code]}: {len(batch)}")

    if not apply:
        print("\nDRY-RUN: ничего не записано. Повторите с -y.")
        return 0

    written = 0
    for code, batch in sorted(batches.items()):
        written += await storage.set_doctor_on_cards(
            card_guids=batch, code=code, name=by_name[code])
    return written


async def _revert(storage, *, org_id: str, doctors: list[dict], limit: int,
                  apply: bool) -> int:
    codes = [d["code"] for d in doctors]
    guids = await storage.list_cards_with_doctor_codes(
        organization_id=org_id, codes=codes, limit=limit)
    print(f"Карт с демо-врачами {', '.join(codes)}: {len(guids)}")
    if not guids:
        return 0
    if not apply:
        print("\nDRY-RUN: ничего не записано. Повторите с -y.")
        return 0
    return await storage.clear_doctor_on_cards(card_guids=guids)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stamp made-up doctors onto a clinic's cards (dry-run unless -y)")
    parser.add_argument("org", choices=ORGS, help="organization name")
    parser.add_argument("--limit", type=int, default=0,
                        help="at most this many cards; 0 — all of them")
    parser.add_argument("--revert", action="store_true",
                        help="remove the made-up doctors instead of writing them")
    parser.add_argument("-y", dest="apply", action="store_true",
                        help="actually write; without it nothing is changed")
    args = parser.parse_args()

    if args.limit < 0:
        raise SystemExit(f"--limit must not be negative, got {args.limit}")

    doctors = load_doctors()
    if not doctors:
        raise SystemExit(f"в файле {doctors_path()} нет ни одного врача с code и name")

    async with OrganizationsStorage() as organizations:
        org_id = await organizations.get_id_by_name(args.org)

    print(f"Организация {args.org} ({org_id}), врачей в файле: {len(doctors)}, "
          f"режим: {'снятие' if args.revert else 'штамп'}")

    async with DoneCardsStorage() as storage:
        run = _revert if args.revert else _stamp
        written = await run(storage, org_id=org_id, doctors=doctors,
                            limit=args.limit, apply=args.apply)

    if args.apply:
        print(f"\nГотово: карт изменено {written}")
        if not args.revert and written:
            print("Дальше на движке:\n"
                  "  scripts/medcheck/medcheck_map_doctors.py <slug> \"<ФИО>\"\n"
                  "  scripts/medcheck/medcheck_replica_pull.py --slug <slug>")


if __name__ == "__main__":
    asyncio.run(main())
