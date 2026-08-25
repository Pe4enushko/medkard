#!/usr/bin/env python3
"""
Stamp a doctor onto already-audited cards, so a demo has a doctor to show.

WHY THIS EXISTS: the personal-report mailing and the "talk about this doctor's
cards" flow both hang on two keys inside card_data — `Прием.Врач_код` (the
filter in the pull API and the key of the engine's doctor_user_map) and
`Прием.Врач` (the name /visits/doctors lists). 1C does not send either yet, so
on real data both are empty and neither feature has anything to demonstrate.

This script fills those two keys on cards that are already audited. Nothing is
generated: the patients, the visit and every audit finding stay exactly as the
pipeline produced them — only the doctor is ours. Cards that already carry a
doctor code are never touched, so a second run for a different doctor picks up
the next batch, and `--revert` removes what this script wrote.

Run from project root (dry-run unless -y):

    python scripts/seed-demo-doctor.py MDS --date 2026-08-20 \
        --code 90001 --name "Панкратов Эдуард Рашитович" [--limit 10] [-y]
    python scripts/seed-demo-doctor.py MDS --date 2026-08-20 --code 90001 --revert -y

Options:
    ORG        Organization as named in the organizations table: Alenka or MDS
    --date     Visit date (YYYY-MM-DD) whose cards to stamp
    --code     Doctor code to write into Прием.Врач_код
    --name     Doctor's full name for Прием.Врач (not needed with --revert)
    --limit    How many cards this doctor should end up with (default: 10)
    --revert   Remove this doctor's code and name from that date's cards
    -y         Actually write. Without it the script only reports what it would do.

AFTERWARDS, on the engine host:
    scripts/medcheck/medcheck_map_doctors.py <slug> "<ФИО>"   # doctor_code -> Iskra user
    scripts/medcheck/medcheck_replica_pull.py --slug <slug>   # cards into the analyst replica
    scripts/medcheck/medcheck_personal_pull.py --slug <slug> --date <дата> \
        --doctor <код> --apply                                # the personal report itself
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from storage.done_cards_storage import DoneCardsStorage
from storage.organizations_storage import OrganizationsStorage

ORGS = ("Alenka", "MDS")

# Stricter than the pull API's own [\w-]{1,64}: Python's \w matches Cyrillic, so
# that pattern would accept a code no operator can type into a URL by hand. This
# code is one we mint ourselves, so there is no reason to leave the door open.
_CODE_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

log = logging.getLogger(__name__)


def _validate_code(value: str) -> str:
    if not _CODE_RE.match(value):
        raise SystemExit(
            f"--code must match [A-Za-z0-9_-]{{1,64}}, got {value!r} — a code outside "
            "that set makes /visits/pull answer 422 for this doctor forever"
        )
    return value


def _parse_date(value: str, option: str = "--date") -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise SystemExit(f"{option} must be YYYY-MM-DD, got {value!r}")


def _findings(row: dict[str, Any]) -> int:
    return row["formal_n"] + row["diag_n"] + row["icd_n"]


@dataclass
class Plan:
    mine: list[dict] = field(default_factory=list)      # already this doctor's
    to_stamp: list[dict] = field(default_factory=list)  # will become this doctor's
    foreign: list[dict] = field(default_factory=list)   # some other doctor's — never touched
    free: list[dict] = field(default_factory=list)      # unstamped, ranked, incl. to_stamp


def _plan(rows: list[dict], *, code: str, limit: int) -> Plan:
    """Split a date's cards into ours / free / someone else's, and pick the batch.

    Free cards are ranked by how much the audit found on them: a personal report
    whose cards are all clean demonstrates nothing. Ties break by guid so a
    repeat run with the same arguments picks the same cards.

    Only free cards are ever picked, and the batch is a top-up to *limit* rather
    than a fixed count — that is what makes the script idempotent (running it
    twice does not give the doctor twenty cards) and what lets a second doctor be
    seeded on the same date without stealing the first one's.
    """
    plan = Plan()
    for row in rows:
        if row["doctor_code"] == code:
            plan.mine.append(row)
        elif row["doctor_code"]:            # None or "" — 1C sends both for "no doctor"
            plan.foreign.append(row)
        else:
            plan.free.append(row)
    plan.free.sort(key=lambda r: (-_findings(r), r["card_guid"]))
    plan.to_stamp = plan.free[:max(0, limit - len(plan.mine))]
    return plan


def _stamp(priem: dict | None, *, name: str, code: str) -> dict | None:
    """The Прием block with our doctor in it, or None if there is no block."""
    if not priem:
        return None
    return {**priem, "Врач": name, "Врач_код": code}


def _unstamp(priem: dict | None) -> dict | None:
    """The Прием block without the two keys we wrote, or None if neither is there."""
    if not priem or not ({"Врач", "Врач_код"} & set(priem)):
        return None
    return {k: v for k, v in priem.items() if k not in ("Врач", "Врач_код")}


@dataclass
class Summary:
    stamped: int = 0
    reverted: int = 0
    skipped: int = 0


async def _run(storage, *, org_id: str, visit_date: date, code: str, name: str,
               limit: int, apply: bool, revert: bool) -> Summary:
    """Report the plan, and carry it out when *apply* is set."""
    rows = await storage.list_audited_by_visit_date(
        organization_id=org_id, visit_date=visit_date)
    plan = _plan(rows, code=code, limit=limit)
    summary = Summary()

    print(f"\nКарт за {visit_date.isoformat()}: {len(rows)} "
          f"(свободных {len(plan.free)}, у врача {code} — {len(plan.mine)}, "
          f"у других врачей {len(plan.foreign)})")

    targets = plan.mine if revert else plan.to_stamp
    verb = "снять" if revert else "проставить"
    if not targets:
        print(f"Нечего {verb}." + ("" if revert else
              f" Карт у врача {code}: {len(plan.mine)} при --limit {limit}."))
        return summary

    print(f"{'Снимаем' if revert else 'Проставляем'} {len(targets)} карт(ы):")
    for row in targets:
        print(f"  {row['card_guid']}  замечаний: "
              f"формальных {row['formal_n']}, по диагнозу {row['diag_n']}, "
              f"по МКБ {row['icd_n']}")

    if not apply:
        print("\nDRY-RUN: ничего не записано. Повторите с -y.")
        return summary

    for row in targets:
        guid = row["card_guid"]
        priem = await storage.get_priem(guid)
        fresh = _unstamp(priem) if revert else _stamp(priem, name=name, code=code)
        if fresh is None:
            # No Прием block at all, or nothing left to remove — either way this
            # card is not ours to rewrite. Reported, not fatal: one odd card must
            # not cost the whole batch.
            print(f"  ПРОПУСК {guid}: нечего {verb}")
            summary.skipped += 1
            continue
        await storage.replace_priem(
            card_guid=guid, priem=json.dumps(fresh, ensure_ascii=False))
        if revert:
            summary.reverted += 1
        else:
            summary.stamped += 1

    print(f"\nГотово: проставлено {summary.stamped}, снято {summary.reverted}, "
          f"пропущено {summary.skipped}")
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stamp a demo doctor onto already-audited cards (dry-run unless -y)")
    parser.add_argument("org", choices=ORGS, help="organization name")
    parser.add_argument("--date", required=True, help="visit date, YYYY-MM-DD")
    parser.add_argument("--code", required=True, help="doctor code for Прием.Врач_код")
    parser.add_argument("--name", help="doctor's full name for Прием.Врач")
    parser.add_argument("--limit", type=int, default=10,
                        help="how many cards this doctor should end up with (default: 10)")
    parser.add_argument("--revert", action="store_true",
                        help="remove this doctor's code and name from that date's cards")
    parser.add_argument("-y", dest="apply", action="store_true",
                        help="actually write; without it nothing is changed")
    args = parser.parse_args()

    code = _validate_code(args.code)
    visit_date = _parse_date(args.date)
    if not args.revert and not args.name:
        raise SystemExit("--name is required unless --revert is given")
    if args.limit < 1:
        raise SystemExit(f"--limit must be positive, got {args.limit}")

    async with OrganizationsStorage() as organizations:
        org_id = await organizations.get_id_by_name(args.org)

    print(f"Организация {args.org} ({org_id}), дата {visit_date.isoformat()}, "
          f"врач {code}" + (f" «{args.name}»" if args.name else ""))

    async with DoneCardsStorage() as storage:
        summary = await _run(storage, org_id=org_id, visit_date=visit_date, code=code,
                             name=args.name or "", limit=args.limit,
                             apply=args.apply, revert=args.revert)

    if args.apply and summary.stamped:
        print("\nДальше на движке:\n"
              f"  scripts/medcheck/medcheck_map_doctors.py <slug> \"{args.name}\"\n"
              "  scripts/medcheck/medcheck_replica_pull.py --slug <slug>\n"
              "  scripts/medcheck/medcheck_personal_pull.py --slug <slug> "
              f"--date {visit_date.isoformat()} --doctor {code} --apply")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main())
