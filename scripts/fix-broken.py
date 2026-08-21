#!/usr/bin/env python3
"""Re-audit replayable ``broken = TRUE`` cards from their stored DB payload.

The script is offline with respect to 1C. It deliberately passes an empty
``done_guids`` set to the audit pipeline, because broken rows have terminal
``status = 'done'`` and the normal nightly dedup otherwise freezes them.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.broken_replay import (
    BrokenGroup,
    diff_outcomes,
    format_summary,
    group_by_org,
)
from audit.filters import CardFilter
from audit.pipeline import AuditPipeline
from parsers.filter_config import load_card_filter
from RAG.retrieval.vector_store import close_pool
from storage.done_cards_storage import DoneCardsStorage
from storage.organizations_storage import OrganizationsStorage

LOGS_DIR = ROOT / "logs"


def build_parser() -> argparse.ArgumentParser:
    """Build a parser requiring exactly one of organization or ``--all``."""
    parser = argparse.ArgumentParser(
        description="Re-audit broken cards from stored done_cards payloads."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "org",
        nargs="?",
        choices=("Alenka", "MDS"),
        help="organization",
    )
    target.add_argument(
        "--all",
        action="store_true",
        help="re-audit every organization's replayable broken cards",
    )
    parser.add_argument("-y", action="store_true", help="skip confirmation")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the replay set without writing",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        metavar="N",
        help="maximum concurrent cards per organization (default: 5)",
    )
    return parser


def _configure_logging() -> tuple[logging.Logger, Path]:
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"fix-broken_{timestamp}.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__), log_file


def _group_label(group: BrokenGroup) -> str:
    return group.org_name or group.org_id or "<no organization>"


def _filter_for(group: BrokenGroup) -> CardFilter:
    """Use the owning organization's filter; NULL/unknown orgs have none."""
    return load_card_filter(group.org_name) if group.org_name else CardFilter([])


def _show_plan(groups: list[BrokenGroup], args: argparse.Namespace) -> None:
    print(f"Mode: {'all organizations' if args.all else args.org}")
    print(f"Broken cards to re-audit: {sum(len(group.guids) for group in groups)}")
    for group in groups:
        print(f"  {_group_label(group)}: {len(group.guids)}")
        print(f"  Filters:\n{_filter_for(group)}")


def _confirmed(groups: list[BrokenGroup], args: argparse.Namespace) -> bool:
    _show_plan(groups, args)
    if args.dry_run:
        print("Dry run — nothing was written.")
        return False
    if args.y:
        return True
    if input("Proceed? [y/N] ").strip().lower() == "y":
        return True
    print("Aborted.")
    return False


async def _load_groups(
    args: argparse.Namespace,
    log: logging.Logger,
) -> list[BrokenGroup]:
    async with OrganizationsStorage() as organizations:
        org_id = await organizations.get_id_by_name(args.org) if args.org else None
        async with DoneCardsStorage() as done_cards:
            rows = await done_cards.get_broken(organization_id=org_id)

        names: dict[str, str] = {}
        for row_org_id in {
            row["organization_id"] for row in rows if row["organization_id"] is not None
        }:
            try:
                names[row_org_id] = await organizations.get_name_by_id(row_org_id)
            except ValueError:
                log.warning(
                    "Organization %s no longer exists; using an empty filter",
                    row_org_id,
                )

    return group_by_org(rows, names)


async def _replay(
    group: BrokenGroup,
    num_batches: int,
    log: logging.Logger,
) -> None:
    label = _group_label(group)
    log.info("🔧 Replaying %d card(s) for org=%s", len(group.visits), label)
    async with AuditPipeline(
        org_id=group.org_id,
        card_filter=_filter_for(group),
    ) as pipeline:
        pairs = await pipeline.run_batched(
            group.visits,
            num_batches=num_batches,
            done_guids=set(),
        )
    log.info("🔧 org=%s produced %d successful result(s)", label, len(pairs))


async def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_batches < 1:
        build_parser().error("--num-batches must be at least 1")
    log, log_file = _configure_logging()

    try:
        groups = await _load_groups(args, log)
        if not groups:
            print("No broken cards to re-audit.")
            return 0
        if not _confirmed(groups, args):
            return 0

        before = {guid for group in groups for guid in group.guids}
        for group in groups:
            await _replay(group, args.num_batches, log)

        async with DoneCardsStorage() as done_cards:
            states = await done_cards.get_states_for_guids(before)

        after_broken = {guid for guid, state in states.items() if state["broken"]}
        after_ignored = {guid for guid, state in states.items() if state["ignored"]}
        stacktraces = {
            guid: state["stacktrace"]
            for guid, state in states.items()
            if state["stacktrace"]
        }
        summary = format_summary(
            diff_outcomes(before, after_broken, after_ignored),
            stacktraces,
        )
        print(summary)
        log.info(summary)
        log.info("Done. Log: %s", log_file)
        return 0
    finally:
        await close_pool()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
