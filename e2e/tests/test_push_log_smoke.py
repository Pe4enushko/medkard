#!/usr/bin/env python3
"""
End-to-end smoke test for push_log / push_metrics_by_date (migration 027).

Exercises the trigger that logs every POST /visits/push as a push_log row,
tagged overrode_audit=true when it destroys a completed audit result and
false otherwise, and the push_metrics_by_date view that aggregates those
rows per organization per day.

Scope is push_log/push_metrics_by_date only. General /visits/push behaviour
(auth rejection, 422s, card_data replacement) is already covered by
scripts/smoke-cards-push.sh and is not re-asserted here.

Creates a throwaway organization + API key + one card, pushes it through the
real HTTP API multiple times (staging a fake completed audit in between via
direct SQL, since running the real audit pipeline costs LLM tokens), and
reads push_log/push_metrics_by_date back to confirm the trigger and view
behave as migration 027 defines. Everything created is removed in a
`finally` block, including on assertion failure or Ctrl-C — unless --keep
is passed.

Run from the repo root against a running API:

    python e2e/tests/test_push_log_smoke.py local
    python e2e/tests/test_push_log_smoke.py https://medkard.example --keep

  local        resolves to http://localhost:{API_PORT}, API_PORT read from
               .env (default 8000 if unset — see .env.example)
  --keep       leave the org/key/card/push_log rows behind for manual
               inspection instead of tearing down; prints what was left
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))

load_dotenv(ROOT / ".env")

from api_keys import ApiKeyFixtures, issue_key  # noqa: E402
from cards import CardFixtures, push_card  # noqa: E402
from organizations import OrganizationFixtures  # noqa: E402

_parser = argparse.ArgumentParser(description="Smoke-test push_log/push_metrics_by_date against a running API")
_parser.add_argument(
    "url",
    help='Base URL of the API, e.g. http://localhost:8000, or "local"/"localhost" '
    "for http://localhost:{API_PORT}",
)
_parser.add_argument("--keep", action="store_true", help="Skip teardown and print what was left behind")
_args = _parser.parse_args()


def _resolve_base_url(url_arg: str) -> str:
    """"local"/"localhost" -> http://localhost:{API_PORT} (API_PORT from .env, default 8000).

    Any other value is used as-is, exactly like the existing smoke scripts'
    positional url argument.
    """
    if url_arg in ("local", "localhost"):
        port = os.environ.get("API_PORT", "8000")
        return f"http://localhost:{port}"
    return url_arg


BASE = _resolve_base_url(_args.url)
TAG = uuid.uuid4().hex[:8]
ORG_NAME = f"e2e-push-log-{TAG}"
KEY_LABEL = f"e2e-push-log-{TAG}"
CARD_GUID = f"e2e-push-log-{TAG}-{uuid.uuid4()}"

_PASS, _FAIL = "  \033[32mok\033[0m", "  \033[31mFAILED\033[0m"
_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record an assertion without aborting — teardown still has to run."""
    if condition:
        print(f"{_PASS}  {label}")
    else:
        print(f"{_FAIL}  {label}{(' — ' + detail) if detail else ''}")
        _failures.append(label)


async def run(client: httpx.AsyncClient, org_id: str, raw_key: str, card_fixtures: CardFixtures) -> None:
    """Filled in by the next task — the actual push_log test scenario."""
    print("\n(scenario not yet implemented)")


async def main() -> int:
    org_id: str | None = None

    print(f"Smoke test push_log/push_metrics_by_date against {BASE}")
    print(f"  org={ORG_NAME}  card_guid={CARD_GUID}")

    async with OrganizationFixtures() as org_fixtures, CardFixtures() as card_fixtures:
        try:
            org_id = await org_fixtures.create_org(ORG_NAME)
            key_id, raw_key = await issue_key(KEY_LABEL, org_id)
            print(f"  created org id={org_id}, key id={key_id}")

            async with httpx.AsyncClient(timeout=30) as client:
                await run(client, org_id, raw_key, card_fixtures)
        finally:
            if _args.keep:
                print(f"\n--keep: leaving org={ORG_NAME} (id={org_id}) card_guid={CARD_GUID.lower()}")
            else:
                print("\nCleaning up ...")
                deleted_log = await card_fixtures.delete_push_log(CARD_GUID.lower())
                print(f"  deleted {deleted_log} push_log row(s)")
                deleted_cards = await card_fixtures.delete_cards(CARD_GUID.lower())
                print(f"  deleted {deleted_cards} done_cards row(s)")
                async with ApiKeyFixtures() as key_fixtures:
                    dropped = await key_fixtures.delete_key(KEY_LABEL)
                    print(f"  deleted {dropped} api key row(s) (scopes cascaded)")
                if org_id is not None:
                    await org_fixtures.delete_org(org_id)
                    print("  deleted organization")

    if _failures:
        print(f"\n\033[31m{len(_failures)} check(s) FAILED:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("\n\033[32mAll checks passed.\033[0m")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
