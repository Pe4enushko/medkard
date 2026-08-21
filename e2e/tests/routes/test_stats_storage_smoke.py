#!/usr/bin/env python3
"""
End-to-end smoke test for GET /stats/storage.

Creates a throwaway organization + API key, checks that a freshly created org
reports zero storage, pushes and stages one card, then asserts the storage
figures reflect it (done_cards_kb > 0, total_kb == done_cards_kb + push_log_kb,
organization field echoes the org name).

Run from the repo root against a running API:

    python e2e/tests/routes/test_stats_storage_smoke.py
    python e2e/tests/routes/test_stats_storage_smoke.py https://medkard.example --keep

  url (optional) defaults to "local", which resolves to http://localhost:{API_PORT},
               API_PORT read from .env (default 8000 if unset — see .env.example).
  --keep       leave the org/key/card behind for manual inspection instead of
               tearing down; prints what was left
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helpers"))

load_dotenv(ROOT / ".env")

from api_keys import ApiKeyFixtures, issue_key  # noqa: E402
from cards import CardFixtures, push_card  # noqa: E402
from organizations import OrganizationFixtures  # noqa: E402

_parser = argparse.ArgumentParser(description="Smoke-test GET /stats/storage against a running API")
_parser.add_argument(
    "url",
    nargs="?",
    default="local",
    help='Base URL of the API, e.g. http://localhost:8000, or "local"/"localhost" '
    "for http://localhost:{API_PORT} (default: local)",
)
_parser.add_argument("--keep", action="store_true", help="Skip teardown and print what was left behind")
_args = _parser.parse_args()


def _resolve_base_url(url_arg: str) -> str:
    if url_arg in ("local", "localhost"):
        port = os.environ.get("API_PORT", "8000")
        return f"http://localhost:{port}"
    return url_arg


BASE = _resolve_base_url(_args.url)
TAG = uuid.uuid4().hex[:8]
ORG_NAME = f"e2e-stats-{TAG}"
KEY_LABEL = f"e2e-stats-{TAG}"
CARD_GUID = f"e2e-stats-{TAG}-{uuid.uuid4()}"

_PASS, _FAIL = "  \033[32mok\033[0m", "  \033[31mFAILED\033[0m"
_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"{_PASS}  {label}")
    else:
        print(f"{_FAIL}  {label}{(' — ' + detail) if detail else ''}")
        _failures.append(label)


def _mock_card() -> dict:
    return {
        "Прием": {"GUID": CARD_GUID, "DATE": datetime.now(timezone.utc).strftime("%d.%m.%Y"), "TYPE": "Первичный"},
        "Пациент": {"Возраст": "42"},
        "e2e_tag": TAG,
    }


async def run(client: httpx.AsyncClient, raw_key: str, card_fixtures: CardFixtures) -> None:
    guid = CARD_GUID.lower()
    today = datetime.now(timezone.utc).strftime("%d.%m.%Y")

    print("\n1. Baseline storage stats for a freshly created org")
    resp = await client.get(
        f"{BASE.rstrip('/')}/stats/storage",
        params={"org": ORG_NAME},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    check("baseline stats accepted (200)", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")
    baseline = resp.json() if resp.status_code == 200 else {}
    check("baseline done_cards_kb == 0", baseline.get("done_cards_kb") == 0, f"got {baseline}")
    check("baseline push_log_kb == 0", baseline.get("push_log_kb") == 0, f"got {baseline}")
    check("baseline total_kb == 0", baseline.get("total_kb") == 0, f"got {baseline}")

    print("\n2. Push and stage one card")
    resp = await push_card(client, BASE, ORG_NAME, raw_key, _mock_card())
    check("push accepted (200)", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")
    await card_fixtures.stage_done_with_meta(guid, visit_date=today)

    print("\n3. Storage stats now reflect the card")
    resp = await client.get(
        f"{BASE.rstrip('/')}/stats/storage",
        params={"org": ORG_NAME},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    check("stats accepted (200)", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")
    stats = resp.json() if resp.status_code == 200 else {}
    check("done_cards_kb > 0", stats.get("done_cards_kb", 0) > 0, f"got {stats}")
    check(
        "total_kb == done_cards_kb + push_log_kb",
        stats.get("total_kb") == stats.get("done_cards_kb", 0) + stats.get("push_log_kb", 0),
        f"got {stats}",
    )
    check("organization field matches", stats.get("organization") == ORG_NAME, f"got {stats}")


async def main() -> int:
    org_id: str | None = None
    key_id: str | None = None

    print(f"Smoke test GET /stats/storage against {BASE}")
    print(f"  org={ORG_NAME}  card_guid={CARD_GUID}")

    async with OrganizationFixtures() as org_fixtures, CardFixtures() as card_fixtures:
        try:
            org_id = await org_fixtures.create_org(ORG_NAME)
            key_id, raw_key = await issue_key(KEY_LABEL, org_id)
            print(f"  created org id={org_id}, key id={key_id}")

            async with httpx.AsyncClient(timeout=30) as client:
                await run(client, raw_key, card_fixtures)
        finally:
            if _args.keep:
                print(f"\n--keep: leaving org={ORG_NAME} (id={org_id}) card_guid={CARD_GUID.lower()}")
            else:
                print("\nCleaning up ...")
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
