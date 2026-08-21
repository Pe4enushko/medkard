#!/usr/bin/env python3
"""
End-to-end smoke test for GET /visits/check.

Creates a throwaway organization + API key + one card, stages it as an
audited ("done") card with a controlled Прием.DATE, and asserts /visits/check
counts it on that date and not on other dates, plus the auth/404 paths.

Run from the repo root against a running API:

    python e2e/tests/test_visits_check_smoke.py
    python e2e/tests/test_visits_check_smoke.py https://medkard.example --keep

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
from datetime import datetime, timedelta, timezone
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

_parser = argparse.ArgumentParser(description="Smoke-test GET /visits/check against a running API")
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
ORG_NAME = f"e2e-check-{TAG}"
KEY_LABEL = f"e2e-check-{TAG}"
CARD_GUID = f"e2e-check-{TAG}-{uuid.uuid4()}"

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
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%d.%m.%Y")

    print("\n1. Push and stage the card as done, dated today")
    resp = await push_card(client, BASE, ORG_NAME, raw_key, _mock_card())
    check("push accepted (200)", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")
    await card_fixtures.stage_done_with_meta(guid, visit_date=today)

    print("\n2. check?date=today counts the card")
    resp = await client.get(
        f"{BASE.rstrip('/')}/visits/check",
        params={"org": ORG_NAME, "date": datetime.now(timezone.utc).date().isoformat()},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    check("check today accepted (200)", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")
    if resp.status_code == 200:
        body = resp.json()
        check("count == 1 for today", body.get("count") == 1, f"body={body}")

    print("\n3. check?date=yesterday counts nothing")
    resp = await client.get(
        f"{BASE.rstrip('/')}/visits/check",
        params={"org": ORG_NAME, "date": (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    check("check yesterday accepted (200)", resp.status_code == 200, f"got {resp.status_code}")
    if resp.status_code == 200:
        body = resp.json()
        check("count == 0 for yesterday", body.get("count") == 0, f"body={body}")

    print("\n4. check against an unknown org -> 404")
    resp = await client.get(
        f"{BASE.rstrip('/')}/visits/check",
        params={"org": f"unknown-org-{TAG}", "date": datetime.now(timezone.utc).date().isoformat()},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    check("unknown org -> 404", resp.status_code == 404, f"got {resp.status_code}: {resp.text[:200]}")

    print("\n5. check without Authorization -> 401")
    resp = await client.get(
        f"{BASE.rstrip('/')}/visits/check",
        params={"org": ORG_NAME, "date": datetime.now(timezone.utc).date().isoformat()},
    )
    check("missing auth -> 401", resp.status_code == 401, f"got {resp.status_code}: {resp.text[:200]}")


async def main() -> int:
    org_id: str | None = None
    key_id: str | None = None

    print(f"Smoke test GET /visits/check against {BASE}")
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
