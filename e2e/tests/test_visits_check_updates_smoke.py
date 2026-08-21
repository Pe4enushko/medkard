#!/usr/bin/env python3
"""
End-to-end smoke test for GET /visits/check_updates.

Creates a throwaway organization + API key + one card, pushes it and
deliberately leaves it pending (no staging as done), and asserts
/visits/check_updates returns the card for an inclusive `since` boundary
at-or-before the push and omits it for a `since` strictly after the push.

Run from the repo root against a running API:

    python e2e/tests/test_visits_check_updates_smoke.py
    python e2e/tests/test_visits_check_updates_smoke.py https://medkard.example --keep

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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))

load_dotenv(ROOT / ".env")

from api_keys import ApiKeyFixtures, issue_key  # noqa: E402
from cards import CardFixtures, push_card  # noqa: E402
from organizations import OrganizationFixtures  # noqa: E402

_parser = argparse.ArgumentParser(description="Smoke-test GET /visits/check_updates against a running API")
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
ORG_NAME = f"e2e-checkupd-{TAG}"
KEY_LABEL = f"e2e-checkupd-{TAG}"
CARD_GUID = f"e2e-checkupd-{TAG}-{uuid.uuid4()}"

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


async def _db_now(card_fixtures: CardFixtures) -> str:
    """Return Postgres's own now() (ISO). done_cards.updated_at is stamped by the
    DB server's clock (migration 022's trigger), which can drift from this
    host's — comparing `since` boundaries against a locally-captured timestamp
    is flaky under any clock skew, so every boundary here comes from the DB."""
    async with card_fixtures._pool.connection() as conn:
        cur = await conn.execute("SELECT now()")
        row = await cur.fetchone()
        return row["now"].isoformat()


async def run(client: httpx.AsyncClient, raw_key: str, card_fixtures: CardFixtures) -> None:
    guid = CARD_GUID.lower()
    before_push = await _db_now(card_fixtures)

    print("\n1. Push a card, leave it pending (no staging)")
    resp = await push_card(client, BASE, ORG_NAME, raw_key, _mock_card())
    check("push accepted (200)", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")

    row = await card_fixtures.card_row(guid)
    check("card is pending after push", row is not None and row["status"] == "pending", f"row={row}")

    after_push = await _db_now(card_fixtures)

    print("\n2. check_updates without since includes the pending card")
    resp = await client.get(
        f"{BASE.rstrip('/')}/visits/check_updates",
        params={"org": ORG_NAME},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    check("check_updates accepted (200)", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")
    rows = resp.json() if resp.status_code == 200 else []
    check("pending card present without since", any(r["card_guid"] == guid for r in rows), f"got {len(rows)} row(s)")

    print("\n3. check_updates?since=<before push> includes the card (inclusive boundary)")
    resp = await client.get(
        f"{BASE.rstrip('/')}/visits/check_updates",
        params={"org": ORG_NAME, "since": before_push},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    rows = resp.json() if resp.status_code == 200 else []
    check("card present for since=before-push", any(r["card_guid"] == guid for r in rows), f"got {len(rows)} row(s)")

    print("\n4. check_updates?since=<after push> excludes the card")
    resp = await client.get(
        f"{BASE.rstrip('/')}/visits/check_updates",
        params={"org": ORG_NAME, "since": after_push},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    rows = resp.json() if resp.status_code == 200 else []
    check("card absent for since=after-push", not any(r["card_guid"] == guid for r in rows), f"got {len(rows)} row(s)")


async def main() -> int:
    org_id: str | None = None
    key_id: str | None = None

    print(f"Smoke test GET /visits/check_updates against {BASE}")
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
