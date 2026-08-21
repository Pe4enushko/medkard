#!/usr/bin/env python3
"""
End-to-end smoke test for GET /visits/export.

Creates a throwaway organization + API key + two cards, stages both as
audited ("done") cards with a controlled Прием.DATE, and asserts
/visits/export returns them, that since/limit/cursor pagination behaves,
and that the include_ignored toggle hides/reveals a card marked ignored.

Run from the repo root against a running API:

    python e2e/tests/test_visits_export_smoke.py
    python e2e/tests/test_visits_export_smoke.py https://medkard.example --keep

  url (optional) defaults to "local", which resolves to http://localhost:{API_PORT},
               API_PORT read from .env (default 8000 if unset — see .env.example).
  --keep       leave the org/key/cards behind for manual inspection instead of
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

_parser = argparse.ArgumentParser(description="Smoke-test GET /visits/export against a running API")
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
ORG_NAME = f"e2e-export-{TAG}"
KEY_LABEL = f"e2e-export-{TAG}"
CARD_GUID_A = f"e2e-export-{TAG}-a-{uuid.uuid4()}"
CARD_GUID_B = f"e2e-export-{TAG}-b-{uuid.uuid4()}"

_PASS, _FAIL = "  \033[32mok\033[0m", "  \033[31mFAILED\033[0m"
_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"{_PASS}  {label}")
    else:
        print(f"{_FAIL}  {label}{(' — ' + detail) if detail else ''}")
        _failures.append(label)


def _mock_card(guid: str) -> dict:
    return {
        "Прием": {"GUID": guid, "DATE": datetime.now(timezone.utc).strftime("%d.%m.%Y"), "TYPE": "Первичный"},
        "Пациент": {"Возраст": "42"},
        "e2e_tag": TAG,
    }


async def run(client: httpx.AsyncClient, raw_key: str, card_fixtures: CardFixtures) -> None:
    guid_a, guid_b = CARD_GUID_A.lower(), CARD_GUID_B.lower()
    today = datetime.now(timezone.utc).strftime("%d.%m.%Y")

    print("\n1. Push and stage two cards as done")
    for guid_raw, guid in ((CARD_GUID_A, guid_a), (CARD_GUID_B, guid_b)):
        resp = await push_card(client, BASE, ORG_NAME, raw_key, _mock_card(guid_raw))
        check(f"push {guid_raw} accepted (200)", resp.status_code == 200, f"got {resp.status_code}")
        await card_fixtures.stage_done_with_meta(guid, visit_date=today)

    print("\n2. export without since/limit returns both cards, status=done")
    resp = await client.get(
        f"{BASE.rstrip('/')}/visits/export",
        params={"org": ORG_NAME},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    check("export accepted (200)", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")
    rows = resp.json() if resp.status_code == 200 else []
    guids_seen = {r["card_guid"] for r in rows}
    check("both cards present", {guid_a, guid_b} <= guids_seen, f"got guids={guids_seen}")
    check(
        "both rows have status=done",
        all(r["status"] == "done" for r in rows if r["card_guid"] in {guid_a, guid_b}),
        str(rows),
    )

    print("\n3. export?since=<future> returns nothing for these cards")
    future = (datetime.now(timezone.utc) + timedelta(days=3650)).isoformat()
    resp = await client.get(
        f"{BASE.rstrip('/')}/visits/export",
        params={"org": ORG_NAME, "since": future},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    check("export since-future accepted (200)", resp.status_code == 200, f"got {resp.status_code}")
    rows = resp.json() if resp.status_code == 200 else []
    check("no rows for since=future", len(rows) == 0, f"got {len(rows)} row(s)")

    print("\n4. pagination: limit=1 across two pages covers both cards without duplication")
    resp1 = await client.get(
        f"{BASE.rstrip('/')}/visits/export",
        params={"org": ORG_NAME, "limit": 1, "cursor": 0},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    resp2 = await client.get(
        f"{BASE.rstrip('/')}/visits/export",
        params={"org": ORG_NAME, "limit": 1, "cursor": 1},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    check("page 1 accepted (200)", resp1.status_code == 200, f"got {resp1.status_code}")
    check("page 2 accepted (200)", resp2.status_code == 200, f"got {resp2.status_code}")
    page1_guids = {r["card_guid"] for r in resp1.json()} if resp1.status_code == 200 else set()
    page2_guids = {r["card_guid"] for r in resp2.json()} if resp2.status_code == 200 else set()
    check("page 1 has exactly one row", len(page1_guids) == 1, f"got {page1_guids}")
    check("page 2 has exactly one row", len(page2_guids) == 1, f"got {page2_guids}")
    check("pages don't overlap", page1_guids.isdisjoint(page2_guids), f"page1={page1_guids} page2={page2_guids}")
    check(
        "union of both pages covers both cards",
        (page1_guids | page2_guids) >= {guid_a, guid_b},
        f"union={page1_guids | page2_guids}",
    )

    print("\n5. mark card B ignored; include_ignored toggles it in/out")
    await card_fixtures.mark_ignored(guid_b)

    resp = await client.get(
        f"{BASE.rstrip('/')}/visits/export",
        params={"org": ORG_NAME},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    guids_default = {r["card_guid"] for r in resp.json()} if resp.status_code == 200 else set()
    check("default export excludes ignored card B", guid_b not in guids_default, f"got {guids_default}")
    check("default export still includes card A", guid_a in guids_default, f"got {guids_default}")

    resp = await client.get(
        f"{BASE.rstrip('/')}/visits/export",
        params={"org": ORG_NAME, "include_ignored": "true"},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    guids_incl = {r["card_guid"] for r in resp.json()} if resp.status_code == 200 else set()
    check("include_ignored=true includes card B", guid_b in guids_incl, f"got {guids_incl}")
    if guid_b in guids_incl:
        row_b = next(r for r in resp.json() if r["card_guid"] == guid_b)
        check("card B status is 'ignored'", row_b["status"] == "ignored", f"row={row_b}")


async def main() -> int:
    org_id: str | None = None
    key_id: str | None = None

    print(f"Smoke test GET /visits/export against {BASE}")
    print(f"  org={ORG_NAME}  card_guids={CARD_GUID_A}, {CARD_GUID_B}")

    async with OrganizationFixtures() as org_fixtures, CardFixtures() as card_fixtures:
        try:
            org_id = await org_fixtures.create_org(ORG_NAME)
            key_id, raw_key = await issue_key(KEY_LABEL, org_id)
            print(f"  created org id={org_id}, key id={key_id}")

            async with httpx.AsyncClient(timeout=30) as client:
                await run(client, raw_key, card_fixtures)
        finally:
            if _args.keep:
                print(f"\n--keep: leaving org={ORG_NAME} (id={org_id}) card_guids={CARD_GUID_A.lower()}, {CARD_GUID_B.lower()}")
            else:
                print("\nCleaning up ...")
                deleted_a = await card_fixtures.delete_cards(CARD_GUID_A.lower())
                deleted_b = await card_fixtures.delete_cards(CARD_GUID_B.lower())
                print(f"  deleted {deleted_a + deleted_b} done_cards row(s)")
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
