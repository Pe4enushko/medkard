#!/usr/bin/env python3
"""
End-to-end smoke test for POST /cards/push + GET /cards/check_updates.

Exercises the pairing the check_updates endpoint exists for: a card pushed by
1C lands as status='pending' with no audit results, and check_updates must
hand it to the consumer straight away — while /cards/export, which is
audited-only, must NOT. Also asserts the inclusive `since` boundary, since
that is what keeps cards from slipping between two polls.

Key issuing and revoking go through scripts/create-api-key.py and
scripts/revoke-api-key.py as subprocesses rather than reimplementing them, so
this exercises the real operator path (they own the key format and the
org-scoping join). Creating the throwaway organization is the one thing done
directly against the DB — no script or route creates organizations.

Everything it creates is namespaced with a random tag and removed on the way
out, including after a failed assertion or Ctrl-C: a throwaway organization,
an api key scoped to it, and the pushed card. Nothing pre-existing is touched.

Run from project root against a running API (reads POSTGRES_* from .env for
the verification/teardown side):

    python scripts/smoke-push-check-updates.py http://localhost:8000
    python scripts/smoke-push-check-updates.py https://medkard.example --keep

  --keep   leave the org/key/card behind for manual poking (prints them)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from storage.base import BaseStorage  # noqa: E402

_parser = argparse.ArgumentParser(description="Smoke-test push + check_updates against a running API")
_parser.add_argument("url", help="Base URL of the API, e.g. http://localhost:8000")
_parser.add_argument("--keep", action="store_true", help="Skip teardown and print what was left behind")
_args = _parser.parse_args()

BASE = _args.url.rstrip("/")
TAG = uuid.uuid4().hex[:8]
ORG_NAME = f"smoke-check-updates-{TAG}"
CARD_GUID = f"smoke-{TAG}-{uuid.uuid4()}"

_PASS, _FAIL = "  \033[32mok\033[0m", "  \033[31mFAILED\033[0m"
_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record an assertion without aborting — teardown still has to run."""
    if condition:
        print(f"{_PASS}  {label}")
    else:
        print(f"{_FAIL}  {label}{(' — ' + detail) if detail else ''}")
        _failures.append(label)


class _Fixtures(BaseStorage):
    """Direct DB access for the bits the API deliberately has no routes for
    (creating/removing an organization) and for teardown."""

    async def create_org(self, name: str) -> str:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO organizations (name) VALUES (%(n)s) RETURNING id", {"n": name}
            )
            return str((await cur.fetchone())["id"])

    async def delete_org(self, org_id: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute("DELETE FROM organizations WHERE id = %(o)s::uuid", {"o": org_id})

    async def delete_cards(self, org_id: str) -> int:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM done_cards WHERE organization_id = %(o)s::uuid", {"o": org_id}
            )
            return cur.rowcount

    async def card_row(self, guid: str) -> dict | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT card_guid, status, ignored, broken, updated_at, "
                "       formal_result, diag_result "
                "FROM done_cards WHERE card_guid = %(g)s",
                {"g": guid},
            )
            return await cur.fetchone()


def _run_script(name: str, *args: str) -> str:
    """Invoke a sibling script the way an operator would, and return its stdout."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *args],
        capture_output=True, text=True, cwd=ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"scripts/{name} failed ({proc.returncode}):\n{proc.stderr.strip()}")
    return proc.stdout


def issue_key(label: str, org_name: str) -> tuple[str, str]:
    """Issue a key via scripts/create-api-key.py, parsing back what it prints.

    It prints the raw key exactly once and the id in the same breath, so both
    are scraped here rather than re-deriving them from the DB.
    """
    out = _run_script("create-api-key.py", label, "--orgs", org_name)
    key_id = re.search(r"id=([0-9a-f-]{36})", out)
    raw_key = re.search(r"^\s+(medkard_\S+)\s*$", out, re.MULTILINE)
    if not key_id or not raw_key:
        raise RuntimeError(f"could not parse create-api-key.py output:\n{out}")
    return key_id.group(1), raw_key.group(1)


def _mock_card() -> dict:
    """Minimal card shaped like 1C's payload — push only requires Прием.GUID."""
    return {
        "Прием": {
            "GUID": CARD_GUID,
            "DATE": datetime.now(timezone.utc).strftime("%d.%m.%Y"),
            "TYPE": "Первичный",
        },
        "Пациент": {"Возраст": "42"},
        "smoke_tag": TAG,
    }


async def run(client: httpx.AsyncClient, org_id: str, raw_key: str, fixtures: _Fixtures) -> None:
    auth = {"Authorization": f"Bearer {raw_key}"}
    org_q = {"org": ORG_NAME}

    # A timestamp from before the push, so every query below is bounded to this
    # run's own row instead of scanning the org's history.
    before_push = datetime.now(timezone.utc) - timedelta(seconds=5)

    print("\n1. POST /cards/push")
    resp = await client.post(f"{BASE}/cards/push", params=org_q, json=_mock_card(), headers=auth)
    check("push accepted (200)", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")
    if resp.status_code == 200:
        body = resp.json()
        check("push reports status=pending", body.get("status") == "pending", json.dumps(body))
        check(
            "push echoes the card_guid (lowercased)",
            body.get("card_guid") == CARD_GUID.lower(),
            f"{body.get('card_guid')} != {CARD_GUID.lower()}",
        )

    row = await fixtures.card_row(CARD_GUID.lower())
    check("card is in done_cards", row is not None)
    if row is not None:
        check("stored status is 'pending'", row["status"] == "pending", str(row["status"]))
        check(
            "no audit results yet",
            row["formal_result"] is None and row["diag_result"] is None,
        )

    print("\n2. GET /cards/check_updates — the pending card must come back")
    resp = await client.get(
        f"{BASE}/cards/check_updates",
        params={**org_q, "since": before_push.isoformat()},
        headers=auth,
    )
    check("check_updates 200", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")
    pushed = None
    if resp.status_code == 200:
        rows = resp.json()
        by_guid = {r["card_guid"]: r for r in rows}
        pushed = by_guid.get(CARD_GUID.lower())
        check("pushed card present", pushed is not None, f"{len(rows)} row(s), none matching")
        if pushed is not None:
            check("status surfaced as 'pending'", pushed.get("status") == "pending", str(pushed.get("status")))
            check("raw card_data returned", isinstance(pushed.get("card_data"), dict))
            check(
                "card_data survived the round trip",
                (pushed.get("card_data") or {}).get("smoke_tag") == TAG,
            )

    print("\n3. GET /cards/export — audited-only, must NOT see it")
    resp = await client.get(
        f"{BASE}/cards/export",
        params={**org_q, "since": before_push.isoformat()},
        headers=auth,
    )
    check("export 200", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")
    if resp.status_code == 200:
        guids = {r["card_guid"] for r in resp.json()}
        check(
            "pending card excluded from export",
            CARD_GUID.lower() not in guids,
            "export leaked an unaudited card",
        )

    print("\n4. Inclusive `since` boundary")
    if pushed is not None:
        resp = await client.get(
            f"{BASE}/cards/check_updates",
            params={**org_q, "since": pushed["updated_at"]},
            headers=auth,
        )
        ok = resp.status_code == 200
        check("check_updates 200 at exact boundary", ok, f"got {resp.status_code}")
        if ok:
            guids = {r["card_guid"] for r in resp.json()}
            check(
                "since == the card's own updated_at still returns it",
                CARD_GUID.lower() in guids,
                "a strict > would have dropped it here",
            )
    else:
        check("boundary check", False, "skipped — no card from step 2")

    print("\n5. Default window (no `since`)")
    resp = await client.get(f"{BASE}/cards/check_updates", params=org_q, headers=auth)
    check("check_updates 200 without since", resp.status_code == 200, f"got {resp.status_code}")
    if resp.status_code == 200:
        guids = {r["card_guid"] for r in resp.json()}
        check("card present in the default week window", CARD_GUID.lower() in guids)

    print("\n6. Auth")
    resp = await client.get(f"{BASE}/cards/check_updates", params=org_q)
    check("no key → 401/403", resp.status_code in (401, 403), f"got {resp.status_code}")

    resp = await client.get(
        f"{BASE}/cards/check_updates", params=org_q,
        headers={"Authorization": f"Bearer medkard_not_a_real_key_{TAG}"},
    )
    check("bad key → 401/403", resp.status_code in (401, 403), f"got {resp.status_code}")

    resp = await client.get(f"{BASE}/cards/check_updates", headers=auth)
    check("missing ?org= → 422", resp.status_code == 422, f"got {resp.status_code}")


async def main() -> int:
    org_id: str | None = None
    key_id: str | None = None
    raw_key: str | None = None

    print(f"Smoke test push + check_updates against {BASE}")
    print(f"  org={ORG_NAME}  card_guid={CARD_GUID}")

    async with _Fixtures() as fixtures:
        try:
            org_id = await fixtures.create_org(ORG_NAME)
            key_id, raw_key = issue_key(f"smoke-{TAG}", ORG_NAME)
            print(f"  created org id={org_id}, key id={key_id} (via create-api-key.py)")

            async with httpx.AsyncClient(timeout=30) as client:
                await run(client, org_id, raw_key, fixtures)
        finally:
            # Teardown runs even on assertion failure or Ctrl-C — the stand must
            # not accumulate smoke-test orgs.
            if _args.keep:
                print(f"\n--keep: leaving org={ORG_NAME} (id={org_id}) key={raw_key}")
            else:
                print("\nCleaning up ...")
                if org_id is not None:
                    deleted = await fixtures.delete_cards(org_id)
                    print(f"  deleted {deleted} card(s)")
                if key_id is not None:
                    _run_script("revoke-api-key.py", key_id)
                    print("  revoked api key (via revoke-api-key.py)")
                if org_id is not None:
                    # api_key_organizations rows cascade from the org (migration 018).
                    await fixtures.delete_org(org_id)
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
