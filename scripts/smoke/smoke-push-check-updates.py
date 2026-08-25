#!/usr/bin/env python3
"""
End-to-end smoke test for POST /visits/push + GET /visits/check_updates.

Exercises the pairing the check_updates endpoint exists for: a card pushed by
1C lands as status='pending' with no audit results, and check_updates must
hand it to the consumer straight away, raw card_data and all. Also asserts
the inclusive `since` boundary, since that is what keeps cards from slipping
between two polls.

Scope is push + check_updates only. Other routes are left alone even where
their behaviour contrasts (export being audited-only, say) — asserting on
them here would be testing code this change never touched.

Keys are minted and dropped here rather than via create-api-key.py /
revoke-api-key.py: those are built for an operator, so the key id would have
to be scraped from printed output, and revoke only sets revoked_at — right
for a real key, but it would leave a dead row behind on every run.

Everything it creates is namespaced with a random tag and removed on the way
out, including after a failed assertion or Ctrl-C: the pushed card, the api
key row (and its org scoping, which cascades), and the throwaway organization.
Nothing pre-existing is touched.

Run from project root against a running API (reads POSTGRES_* from .env for
the verification/teardown side):

    python scripts/smoke/smoke-push-check-updates.py http://localhost:8000
    python scripts/smoke/smoke-push-check-updates.py https://medkard.example --keep

  --keep   leave the org/key/card behind for manual poking (prints them)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from storage.api_keys_storage import ApiKeysStorage  # noqa: E402
from storage.base import BaseStorage  # noqa: E402

_parser = argparse.ArgumentParser(description="Smoke-test push + check_updates against a running API")
_parser.add_argument("url", help="Base URL of the API, e.g. http://localhost:8000")
_parser.add_argument("--keep", action="store_true", help="Skip teardown and print what was left behind")
_args = _parser.parse_args()

BASE = _args.url.rstrip("/")
TAG = uuid.uuid4().hex[:8]
ORG_NAME = f"smoke-check-updates-{TAG}"
KEY_LABEL = f"smoke-{TAG}"
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

    async def delete_keys_by_label(self, label: str) -> int:
        """Delete this run's key row(s) outright.

        revoke-api-key.py only sets revoked_at — right for a real key, whose
        history is worth keeping, but it would leave a dead row behind on every
        smoke run. This key is minted here, never handed to anyone, and lives
        for seconds, so there is no audit trail to preserve.

        By label rather than by id: create_key inserts the key and its org
        scoping as two statements without an explicit transaction, so a failure
        between them leaves an inserted key whose id the caller never received.
        Nothing ties a key to an organization directly, so dropping the org does
        NOT cascade to it — verified on a real Postgres: the org and scope rows
        go, the key survives. The label is smoke-<uuid4>, unique to this run, so
        this cannot match anyone else's key; the scope rows go with it
        (ON DELETE CASCADE, migration 018).
        """
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM api_keys WHERE label = %(l)s", {"l": label}
            )
            return cur.rowcount

    async def count_key_scopes(self, key_id: str) -> int:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT count(*) AS n FROM api_key_organizations WHERE api_key_id = %(k)s::uuid",
                {"k": key_id},
            )
            return (await cur.fetchone())["n"]

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


async def issue_key(label: str, org_id: str) -> tuple[str, str]:
    """Mint a key scoped to one org, returning (key_id, raw_key).

    Done through ApiKeysStorage rather than by shelling out to
    create-api-key.py: that script prints the id and raw key for a human, so
    reusing it would mean scraping its stdout, and the id is exactly what
    teardown needs to delete the row afterwards.
    """
    raw_key = f"medkard_smoke_{secrets.token_urlsafe(24)}"
    async with ApiKeysStorage() as api_keys:
        key_id = await api_keys.create_key(label, raw_key, [org_id])
    return str(key_id), raw_key


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

    print("\n1. POST /visits/push")
    resp = await client.post(f"{BASE}/visits/push", params=org_q, json=_mock_card(), headers=auth)
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

    print("\n2. GET /visits/check_updates — the pending card must come back")
    resp = await client.get(
        f"{BASE}/visits/check_updates",
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

    print("\n3. Inclusive `since` boundary")
    if pushed is not None:
        resp = await client.get(
            f"{BASE}/visits/check_updates",
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

    print("\n4. Default window (no `since`)")
    resp = await client.get(f"{BASE}/visits/check_updates", params=org_q, headers=auth)
    check("check_updates 200 without since", resp.status_code == 200, f"got {resp.status_code}")
    if resp.status_code == 200:
        guids = {r["card_guid"] for r in resp.json()}
        check("card present in the default week window", CARD_GUID.lower() in guids)

    print("\n5. Auth")
    resp = await client.get(f"{BASE}/visits/check_updates", params=org_q)
    check("no key → 401/403", resp.status_code in (401, 403), f"got {resp.status_code}")

    resp = await client.get(
        f"{BASE}/visits/check_updates", params=org_q,
        headers={"Authorization": f"Bearer medkard_not_a_real_key_{TAG}"},
    )
    check("bad key → 401/403", resp.status_code in (401, 403), f"got {resp.status_code}")

    resp = await client.get(f"{BASE}/visits/check_updates", headers=auth)
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
            key_id, raw_key = await issue_key(KEY_LABEL, org_id)
            print(f"  created org id={org_id}, key id={key_id}")

            async with httpx.AsyncClient(timeout=30) as client:
                await run(client, org_id, raw_key, fixtures)
        finally:
            # Teardown runs even on assertion failure or Ctrl-C — the stand must
            # not accumulate smoke-test rows.
            if _args.keep:
                print(f"\n--keep: leaving org={ORG_NAME} (id={org_id}) key={raw_key}")
            else:
                print("\nCleaning up ...")
                if org_id is not None:
                    deleted = await fixtures.delete_cards(org_id)
                    print(f"  deleted {deleted} card(s)")
                # Before the org: nothing links a key to an organization, so the
                # org's cascade would not take the key with it. By label rather
                # than by id, so a key inserted by a create_key that then failed
                # mid-way — leaving the caller without an id — is still removed.
                dropped = await fixtures.delete_keys_by_label(KEY_LABEL)
                print(f"  deleted {dropped} api key row(s) (scopes cascaded)")
                if key_id is not None:
                    left = await fixtures.count_key_scopes(key_id)
                    if left:
                        print(f"  \033[31mWARNING: {left} scope row(s) still present\033[0m")
                if org_id is not None:
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
