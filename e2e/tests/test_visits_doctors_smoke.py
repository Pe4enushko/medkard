#!/usr/bin/env python3
"""
End-to-end smoke test for GET /visits/doctors.

Creates a throwaway organization + API key + three cards, stages them as
audited ("done") cards with controlled Прием.Врач_код/Врач, and asserts
/visits/doctors returns the unique (code, name) doctors of the org sorted
by name, deduplicated by code with the latest name winning.

Run from the repo root against a running API:

    python e2e/tests/test_visits_doctors_smoke.py
    python e2e/tests/test_visits_doctors_smoke.py https://medkard.example --keep

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

_parser = argparse.ArgumentParser(description="Smoke-test GET /visits/doctors against a running API")
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
ORG_NAME = f"e2e-doctors-{TAG}"
KEY_LABEL = f"e2e-doctors-{TAG}"
CARD_GUID_A = f"e2e-doctors-{TAG}-a-{uuid.uuid4()}"
CARD_GUID_B = f"e2e-doctors-{TAG}-b-{uuid.uuid4()}"
CARD_GUID_C = f"e2e-doctors-{TAG}-c-{uuid.uuid4()}"

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
    guid_a, guid_b, guid_c = CARD_GUID_A.lower(), CARD_GUID_B.lower(), CARD_GUID_C.lower()
    today = datetime.now(timezone.utc).strftime("%d.%m.%Y")
    doc_code_1, doc_code_2 = f"e2e-doc1-{TAG}", f"e2e-doc2-{TAG}"

    print("\n1. Push and stage two cards with two different doctors")
    resp = await push_card(client, BASE, ORG_NAME, raw_key, _mock_card(CARD_GUID_A))
    check("push A accepted (200)", resp.status_code == 200, f"got {resp.status_code}")
    await card_fixtures.stage_done_with_meta(guid_a, visit_date=today, doctor_code=doc_code_1, doctor_name="Бетова Анна")

    resp = await push_card(client, BASE, ORG_NAME, raw_key, _mock_card(CARD_GUID_B))
    check("push B accepted (200)", resp.status_code == 200, f"got {resp.status_code}")
    await card_fixtures.stage_done_with_meta(guid_b, visit_date=today, doctor_code=doc_code_2, doctor_name="Азарова Ирина")

    print("\n2. doctors returns both, sorted by name")
    resp = await client.get(
        f"{BASE.rstrip('/')}/visits/doctors",
        params={"org": ORG_NAME},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    check("doctors accepted (200)", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")
    entries = resp.json() if resp.status_code == 200 else []
    codes_seen = {e["code"] for e in entries}
    check("both doctor codes present", {doc_code_1, doc_code_2} <= codes_seen, f"got {codes_seen}")
    ours = [e for e in entries if e["code"] in {doc_code_1, doc_code_2}]
    check(
        "our two doctors sorted by name (Азарова before Бетова)",
        [e["code"] for e in ours] == [doc_code_2, doc_code_1],
        f"got order={ours}",
    )

    print("\n3. push+stage a third card renaming doctor 1 — latest name wins, no duplicate code")
    resp = await push_card(client, BASE, ORG_NAME, raw_key, _mock_card(CARD_GUID_C))
    check("push C accepted (200)", resp.status_code == 200, f"got {resp.status_code}")
    await card_fixtures.stage_done_with_meta(guid_c, visit_date=today, doctor_code=doc_code_1, doctor_name="Бетова Анна Викторовна")

    resp = await client.get(
        f"{BASE.rstrip('/')}/visits/doctors",
        params={"org": ORG_NAME},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    entries = resp.json() if resp.status_code == 200 else []
    matches = [e for e in entries if e["code"] == doc_code_1]
    check("doctor code 1 appears exactly once", len(matches) == 1, f"got {matches}")
    if matches:
        check(
            "doctor code 1 has the latest name",
            matches[0]["name"] == "Бетова Анна Викторовна",
            f"got {matches[0]}",
        )


async def main() -> int:
    org_id: str | None = None
    key_id: str | None = None

    print(f"Smoke test GET /visits/doctors against {BASE}")
    print(f"  org={ORG_NAME}  card_guids={CARD_GUID_A}, {CARD_GUID_B}, {CARD_GUID_C}")

    async with OrganizationFixtures() as org_fixtures, CardFixtures() as card_fixtures:
        try:
            org_id = await org_fixtures.create_org(ORG_NAME)
            key_id, raw_key = await issue_key(KEY_LABEL, org_id)
            print(f"  created org id={org_id}, key id={key_id}")

            async with httpx.AsyncClient(timeout=30) as client:
                await run(client, raw_key, card_fixtures)
        finally:
            if _args.keep:
                print(
                    f"\n--keep: leaving org={ORG_NAME} (id={org_id}) "
                    f"card_guids={CARD_GUID_A.lower()}, {CARD_GUID_B.lower()}, {CARD_GUID_C.lower()}"
                )
            else:
                print("\nCleaning up ...")
                deleted_a = await card_fixtures.delete_cards(CARD_GUID_A.lower())
                deleted_b = await card_fixtures.delete_cards(CARD_GUID_B.lower())
                deleted_c = await card_fixtures.delete_cards(CARD_GUID_C.lower())
                print(f"  deleted {deleted_a + deleted_b + deleted_c} done_cards row(s)")
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
