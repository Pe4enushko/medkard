#!/usr/bin/env python3
"""
Issue a new pull-API key for the integrating app, scoped to specific
organizations.

One unified key authenticates the app itself, but only grants access to
the organizations named here — every key must be scoped to at least one
org; each request still names which org's cards it wants via ?org=.

Run from project root:
    python scripts/operator/create-api-key.py "1C integration" --orgs Alenka MDS
    python scripts/operator/create-api-key.py "1C integration" --orgs Alenka

Prints the raw key to stdout exactly once. Only its hash is stored in the
database — if the key is lost, issue a new one and revoke the old one
(scripts/operator/revoke-api-key.py <key-id>, printed alongside the new key).
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from storage.api_keys_storage import ApiKeysStorage  # noqa: E402
from storage.organizations_storage import OrganizationsStorage  # noqa: E402

_parser = argparse.ArgumentParser(description="Issue a new pull-API key scoped to specific organizations")
_parser.add_argument("label", help="Human-readable label for this key, e.g. the integrating app's name")
_parser.add_argument(
    "--orgs", nargs="+", required=True, metavar="ORG",
    help="One or more exact organization names this key may access, e.g. --orgs Alenka MDS",
)
_args = _parser.parse_args()


def _generate_key() -> str:
    return f"medkard_{secrets.token_urlsafe(32)}"


async def main() -> None:
    async with OrganizationsStorage() as organizations:
        org_ids = [await organizations.get_id_by_name(name) for name in _args.orgs]

    raw_key = _generate_key()
    async with ApiKeysStorage() as api_keys:
        key_id = await api_keys.create_key(_args.label, raw_key, org_ids)

    print(f"New API key '{_args.label}' (id={key_id}), scoped to: {', '.join(_args.orgs)}\n")
    print(f"    {raw_key}\n")
    print("Save this now — it cannot be shown again.")
    print(f"To revoke it later: python scripts/operator/revoke-api-key.py {key_id}")


if __name__ == "__main__":
    asyncio.run(main())
