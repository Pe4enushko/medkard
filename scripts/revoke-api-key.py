#!/usr/bin/env python3
"""
Revoke a pull-API key, by its id or by the raw key itself — whichever you
have on hand.

Run from project root:
    python scripts/revoke-api-key.py <key-id>
    python scripts/revoke-api-key.py medkard_<...>
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from storage.api_keys_storage import ApiKeysStorage  # noqa: E402

_parser = argparse.ArgumentParser(description="Revoke a pull-API key by id or by its raw value")
_parser.add_argument("key_or_id", help="Key id (as printed by create-api-key.py) or the raw API key itself")
_args = _parser.parse_args()


def _looks_like_key_id(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


async def main() -> None:
    async with ApiKeysStorage() as api_keys:
        if _looks_like_key_id(_args.key_or_id):
            await api_keys.revoke_key(_args.key_or_id)
            print(f"Revoked key {_args.key_or_id}")
        else:
            revoked = await api_keys.revoke_by_raw_key(_args.key_or_id)
            if revoked:
                print("Revoked key")
            else:
                print("No active key matched — already revoked or never existed", file=sys.stderr)
                sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
