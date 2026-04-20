"""
Test 1C connection — fetches today's JSON payload and prints it.

Requires ONE_C_APPOINTMENTS_URL, ONE_C_LOGIN, ONE_C_PASSWORD env vars to be set.
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from integrations.one_c import OneCClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    client = OneCClient.from_env()
    print(client.url)
    print(client.login)
    data = client.fetch_json_for_today()

    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json"
    out_path = PROJECT_ROOT / filename
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
