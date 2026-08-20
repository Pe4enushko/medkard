# E2E Push Log Smoke Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable `e2e/tests/helpers/` package (organizations, API keys, cards) for
creating and tearing down throwaway fixtures against the real API + Postgres, and a standalone
smoke-test script that exercises `push_log`/`push_metrics_by_date` (migration 027) end to end.

**Architecture:** Three thin helper modules wrap `BaseStorage` (direct psycopg3 DB access) plus one
HTTP helper for `POST /visits/push`, following the exact pattern already used inline in
`scripts/smoke-push-check-updates.py`. A new top-level `e2e/` package (not under `src/`, not under
`tests/`) holds them plus one `argparse`-driven script, `e2e/tests/test_push_log_smoke.py`, that
creates a throwaway org + API key, drives a sequence of pushes through the real HTTP API, and
asserts on `push_log`/`push_metrics_by_date` state via direct SQL reads — cleaning up in a
`finally` block even on failure or Ctrl-C.

**Tech Stack:** Python 3.11, `httpx.AsyncClient` for HTTP, `psycopg3` via the existing
`BaseStorage` pool pattern, `python-dotenv` for `.env` loading — no new dependencies.

## Global Constraints

- Branch: `push-log-e2e-tests`, based on `origin/audit-overwrite-journal` (worktree already exists
  at `/home/okabe/projects/medkard/.worktrees/push-log-e2e`, HEAD at commit `a85dda4` — the spec
  commit). All work happens in this worktree.
- New top-level directory `e2e/` (sibling to `src/`, `tests/`, `scripts/`) — not under
  `pythonpath = src` (`pytest.ini`), and not run via `pytest`. Every module that needs
  `src`-rooted imports (`storage.*`) inserts `ROOT / "src"` into `sys.path` itself at import time,
  exactly as `scripts/smoke-push-check-updates.py:47-48` already does:
  ```python
  ROOT = Path(__file__).resolve().parent.parent.parent  # e2e/tests/helpers/x.py -> repo root
  sys.path.insert(0, str(ROOT / "src"))
  ```
  (The exact number of `.parent` calls depends on each file's depth under `e2e/` — computed per
  file in its task below.)
- `API_PORT` is a new `.env`/`.env.example` variable. Default in `.env.example` is `8000` — do
  NOT set it to `13742` or any other value; the user sets their own real value in their own
  `.env`.
- Helper modules are thin wrappers over `BaseStorage` (`src/storage/base.py`), reusing the shared
  connection pool pattern — no new pool, no new connection-string logic.
- Every fixture the smoke test creates (organization, API key, `done_cards` row, `push_log` rows)
  must be deleted in a `finally` block that runs even on assertion failure or `KeyboardInterrupt`,
  unless `--keep` is passed — matching `scripts/smoke-push-check-updates.py`'s existing contract.
- This is a standalone script with `argparse`, not a pytest test file — no `pytest.mark`, no
  fixtures in the pytest sense, no `assert` that aborts the run (use the existing repo's
  `check(label, condition, detail)` accumulator pattern from `smoke-push-check-updates.py:68-74` so
  teardown always runs and every check gets its own PASS/FAIL line).
- Card GUIDs, organization names, and API key labels must be namespaced with a random per-run tag
  (`uuid.uuid4().hex[:8]`, matching `smoke-push-check-updates.py:59`) so concurrent runs against
  the same shared Postgres never collide.

---

### Task 1: `API_PORT` in `.env.example` and `docker-compose.yml`

**Files:**
- Modify: `.env.example`
- Modify: `docker-compose.yml`

**Interfaces:** none (no Python interfaces — this is config only)

- [ ] **Step 1: Add `API_PORT` to `.env.example`**

In `.env.example`, find this block (near the end of the file):

```env
# Docker: bind the published pull-API port to this address only (e.g. the host's
# WireGuard interface IP) so the API is unreachable except through the tunnel.
WG_BIND_IP=
```

Replace with:

```env
# Docker: bind the published pull-API port to this address only (e.g. the host's
# WireGuard interface IP) so the API is unreachable except through the tunnel.
WG_BIND_IP=
# Host port the pull-API container is published on (docker-compose.yml) — also
# the default port e2e/tests/*.py use when given "local" as the target URL.
API_PORT=8000
```

- [ ] **Step 2: Point `docker-compose.yml`'s published port at `${API_PORT}`**

Replace the full contents of `docker-compose.yml`:

```yaml
services:
  api:
    build: .
    env_file: .env
    ports:
      - "${WG_BIND_IP}:${API_PORT}:8000"
    restart: unless-stopped
```

(Only the `ports:` line changes — `13742` becomes `${API_PORT}`. The container-internal port stays
literal `8000`, matching `Dockerfile:15`'s `--port 8000`.)

- [ ] **Step 3: Verify docker-compose config parses**

Run: `docker compose config`

Expected: no YAML/interpolation errors. If `API_PORT` is unset in the actual `.env` (not
`.env.example`), docker compose will print a warning like `WARN[0000] The "API_PORT" variable is
not set. Defaulting to a blank string.` — that's expected in an environment without a real `.env`
configured yet, and is not a failure of this step. If a real `.env` exists with `API_PORT` set,
confirm the rendered `ports:` line shows the real value substituted in place of `${API_PORT}`.

- [ ] **Step 4: Commit**

```bash
git add .env.example docker-compose.yml
git commit -m "feat(docker): move published API port to API_PORT env var"
```

---

### Task 2: `e2e/` package scaffolding + `organizations.py` helper

**Files:**
- Create: `e2e/__init__.py`
- Create: `e2e/tests/__init__.py`
- Create: `e2e/tests/helpers/__init__.py`
- Create: `e2e/tests/helpers/organizations.py`

**Interfaces:**
- Produces: `class OrganizationFixtures(BaseStorage)` with `async def create_org(self, name: str)
  -> str` and `async def delete_org(self, org_id: str) -> None`.

- [ ] **Step 1: Create the three empty `__init__.py` files**

```bash
mkdir -p e2e/tests/helpers
touch e2e/__init__.py e2e/tests/__init__.py e2e/tests/helpers/__init__.py
```

All three stay empty — they exist only to make `e2e`, `e2e.tests`, and `e2e.tests.helpers`
importable as packages if ever imported that way (the smoke test itself is run as a script, not
imported, but the helpers are imported by it via `sys.path` manipulation, so being a proper
package keeps `from helpers.organizations import ...`-style imports unambiguous).

- [ ] **Step 2: Write `e2e/tests/helpers/organizations.py`**

```python
"""
organizations.py — create/delete throwaway organizations for e2e tests.

Direct INSERT/DELETE against the organizations table: there is no public API
route for creating an organization (organizations are provisioned manually
today), so a throwaway test org has no contract to go through — it talks to
the table the same way any operator/migration would.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from storage.base import BaseStorage  # noqa: E402


class OrganizationFixtures(BaseStorage):
    """Async context-manager for creating/removing e2e-test organizations.

    Usage::
        async with OrganizationFixtures() as orgs:
            org_id = await orgs.create_org("smoke-push-log-a1b2c3d4")
            ...
            await orgs.delete_org(org_id)
    """

    async def create_org(self, name: str) -> str:
        """Insert a new organization and return its UUID (as text)."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO organizations (name) VALUES (%(name)s) RETURNING id::text",
                {"name": name},
            )
            row = await cur.fetchone()
        return row["id"]

    async def delete_org(self, org_id: str) -> None:
        """Delete an organization by id. No-op if it no longer exists."""
        async with self._pool.connection() as conn:
            await conn.execute(
                "DELETE FROM organizations WHERE id = %(id)s::uuid",
                {"id": org_id},
            )
```

- [ ] **Step 3: Smoke-check the import resolves**

Run from the repo root:

```bash
python3 -c "
import sys
sys.path.insert(0, 'e2e/tests/helpers')
import organizations
print(organizations.OrganizationFixtures)
"
```

Expected: prints `<class 'organizations.OrganizationFixtures'>` with no `ImportError`. This does
not touch the database (nothing is instantiated/connected) — it only confirms the module and its
own internal `sys.path` insert for `storage.base` resolve correctly.

- [ ] **Step 4: Commit**

```bash
git add e2e/__init__.py e2e/tests/__init__.py e2e/tests/helpers/__init__.py e2e/tests/helpers/organizations.py
git commit -m "feat(e2e): scaffold e2e package + organization fixture helper"
```

---

### Task 3: `api_keys.py` helper

**Files:**
- Create: `e2e/tests/helpers/api_keys.py`

**Interfaces:**
- Consumes: `storage.api_keys_storage.ApiKeysStorage.create_key(label: str, raw_key: str,
  organization_ids: list[str]) -> str` (unchanged, existing method).
- Produces: `async def issue_key(label: str, org_id: str) -> tuple[str, str]` (returns
  `(key_id, raw_key)`); `class ApiKeyFixtures(BaseStorage)` with `async def delete_key(self, label:
  str) -> int` and `async def count_key_scopes(self, key_id: str) -> int`.

- [ ] **Step 1: Write `e2e/tests/helpers/api_keys.py`**

```python
"""
api_keys.py — mint/delete throwaway API keys for e2e tests.

issue_key goes through ApiKeysStorage.create_key (the same path
scripts/create-api-key.py uses) rather than inserting rows directly, so a
test key is authorized exactly the way a real one would be.

delete_key removes the row outright (not ApiKeysStorage.revoke_key, which
only sets revoked_at): a real key's revocation history is worth keeping, but
a key minted here lives for seconds and leaves no audit trail worth
preserving — a dead revoked row on every run would just accumulate.

Deletion is by label, not by id: create_key inserts the key and its org
scope as two separate statements with no explicit transaction, so a failure
between them can leave an inserted key whose id the caller never received.
The label is always known up front (the caller generates it), so it is the
reliable handle for cleanup.
"""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from storage.api_keys_storage import ApiKeysStorage  # noqa: E402
from storage.base import BaseStorage  # noqa: E402


async def issue_key(label: str, org_id: str) -> tuple[str, str]:
    """Mint a key scoped to one organization. Returns (key_id, raw_key)."""
    raw_key = f"medkard_e2e_{secrets.token_urlsafe(24)}"
    async with ApiKeysStorage() as api_keys:
        key_id = await api_keys.create_key(label, raw_key, [org_id])
    return str(key_id), raw_key


class ApiKeyFixtures(BaseStorage):
    """Async context-manager for removing e2e-test API keys.

    Usage::
        async with ApiKeyFixtures() as keys:
            deleted = await keys.delete_key("smoke-push-log-a1b2c3d4")
    """

    async def delete_key(self, label: str) -> int:
        """Delete every key row with this label. Returns the row count deleted."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM api_keys WHERE label = %(label)s",
                {"label": label},
            )
            return cur.rowcount

    async def count_key_scopes(self, key_id: str) -> int:
        """Count remaining api_key_organizations rows for a key id.

        Used only to assert teardown actually cascaded — organizations does
        NOT cascade-delete api_key_organizations (nothing links a key to an
        org in that direction), so this is a sanity check, not a cleanup step.
        """
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT count(*) AS n FROM api_key_organizations WHERE api_key_id = %(id)s::uuid",
                {"id": key_id},
            )
            row = await cur.fetchone()
        return row["n"]
```

- [ ] **Step 2: Smoke-check the import resolves**

Run from the repo root:

```bash
python3 -c "
import sys
sys.path.insert(0, 'e2e/tests/helpers')
import api_keys
print(api_keys.issue_key, api_keys.ApiKeyFixtures)
"
```

Expected: prints the function and class objects, no `ImportError`.

- [ ] **Step 3: Commit**

```bash
git add e2e/tests/helpers/api_keys.py
git commit -m "feat(e2e): add API key fixture helper"
```

---

### Task 4: `cards.py` helper

**Files:**
- Create: `e2e/tests/helpers/cards.py`

**Interfaces:**
- Consumes: `httpx.AsyncClient` (caller-supplied, not created inside this module).
- Produces: `async def push_card(client: httpx.AsyncClient, base_url: str, org: str, raw_key: str,
  card: dict) -> httpx.Response`; `class CardFixtures(BaseStorage)` with `async def
  stage_audited(self, card_guid: str) -> None`, `async def card_row(self, card_guid: str) -> dict |
  None`, `async def push_log_rows(self, card_guid: str) -> list[dict]`, `async def
  delete_cards(self, card_guid: str) -> int`, `async def delete_push_log(self, card_guid: str) ->
  int`, `async def push_metrics_for_org_today(self, organization_name: str) -> dict | None`.

- [ ] **Step 1: Write `e2e/tests/helpers/cards.py`**

```python
"""
cards.py — push cards over HTTP and inspect/clean up done_cards + push_log
rows for e2e tests.

push_card is a thin wrapper over POST /visits/push — no retry, no auth
handling beyond passing the bearer token through, so a test's assertions
see the real HTTP response untouched.

stage_audited fabricates a completed formal-structure audit result directly
in the database (status='done' with a non-null formal_result), without
running any real LLM checker. This exists purely to put a done_cards row
into the state migration 027's push_log trigger needs to see in order to
log overrode_audit=true on the next push — it is not a substitute for
actually exercising the audit pipeline (scripts/smoke-cards-push.sh's
--with-audit flag does that, at the cost of real LLM calls).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from storage.base import BaseStorage  # noqa: E402


async def push_card(
    client: httpx.AsyncClient, base_url: str, org: str, raw_key: str, card: dict
) -> httpx.Response:
    """POST card to /visits/push?org=<org>, bearer-authenticated. Returns the raw response."""
    return await client.post(
        f"{base_url.rstrip('/')}/visits/push",
        params={"org": org},
        json=card,
        headers={"Authorization": f"Bearer {raw_key}"},
    )


class CardFixtures(BaseStorage):
    """Async context-manager for staging/inspecting/cleaning up e2e-test cards.

    Usage::
        async with CardFixtures() as cards:
            await cards.stage_audited(card_guid)
            row = await cards.card_row(card_guid)
            log = await cards.push_log_rows(card_guid)
            await cards.delete_push_log(card_guid)
            await cards.delete_cards(card_guid)
    """

    async def stage_audited(self, card_guid: str) -> None:
        """Mark an existing done_cards row as a completed formal-structure audit.

        Sets status='done' and a non-null formal_result (one fabricated
        finding), ignored=FALSE, broken=FALSE. The row must already exist
        (created by a prior push) — this only flips its state.
        """
        fake_formal_result = json.dumps(
            [{"flag": "e2e_fixture", "issue": "e2e fixture finding", "source": "", "comment": ""}],
            ensure_ascii=False,
        )
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                UPDATE done_cards
                SET status = 'done',
                    formal_result = %(formal)s::jsonb,
                    ignored = FALSE,
                    broken = FALSE
                WHERE card_guid = %(guid)s
                """,
                {"guid": card_guid, "formal": fake_formal_result},
            )

    async def card_row(self, card_guid: str) -> dict | None:
        """Return the full done_cards row for a guid, or None if it doesn't exist."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM done_cards WHERE card_guid = %(guid)s",
                {"guid": card_guid},
            )
            return await cur.fetchone()

    async def push_log_rows(self, card_guid: str) -> list[dict]:
        """Return every push_log row for a guid, oldest first."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM push_log WHERE card_guid = %(guid)s ORDER BY pushed_at",
                {"guid": card_guid},
            )
            return await cur.fetchall()

    async def delete_cards(self, card_guid: str) -> int:
        """Delete the done_cards row for a guid. Returns the row count deleted."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM done_cards WHERE card_guid = %(guid)s",
                {"guid": card_guid},
            )
            return cur.rowcount

    async def delete_push_log(self, card_guid: str) -> int:
        """Delete every push_log row for a guid. Returns the row count deleted."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM push_log WHERE card_guid = %(guid)s",
                {"guid": card_guid},
            )
            return cur.rowcount

    async def push_metrics_for_org_today(self, organization_name: str) -> dict | None:
        """Return today's push_metrics_by_date row for an organization, or None if absent.

        Keys: pushes_total, pushes_overrode_audit, pushes_no_override (matching
        the view's columns exactly — see migrations/027_audit_overwrite_journal.sql).
        """
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT pushes_total, pushes_overrode_audit, pushes_no_override "
                "FROM push_metrics_by_date "
                "WHERE organization_name = %(org)s AND push_date = current_date",
                {"org": organization_name},
            )
            return await cur.fetchone()
```

Note the card_guid case: `POST /visits/push` lowercases the guid before storing it (see
`api/routes/visits.py`'s `_extract_card_guid` / the existing smoke test's
`CARD_GUID.lower()` usage at `scripts/smoke-push-check-updates.py:184`). Every caller of
`CardFixtures` methods in Task 6 must pass the **lowercased** guid, matching what's actually
stored — this is the caller's responsibility, not something `cards.py` normalizes internally,
since `push_log_rows`/`card_row`/`delete_*` are generic lookups-by-exact-guid, not
push-specific.

- [ ] **Step 2: Smoke-check the import resolves**

Run from the repo root:

```bash
python3 -c "
import sys
sys.path.insert(0, 'e2e/tests/helpers')
import cards
print(cards.push_card, cards.CardFixtures)
"
```

Expected: prints the function and class objects, no `ImportError`.

- [ ] **Step 3: Commit**

```bash
git add e2e/tests/helpers/cards.py
git commit -m "feat(e2e): add card push/stage/inspect/cleanup fixture helper"
```

---

### Task 5: `test_push_log_smoke.py` — URL resolution + argument parsing + scaffolding

**Files:**
- Create: `e2e/tests/test_push_log_smoke.py` (this task writes the file's argument parsing, URL
  resolution, and `main()`/teardown skeleton; Task 6 fills in the test scenario body)

**Interfaces:**
- Consumes: nothing yet from Tasks 2-4 (those are wired in Task 6) — this task only needs
  `argparse`, `os`, `dotenv.load_dotenv`.
- Produces: a `_resolve_base_url(url_arg: str) -> str` function and a `check(label: str, condition:
  bool, detail: str = "") -> None` accumulator function that Task 6 calls into.

- [ ] **Step 1: Write the file's header, imports, URL resolution, and check() accumulator**

```python
#!/usr/bin/env python3
"""
End-to-end smoke test for push_log / push_metrics_by_date (migration 027).

Exercises the trigger that logs every POST /visits/push as a push_log row,
tagged overrode_audit=true when it destroys a completed audit result and
false otherwise, and the push_metrics_by_date view that aggregates those
rows per organization per day.

Scope is push_log/push_metrics_by_date only. General /visits/push behaviour
(auth rejection, 422s, card_data replacement) is already covered by
scripts/smoke-cards-push.sh and is not re-asserted here.

Creates a throwaway organization + API key + one card, pushes it through the
real HTTP API multiple times (staging a fake completed audit in between via
direct SQL, since running the real audit pipeline costs LLM tokens), and
reads push_log/push_metrics_by_date back to confirm the trigger and view
behave as migration 027 defines. Everything created is removed in a
`finally` block, including on assertion failure or Ctrl-C — unless --keep
is passed.

Run from the repo root against a running API:

    python e2e/tests/test_push_log_smoke.py local
    python e2e/tests/test_push_log_smoke.py https://medkard.example --keep

  local        resolves to http://localhost:{API_PORT}, API_PORT read from
               .env (default 8000 if unset — see .env.example)
  --keep       leave the org/key/card/push_log rows behind for manual
               inspection instead of tearing down; prints what was left
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
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

_parser = argparse.ArgumentParser(description="Smoke-test push_log/push_metrics_by_date against a running API")
_parser.add_argument(
    "url",
    help='Base URL of the API, e.g. http://localhost:8000, or "local"/"localhost" '
    "for http://localhost:{API_PORT}",
)
_parser.add_argument("--keep", action="store_true", help="Skip teardown and print what was left behind")
_args = _parser.parse_args()


def _resolve_base_url(url_arg: str) -> str:
    """"local"/"localhost" -> http://localhost:{API_PORT} (API_PORT from .env, default 8000).

    Any other value is used as-is, exactly like the existing smoke scripts'
    positional url argument.
    """
    if url_arg in ("local", "localhost"):
        port = os.environ.get("API_PORT", "8000")
        return f"http://localhost:{port}"
    return url_arg


BASE = _resolve_base_url(_args.url)
TAG = uuid.uuid4().hex[:8]
ORG_NAME = f"e2e-push-log-{TAG}"
KEY_LABEL = f"e2e-push-log-{TAG}"
CARD_GUID = f"e2e-push-log-{TAG}-{uuid.uuid4()}"

_PASS, _FAIL = "  \033[32mok\033[0m", "  \033[31mFAILED\033[0m"
_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record an assertion without aborting — teardown still has to run."""
    if condition:
        print(f"{_PASS}  {label}")
    else:
        print(f"{_FAIL}  {label}{(' — ' + detail) if detail else ''}")
        _failures.append(label)
```

This mirrors `scripts/smoke-push-check-updates.py`'s top-level structure (`TAG`, `_PASS`/`_FAIL`,
`check()`) exactly, so anyone already familiar with that script recognizes this one immediately.

- [ ] **Step 2: Add a placeholder `run()` and `main()` so the file is executable end-to-end already**

Append to the same file:

```python
async def run(client: httpx.AsyncClient, org_id: str, raw_key: str, card_fixtures: CardFixtures) -> None:
    """Filled in by the next task — the actual push_log test scenario."""
    print("\n(scenario not yet implemented)")


async def main() -> int:
    org_id: str | None = None

    print(f"Smoke test push_log/push_metrics_by_date against {BASE}")
    print(f"  org={ORG_NAME}  card_guid={CARD_GUID}")

    async with OrganizationFixtures() as org_fixtures, CardFixtures() as card_fixtures:
        try:
            org_id = await org_fixtures.create_org(ORG_NAME)
            key_id, raw_key = await issue_key(KEY_LABEL, org_id)
            print(f"  created org id={org_id}, key id={key_id}")

            async with httpx.AsyncClient(timeout=30) as client:
                await run(client, org_id, raw_key, card_fixtures)
        finally:
            if _args.keep:
                print(f"\n--keep: leaving org={ORG_NAME} (id={org_id}) card_guid={CARD_GUID.lower()}")
            else:
                print("\nCleaning up ...")
                deleted_log = await card_fixtures.delete_push_log(CARD_GUID.lower())
                print(f"  deleted {deleted_log} push_log row(s)")
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
```

`push_log` is deleted before `done_cards` in teardown deliberately: `push_log.card_guid` has no
foreign key back to `done_cards` (checked in migration 027 — `card_guid TEXT`, no `REFERENCES`), so
order doesn't matter for referential integrity here, but deleting the log rows first means a
`Ctrl-C` between the two deletes never leaves an orphaned `done_cards` row with no corresponding
explanation of why `push_log` still has entries for it.

- [ ] **Step 3: Make the file executable and do a syntax/import dry run**

```bash
chmod +x e2e/tests/test_push_log_smoke.py
python3 -c "import ast; ast.parse(open('e2e/tests/test_push_log_smoke.py').read())"
```

Expected: no `SyntaxError`. This only checks the file parses — it does not run `main()` (that
needs a live API + DB, which Task 6's test run provides).

- [ ] **Step 4: Commit**

```bash
git add e2e/tests/test_push_log_smoke.py
git commit -m "feat(e2e): scaffold push_log smoke test — url resolution, argparse, teardown"
```

---

### Task 6: `test_push_log_smoke.py` — the actual test scenario

**Files:**
- Modify: `e2e/tests/test_push_log_smoke.py` (replace the placeholder `run()` from Task 5)

**Interfaces:**
- Consumes: `push_card` (Task 4), `CardFixtures.stage_audited/card_row/push_log_rows/
  push_metrics_for_org_today` (Task 4), `check()` (Task 5).

- [ ] **Step 1: Replace the placeholder `run()` function**

Find (from Task 5):

```python
async def run(client: httpx.AsyncClient, org_id: str, raw_key: str, card_fixtures: CardFixtures) -> None:
    """Filled in by the next task — the actual push_log test scenario."""
    print("\n(scenario not yet implemented)")
```

Replace with:

```python
def _mock_card(version: int) -> dict:
    """Minimal card shaped like 1C's payload — push only requires Прием.GUID."""
    from datetime import datetime, timezone

    return {
        "Прием": {
            "GUID": CARD_GUID,
            "DATE": datetime.now(timezone.utc).strftime("%d.%m.%Y"),
            "TYPE": "Первичный",
        },
        "Пациент": {"Возраст": "42"},
        "e2e_tag": TAG,
        "e2e_version": version,
    }


async def run(client: httpx.AsyncClient, org_id: str, raw_key: str, card_fixtures: CardFixtures) -> None:
    guid = CARD_GUID.lower()  # POST /visits/push stores the guid lowercased

    print("\n1. First push (new card) — INSERT, not an overwrite, logs nothing")
    resp = await push_card(client, BASE, ORG_NAME, raw_key, _mock_card(1))
    check("first push accepted (200)", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")

    row = await card_fixtures.card_row(guid)
    check("card landed in done_cards", row is not None)
    if row is not None:
        check("status is 'pending' after first push", row["status"] == "pending", str(row["status"]))

    log_after_insert = await card_fixtures.push_log_rows(guid)
    check(
        "no push_log row from the initial INSERT",
        len(log_after_insert) == 0,
        f"found {len(log_after_insert)} row(s), expected 0 — the trigger is BEFORE UPDATE, not BEFORE INSERT",
    )

    print("\n2. Re-push the same (still-pending) card — logs overrode_audit=false")
    resp = await push_card(client, BASE, ORG_NAME, raw_key, _mock_card(2))
    check("second push accepted (200)", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")

    log_after_repush = await card_fixtures.push_log_rows(guid)
    check(
        "exactly one push_log row after the pending re-push",
        len(log_after_repush) == 1,
        f"found {len(log_after_repush)} row(s)",
    )
    if log_after_repush:
        check(
            "that row has overrode_audit=false",
            log_after_repush[0]["overrode_audit"] is False,
            f"overrode_audit={log_after_repush[0]['overrode_audit']}",
        )

    print("\n3. Stage a fake completed audit, then push over it — logs overrode_audit=true")
    await card_fixtures.stage_audited(guid)
    staged = await card_fixtures.card_row(guid)
    check(
        "card is staged as done with a formal_result",
        staged is not None and staged["status"] == "done" and staged["formal_result"] is not None,
        f"row={staged}",
    )

    resp = await push_card(client, BASE, ORG_NAME, raw_key, _mock_card(3))
    check("third push accepted (200)", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")

    row_after_override = await card_fixtures.card_row(guid)
    if row_after_override is not None:
        check(
            "audit columns wiped again after the override push",
            row_after_override["status"] == "pending" and row_after_override["formal_result"] is None,
            f"status={row_after_override['status']} formal_result={row_after_override['formal_result']}",
        )

    log_after_override = await card_fixtures.push_log_rows(guid)
    check(
        "exactly two push_log rows total",
        len(log_after_override) == 2,
        f"found {len(log_after_override)} row(s)",
    )
    if len(log_after_override) == 2:
        check(
            "second row has overrode_audit=true",
            log_after_override[1]["overrode_audit"] is True,
            f"overrode_audit={log_after_override[1]['overrode_audit']}",
        )

    print("\n4. push_metrics_by_date reflects exactly these two pushes for today")
    metrics = await card_fixtures.push_metrics_for_org_today(ORG_NAME)

    check("push_metrics_by_date has a row for this org/today", metrics is not None, str(metrics))
    if metrics is not None:
        check("pushes_total == 2", metrics["pushes_total"] == 2, f"got {metrics['pushes_total']}")
        check(
            "pushes_overrode_audit == 1", metrics["pushes_overrode_audit"] == 1,
            f"got {metrics['pushes_overrode_audit']}",
        )
        check(
            "pushes_no_override == 1", metrics["pushes_no_override"] == 1,
            f"got {metrics['pushes_no_override']}",
        )
        check(
            "total == overrode + no_override",
            metrics["pushes_total"] == metrics["pushes_overrode_audit"] + metrics["pushes_no_override"],
            str(metrics),
        )
```

This scenario uses an org created fresh for this run (`ORG_NAME` is unique per run via `TAG`), so
`push_metrics_by_date WHERE organization_name = ORG_NAME` can assert absolute counts (`== 2`, not
a before/after delta) — unlike `tests/test_push_log.py`'s integration tests, which share the
`"Alenka"` fixture org across concurrent test runs and must use deltas. A brand-new org has no
pre-existing rows to collide with.

- [ ] **Step 2: Run the smoke test against a live API**

This step requires the pull API running and reachable, and the `.env`-configured Postgres
reachable with migration 027 already applied (it is, per Tasks 2-7 of the prior push_log plan on
this same branch lineage). Start the API however your environment normally runs it — for example:

```bash
cd /home/okabe/projects/medkard/.worktrees/push-log-e2e
PYTHONPATH=src /home/okabe/projects/medkard/.venv/bin/python3 -m uvicorn api.app:create_app --factory --port 8000 &
sleep 2
```

Then run the smoke test:

```bash
python3 e2e/tests/test_push_log_smoke.py local
```

Expected: every check prints `ok`, ending with `All checks passed.` and exit code 0. Run `echo
$?` afterward to confirm.

If you started a background `uvicorn` for this step, stop it afterward:

```bash
kill %1
```

- [ ] **Step 3: Run once more with `--keep` and manually verify, then clean up by hand**

```bash
python3 e2e/tests/test_push_log_smoke.py local --keep
```

Expected: same checks pass, but the final lines show `--keep: leaving org=... card_guid=...`
instead of the cleanup log. Manually confirm the org/card/push_log rows are still present (e.g.
via `psql` or the `card_row`/`push_log_rows` helpers in a one-off `python3 -c` snippet), then
delete them by hand so the run doesn't leave permanent test data:

```bash
python3 -c "
import asyncio, sys
sys.path.insert(0, 'e2e/tests/helpers')
from organizations import OrganizationFixtures
from api_keys import ApiKeyFixtures
from cards import CardFixtures

async def main():
    # Replace with the actual TAG printed by the --keep run above.
    tag = 'REPLACE_ME'
    guid = f'e2e-push-log-{tag}'.lower()  # adjust to match the printed card_guid exactly
    async with CardFixtures() as cards, ApiKeyFixtures() as keys, OrganizationFixtures() as orgs:
        # look up the org id by name if needed, then delete_push_log/delete_cards/delete_key/delete_org
        pass

asyncio.run(main())
"
```

(This manual cleanup snippet is illustrative — the exact commands depend on the tag printed by
your `--keep` run; the point of this step is confirming `--keep` actually leaves inspectable state
behind, not scripting its cleanup.)

- [ ] **Step 4: Commit**

```bash
git add e2e/tests/test_push_log_smoke.py
git commit -m "feat(e2e): implement push_log smoke test scenario"
```

---

### Task 7: Final regression pass

**Files:** none (verification only)

**Interfaces:** none

- [ ] **Step 1: Confirm no other repo test suite is affected**

Run: `pytest -q`

Expected: the same pass/fail counts as the baseline already established on this branch lineage
(306 passed, 7 pre-existing failed, 1 pre-existing error — none related to `e2e/`, since
`pytest.ini` does not point at `e2e/` and nothing under `e2e/` uses `pytest` conventions
`test_*` functions are picked up by pytest's default discovery only within its configured
rootdir/testpaths; if `pytest -q` unexpectedly tries to collect `e2e/tests/test_push_log_smoke.py`
as a pytest test file (possible since the filename matches pytest's default `test_*.py` pattern)
and it errors on the module-level `_parser.parse_args()` call (since no CLI args are passed under
pytest), add `norecursedirs = e2e` to the `[pytest]` section of `pytest.ini`.

- [ ] **Step 2: If a `norecursedirs` addition was needed, apply it and re-run**

If Step 1 showed `e2e/tests/test_push_log_smoke.py` being collected/erroring under plain `pytest
-q`, modify `pytest.ini`:

Find:
```ini
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = session
asyncio_default_test_loop_scope = session
pythonpath = src
```

Replace with:
```ini
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = session
asyncio_default_test_loop_scope = session
pythonpath = src
norecursedirs = e2e
```

Then re-run `pytest -q` and confirm the count returns to the established baseline with no
`e2e/`-related collection errors.

- [ ] **Step 3: Commit, only if Step 2 was needed**

```bash
git add pytest.ini
git commit -m "fix(pytest): exclude e2e/ from unit-test collection"
```

If Step 1 showed no interference (pytest's default rootdir-based discovery already didn't reach
into `e2e/`, or did but caused no error), skip this commit — there's nothing to change.

---

## Self-Review Notes

- **Spec coverage:** `.env`/`docker-compose.yml` API_PORT (Task 1) ✓, `e2e/` package scaffolding
  (Task 2) ✓, all three helper modules — organizations/api_keys/cards — matching the spec's exact
  function signatures (Tasks 2-4) ✓, `local`→`http://localhost:{API_PORT}` resolution (Task 5) ✓,
  `--keep` contract (Task 5) ✓, all 7 scenario steps from the spec (push new card → no log; re-push
  pending → overrode_audit=false; stage_audited → push over it → overrode_audit=true;
  push_metrics_by_date totals; teardown) (Task 6) ✓. "Не в скоупе" items (server auto-start,
  migrating old smoke scripts, pytest wrapping, re-testing generic /visits/push behavior) —
  correctly excluded, no task attempts any of them ✓.
- **Type/name consistency checked:** `OrganizationFixtures`/`ApiKeyFixtures`/`CardFixtures` class
  names and their method signatures are used identically between their defining tasks (2-4) and
  their consumption in Task 5/6's `test_push_log_smoke.py` (`org_fixtures.create_org`,
  `key_fixtures.delete_key`, `card_fixtures.stage_audited`/`card_row`/`push_log_rows`/
  `delete_cards`/`delete_push_log`/`push_metrics_for_org_today`, `issue_key`, `push_card`) —
  verified matching across all tasks. `API_PORT` used identically in `.env.example` (Task 1) and
  `_resolve_base_url` (Task 5). Task 6's scenario reads the metrics view through
  `CardFixtures.push_metrics_for_org_today` rather than reaching into `card_fixtures._pool`
  directly, keeping all DB access behind the fixture classes' public methods.
- **Card GUID lowercasing** is called out explicitly in both Task 4 (helper docstring note) and
  Task 6 (scenario uses `CARD_GUID.lower()` throughout) to avoid a lookup mismatch bug, matching
  the existing `smoke-push-check-updates.py` pattern this plan follows.
