# E2E Full Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add e2e coverage for the five remaining `/visits/*` routes and `/stats/storage` (standalone
scripts against a live API + Postgres), audit e2e fixtures covering all 5 `visit_type`s × 2
`age_group`s (10 fixtures in 5 files, grouped by `visit_type`, using real LLM calls, where each
fixture's NMU code drives `FormalValidator`'s own classification rather than being force-fed), and a
written e2e methodology doc.

**Architecture:** Route tests extend the existing `e2e/tests/helpers/cards.py` with two new fixture
methods (`stage_done_with_meta`, `mark_ignored`) and otherwise copy `test_push_log_smoke.py`'s exact
skeleton (argparse `url`/`--keep`, `check()` accumulator, `finally` teardown). Audit tests live in a
new `e2e/tests/audit/` subpackage: a shared `fixtures.py` (`dx()`/`base_visit()` builders) and a
shared `harness.py` (`Case`/`run_cases()` — a two-stage runner: stage 1 confirms
`FormalValidator.get_visit_types()`/`get_rules()` classify and select the target rule deterministically,
before any LLM call; stage 2 runs the real `AuditPipeline._audit_visit()` and asserts the **complete**
set of formal flags equals exactly the one flag each fixture's single deliberate defect targets, plus
a log-watching guard against a silently unparsed LLM response). This harness pattern is adopted from
`e2e/tests/audit/harness.py` on the sibling branch `formal-rules-npa-revision`, which branched off the
same base commit and independently solved the same problem for its own (differently-scoped) fixture
set — see `docs/superpowers/specs/2026-08-20-e2e-full-suite-design.md` §2 for the full rationale.
`AuditPipeline._audit_visit()` is called directly, without `async with` — no HTTP, no DB writes to
clean up, since `_upsert_done_card` no-ops while `self._done_cards is None`. The methodology doc is a
new top-level `docs/e2e-testing.md`.

**Tech Stack:** Python 3.11, `httpx.AsyncClient`, `psycopg3` via `BaseStorage`, `python-dotenv` — no
new dependencies. All work happens in the existing worktree at
`/home/okabe/projects/medkard/.worktrees/push-log-e2e` on branch `push-log-e2e-tests`.

---

## Global Constraints

- Every new script is a standalone file under `e2e/tests/` (route tests) or `e2e/tests/audit/` (audit
  tests) — no `pytest.mark`, no pytest fixtures. Route tests take `argparse` (positional `url`, default
  `"local"`, and `--keep`); audit tests take no arguments at all (no network, no `--keep` — nothing to
  keep, since `AuditPipeline._audit_visit()` used without `async with` persists nothing — see Task 9).
- Route tests use the `check(label, condition, detail)` accumulator + `_PASS`/`_FAIL` pattern from
  `e2e/tests/test_push_log_smoke.py` verbatim — copy the top-of-file block (imports, `_resolve_base_url`,
  `TAG`, `check()`) into each new file, adjusting only the org/key/card-guid prefix string.
- Every fixture a route test creates (org, key, card, push_log rows) is deleted in a `finally` block
  that runs on assertion failure or `KeyboardInterrupt`, unless `--keep` is passed — matching
  `test_push_log_smoke.py`'s existing contract.
- Card GUIDs, organization names, and API key labels are namespaced with `TAG = uuid.uuid4().hex[:8]`
  so concurrent runs on the shared Postgres never collide.
- Audit test fixtures use ICD code `J06.9` (guideline `306_3`, age_category `{Взрослые,дети}` —
  confirmed present in the live `guidelines`/`docs` tables) so every fixture exercises
  `DiagnosisValidator` against a real, already-ingested guideline, except the `primary`/child fixture
  which deliberately omits `Диагнозы` (see Task 10) and the `prophylactic`/child fixture (Task 12),
  which uses `Z00.1` instead — J06.9 on a prophylactic child visit was found empirically to trip
  `НЕСООТВЕТСТВИЕ_УСЛУГИ_И_ВИЗИТА` (an acute-infection code on a routine visit with no treatment reads
  as a type/diagnosis mismatch to the model), and `Z00.1` is one of `ClinicRecs._SKIP_CODES`, so
  `DiagnosisValidator` correctly returns "no guideline found" for that one fixture rather than
  exercising a real guideline lookup — a known, narrow coverage gap, not an oversight.
- Audit test assertions check that the **complete** set of formal flags equals exactly `{case.expect}`
  — not presence-only. Every fixture must therefore be otherwise flawless against every rule
  applicable to its `visit_type`/`age_group`, including the `"visit_types": ["all"]` rules
  (`ОТСУТСТВУЮТ_МЕТАДАННЫЕ_ВИЗИТА`, `ОБНАРУЖЕНЫ_ЗАГЛУШКИ`, `ДУБЛИРОВАНИЕ_СМЫСЛОВЫХ_БЛОКОВ`,
  `has_typos`) and, for every child fixture, `legal_representative_info` (needs a named escort in the
  exam narrative, e.g. "Осмотр проведён в сопровождении матери").
- This plan does not touch `pytest.ini`, `.env.example`, or `docker-compose.yml` — those are already
  correct from the prior `push_log_smoke` plan on this branch.

---

### Task 1: `CardFixtures.stage_done_with_meta` helper

**Files:**
- Modify: `e2e/tests/helpers/cards.py`

**Interfaces:**
- Produces: `async def stage_done_with_meta(self, card_guid: str, *, visit_date: str, doctor_code: str
  | None = None, doctor_name: str | None = None) -> None` on `CardFixtures`.

- [ ] **Step 1: Read the current file to confirm the class layout before editing**

Run: `sed -n '1,80p' e2e/tests/helpers/cards.py`

Confirm `CardFixtures.stage_audited` is present (added by the prior `push_log_smoke` plan) — the new
method goes directly after it.

- [ ] **Step 2: Add `stage_done_with_meta` to `CardFixtures`**

In `e2e/tests/helpers/cards.py`, find the end of `stage_audited` (the method ends right before
`async def card_row`). Insert this new method between `stage_audited` and `card_row`:

```python
    async def stage_done_with_meta(
        self,
        card_guid: str,
        *,
        visit_date: str,
        doctor_code: str | None = None,
        doctor_name: str | None = None,
    ) -> None:
        """Mark an existing done_cards row as done, with a controllable Прием.DATE
        (and optionally Прием.Врач_код/Врач), for routes that filter or group by
        those fields (check/pull/export/doctors).

        Unlike stage_audited (which only flips status/audit columns for the
        push_log override scenario), this also rewrites card_data's Прием block
        in place — visit_date must be DD.MM.YYYY, matching what 1C sends and what
        reporting.api_formatter's medkard_visit_date() parses.
        """
        fake_formal_result = json.dumps(
            [{"flag": "e2e_fixture", "issue": "e2e fixture finding", "source": "", "comment": ""}],
            ensure_ascii=False,
        )
        priem_patch: dict = {"DATE": visit_date}
        if doctor_code is not None:
            priem_patch["Врач_код"] = doctor_code
        if doctor_name is not None:
            priem_patch["Врач"] = doctor_name

        async with self._pool.connection() as conn:
            await conn.execute(
                """
                UPDATE done_cards
                SET status = 'done',
                    formal_result = %(formal)s::jsonb,
                    ignored = FALSE,
                    broken = FALSE,
                    card_data = jsonb_set(
                        card_data,
                        '{Прием}',
                        COALESCE(card_data -> 'Прием', '{}'::jsonb) || %(priem)s::jsonb
                    )
                WHERE card_guid = %(guid)s
                """,
                {"guid": card_guid, "formal": fake_formal_result, "priem": json.dumps(priem_patch, ensure_ascii=False)},
            )
```

- [ ] **Step 3: Smoke-check the import still resolves**

Run from the repo root:

```bash
python3 -c "
import sys
sys.path.insert(0, 'e2e/tests/helpers')
import cards
print(cards.CardFixtures.stage_done_with_meta)
"
```

Expected: prints the function object, no `ImportError`/`SyntaxError`.

- [ ] **Step 4: Commit**

```bash
git add e2e/tests/helpers/cards.py
git commit -m "feat(e2e): add stage_done_with_meta fixture helper for route tests"
```

---

### Task 2: `test_visits_check_smoke.py`

**Files:**
- Create: `e2e/tests/test_visits_check_smoke.py`

**Interfaces:**
- Consumes: `OrganizationFixtures`, `ApiKeyFixtures`/`issue_key`, `CardFixtures` (incl.
  `stage_done_with_meta` from Task 1), `push_card` — all from `e2e/tests/helpers/*`.

- [ ] **Step 1: Write the full file**

```python
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
```

- [ ] **Step 2: Make it executable and syntax-check**

```bash
chmod +x e2e/tests/test_visits_check_smoke.py
python3 -c "import ast; ast.parse(open('e2e/tests/test_visits_check_smoke.py').read())"
```

Expected: no `SyntaxError`.

- [ ] **Step 3: Run against a live API**

Requires the pull API reachable (see Task 5's Step 2 in the prior `push_log_smoke` plan for how to
start it locally) and `.env`-configured Postgres with migrations applied.

```bash
python3 e2e/tests/test_visits_check_smoke.py local
```

Expected: every check prints `ok`, ending with `All checks passed.`, exit code 0 (`echo $?`).

- [ ] **Step 4: Commit**

```bash
git add e2e/tests/test_visits_check_smoke.py
git commit -m "feat(e2e): add GET /visits/check smoke test"
```

---

### Task 3: `test_visits_pull_smoke.py`

**Files:**
- Create: `e2e/tests/test_visits_pull_smoke.py`

**Interfaces:**
- Consumes: same helpers as Task 2.

- [ ] **Step 1: Write the full file**

Base it on Task 2's file with these differences: rename `ORG_NAME`/`KEY_LABEL`/`CARD_GUID` prefixes
to `e2e-pull-`, and replace the `run()` body with:

```python
async def run(client: httpx.AsyncClient, raw_key: str, card_fixtures: CardFixtures) -> None:
    guid = CARD_GUID.lower()
    today = datetime.now(timezone.utc).strftime("%d.%m.%Y")
    today_iso = datetime.now(timezone.utc).date().isoformat()
    yesterday_iso = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()

    print("\n1. Push and stage the card as done, dated today")
    resp = await push_card(client, BASE, ORG_NAME, raw_key, _mock_card())
    check("push accepted (200)", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")
    await card_fixtures.stage_done_with_meta(guid, visit_date=today)

    print("\n2. pull?date=today returns an xlsx workbook")
    resp = await client.get(
        f"{BASE.rstrip('/')}/visits/pull",
        params={"org": ORG_NAME, "date": today_iso},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    check("pull today accepted (200)", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")
    check(
        "content-type is xlsx",
        resp.headers.get("content-type", "") == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        f"got {resp.headers.get('content-type')}",
    )
    disposition = resp.headers.get("content-disposition", "")
    check(
        "content-disposition names the report file",
        f"report_{ORG_NAME}_{today_iso}.xlsx" in disposition,
        f"got {disposition!r}",
    )
    check("body is non-empty", len(resp.content) > 0)

    print("\n3. pull?date=yesterday (no cards, no doctor_code) -> 404")
    resp = await client.get(
        f"{BASE.rstrip('/')}/visits/pull",
        params={"org": ORG_NAME, "date": yesterday_iso},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    check("empty day without doctor_code -> 404", resp.status_code == 404, f"got {resp.status_code}: {resp.text[:200]}")

    print("\n4. pull?date=yesterday&doctor_code=... (no cards, doctor_code given) -> 200 with empty-report xlsx")
    resp = await client.get(
        f"{BASE.rstrip('/')}/visits/pull",
        params={"org": ORG_NAME, "date": yesterday_iso, "doctor_code": "e2e-doc-1"},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    check(
        "empty day with doctor_code -> 200, not 404",
        resp.status_code == 200,
        f"got {resp.status_code}: {resp.text[:200]}",
    )
    check("empty-report body is non-empty", len(resp.content) > 0)
```

The rest of the file (header, argparse, `TAG`/`ORG_NAME`/etc. constants, `_mock_card`, `main()`,
teardown, `__main__` guard) is identical to Task 2's file, with `check` → `pull` in the docstring and
`e2e-pull-{TAG}` name prefixes.

- [ ] **Step 2: Make it executable and syntax-check**

```bash
chmod +x e2e/tests/test_visits_pull_smoke.py
python3 -c "import ast; ast.parse(open('e2e/tests/test_visits_pull_smoke.py').read())"
```

- [ ] **Step 3: Run against a live API**

```bash
python3 e2e/tests/test_visits_pull_smoke.py local
```

Expected: all checks `ok`, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add e2e/tests/test_visits_pull_smoke.py
git commit -m "feat(e2e): add GET /visits/pull smoke test"
```

---

### Task 4: `test_visits_export_smoke.py`

**Files:**
- Create: `e2e/tests/test_visits_export_smoke.py`

**Interfaces:**
- Consumes: same helpers as Task 2, plus a new `CardFixtures.mark_ignored` method added in Step 1
  below (needed for the `include_ignored` check — no existing helper sets `ignored=TRUE`).

- [ ] **Step 1: Add `CardFixtures.mark_ignored`**

In `e2e/tests/helpers/cards.py`, add this method directly after `stage_done_with_meta` (from Task 1):

```python
    async def mark_ignored(self, card_guid: str) -> None:
        """Flip an existing done_cards row to ignored=TRUE, status='done'.

        Exists to test /visits/export's include_ignored toggle — no other
        fixture path produces an ignored row (the real pipeline sets it via
        audit/filters.py, which this e2e suite does not invoke).
        """
        async with self._pool.connection() as conn:
            await conn.execute(
                "UPDATE done_cards SET status = 'done', ignored = TRUE WHERE card_guid = %(guid)s",
                {"guid": card_guid},
            )
```

- [ ] **Step 2: Smoke-check the import**

```bash
python3 -c "
import sys
sys.path.insert(0, 'e2e/tests/helpers')
import cards
print(cards.CardFixtures.mark_ignored)
"
```

- [ ] **Step 3: Commit the helper addition**

```bash
git add e2e/tests/helpers/cards.py
git commit -m "feat(e2e): add mark_ignored fixture helper for export smoke test"
```

- [ ] **Step 4: Write `e2e/tests/test_visits_export_smoke.py`**

Base it on Task 2's skeleton (header/argparse/constants/`main()`/teardown identical, `e2e-export-`
prefix), using **two** card GUIDs instead of one:

```python
CARD_GUID_A = f"e2e-export-{TAG}-a-{uuid.uuid4()}"
CARD_GUID_B = f"e2e-export-{TAG}-b-{uuid.uuid4()}"
```

(Replace the single `CARD_GUID` constant with these two everywhere it's used — in `_mock_card`,
`run`, and teardown's `delete_cards` calls, which now run twice.)

```python
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
```

Teardown in `main()` must delete both cards:

```python
                deleted_a = await card_fixtures.delete_cards(CARD_GUID_A.lower())
                deleted_b = await card_fixtures.delete_cards(CARD_GUID_B.lower())
                print(f"  deleted {deleted_a + deleted_b} done_cards row(s)")
```

And the startup print line:

```python
    print(f"  org={ORG_NAME}  card_guids={CARD_GUID_A}, {CARD_GUID_B}")
```

And the `--keep` message:

```python
                print(f"\n--keep: leaving org={ORG_NAME} (id={org_id}) card_guids={CARD_GUID_A.lower()}, {CARD_GUID_B.lower()}")
```

- [ ] **Step 5: Make it executable and syntax-check**

```bash
chmod +x e2e/tests/test_visits_export_smoke.py
python3 -c "import ast; ast.parse(open('e2e/tests/test_visits_export_smoke.py').read())"
```

- [ ] **Step 6: Run against a live API**

```bash
python3 e2e/tests/test_visits_export_smoke.py local
```

Expected: all checks `ok`, exit code 0.

- [ ] **Step 7: Commit**

```bash
git add e2e/tests/test_visits_export_smoke.py
git commit -m "feat(e2e): add GET /visits/export smoke test"
```

---

### Task 5: `test_visits_check_updates_smoke.py`

**Files:**
- Create: `e2e/tests/test_visits_check_updates_smoke.py`

**Interfaces:**
- Consumes: `OrganizationFixtures`, `ApiKeyFixtures`/`issue_key`, `CardFixtures.card_row`/
  `delete_cards`, `push_card`. No `stage_done_with_meta` — this route deliberately checks a
  still-`pending` card.

- [ ] **Step 1: Write the full file**

Base it on Task 2's skeleton (`e2e-checkupd-` prefix). Replace `run()`:

```python
async def run(client: httpx.AsyncClient, raw_key: str, card_fixtures: CardFixtures) -> None:
    guid = CARD_GUID.lower()
    before_push = datetime.now(timezone.utc).isoformat()

    print("\n1. Push a card, leave it pending (no staging)")
    resp = await push_card(client, BASE, ORG_NAME, raw_key, _mock_card())
    check("push accepted (200)", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")

    row = await card_fixtures.card_row(guid)
    check("card is pending after push", row is not None and row["status"] == "pending", f"row={row}")

    after_push = datetime.now(timezone.utc).isoformat()

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
```

`_mock_card`, header, argparse, constants, teardown, `main()` are identical to Task 2's file with the
`e2e-checkupd-` prefix (no `stage_done_with_meta` import needed — remove that from imports since it's
unused here; keep `card_row`).

- [ ] **Step 2: Make it executable and syntax-check**

```bash
chmod +x e2e/tests/test_visits_check_updates_smoke.py
python3 -c "import ast; ast.parse(open('e2e/tests/test_visits_check_updates_smoke.py').read())"
```

- [ ] **Step 3: Run against a live API**

```bash
python3 e2e/tests/test_visits_check_updates_smoke.py local
```

Expected: all checks `ok`, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add e2e/tests/test_visits_check_updates_smoke.py
git commit -m "feat(e2e): add GET /visits/check_updates smoke test"
```

---

### Task 6: `test_visits_doctors_smoke.py`

**Files:**
- Create: `e2e/tests/test_visits_doctors_smoke.py`

**Interfaces:**
- Consumes: same helpers as Task 2, using `stage_done_with_meta(doctor_code=..., doctor_name=...)`
  from Task 1.

- [ ] **Step 1: Write the full file**

Base on Task 2's skeleton (`e2e-doctors-` prefix), three card GUIDs:

```python
CARD_GUID_A = f"e2e-doctors-{TAG}-a-{uuid.uuid4()}"
CARD_GUID_B = f"e2e-doctors-{TAG}-b-{uuid.uuid4()}"
CARD_GUID_C = f"e2e-doctors-{TAG}-c-{uuid.uuid4()}"
```

```python
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
```

Teardown deletes all three cards (same pattern as Task 4's two-card teardown, extended to three).
`main()`/header/argparse/constants otherwise identical to Task 2 with `e2e-doctors-` prefix.

- [ ] **Step 2: Make it executable and syntax-check**

```bash
chmod +x e2e/tests/test_visits_doctors_smoke.py
python3 -c "import ast; ast.parse(open('e2e/tests/test_visits_doctors_smoke.py').read())"
```

- [ ] **Step 3: Run against a live API**

```bash
python3 e2e/tests/test_visits_doctors_smoke.py local
```

Expected: all checks `ok`, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add e2e/tests/test_visits_doctors_smoke.py
git commit -m "feat(e2e): add GET /visits/doctors smoke test"
```

---

### Task 7: `test_stats_storage_smoke.py`

**Files:**
- Create: `e2e/tests/test_stats_storage_smoke.py`

**Interfaces:**
- Consumes: same helpers as Task 2.

- [ ] **Step 1: Write the full file**

Base on Task 2's skeleton (`e2e-stats-` prefix). Replace `run()`:

```python
async def run(client: httpx.AsyncClient, raw_key: str, card_fixtures: CardFixtures) -> None:
    guid = CARD_GUID.lower()
    today = datetime.now(timezone.utc).strftime("%d.%m.%Y")

    print("\n1. Baseline storage stats for a freshly created org")
    resp = await client.get(
        f"{BASE.rstrip('/')}/stats/storage",
        params={"org": ORG_NAME},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    check("baseline stats accepted (200)", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")
    baseline = resp.json() if resp.status_code == 200 else {}
    check("baseline done_cards_kb == 0", baseline.get("done_cards_kb") == 0, f"got {baseline}")
    check("baseline push_log_kb == 0", baseline.get("push_log_kb") == 0, f"got {baseline}")
    check("baseline total_kb == 0", baseline.get("total_kb") == 0, f"got {baseline}")

    print("\n2. Push and stage one card")
    resp = await push_card(client, BASE, ORG_NAME, raw_key, _mock_card())
    check("push accepted (200)", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")
    await card_fixtures.stage_done_with_meta(guid, visit_date=today)

    print("\n3. Storage stats now reflect the card")
    resp = await client.get(
        f"{BASE.rstrip('/')}/stats/storage",
        params={"org": ORG_NAME},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    check("stats accepted (200)", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")
    stats = resp.json() if resp.status_code == 200 else {}
    check("done_cards_kb > 0", stats.get("done_cards_kb", 0) > 0, f"got {stats}")
    check(
        "total_kb == done_cards_kb + push_log_kb",
        stats.get("total_kb") == stats.get("done_cards_kb", 0) + stats.get("push_log_kb", 0),
        f"got {stats}",
    )
    check("organization field matches", stats.get("organization") == ORG_NAME, f"got {stats}")
```

`_mock_card`, header, argparse, constants, teardown, `main()` identical to Task 2 with `e2e-stats-`
prefix.

- [ ] **Step 2: Make it executable and syntax-check**

```bash
chmod +x e2e/tests/test_stats_storage_smoke.py
python3 -c "import ast; ast.parse(open('e2e/tests/test_stats_storage_smoke.py').read())"
```

- [ ] **Step 3: Run against a live API**

```bash
python3 e2e/tests/test_stats_storage_smoke.py local
```

Expected: all checks `ok`, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add e2e/tests/test_stats_storage_smoke.py
git commit -m "feat(e2e): add GET /stats/storage smoke test"
```

---

### Task 8: `e2e/tests/audit/` package scaffolding + `fixtures.py`

**Files:**
- Create: `e2e/tests/audit/__init__.py`
- Create: `e2e/tests/audit/fixtures.py`

**Interfaces:**
- Produces: `def dx(code: str, name: str, *, detail: str = "", first_time: bool = False) -> dict`,
  `def base_visit(*, guid: str, service_code: str, service_name: str, specialty: str, age: int,
  inspection: list[tuple[str, str]], diagnoses: list[dict], gender: str = "Женский", visit_date: str =
  "20.08.2026") -> dict`.

- [ ] **Step 1: Create the empty `__init__.py`**

```bash
mkdir -p e2e/tests/audit
touch e2e/tests/audit/__init__.py
```

- [ ] **Step 2: Write `e2e/tests/audit/fixtures.py`**

```python
"""
fixtures.py — building blocks for audit e2e fixture cards.

Every fixture card is a visit dict shaped like 1C's payload
(docs/clinic-data-requirements.md §3) and carries exactly one defect.
Everything else in the card must be flawless, because the audit harness
(harness.py) asserts the complete set of formal findings, not just the
presence of the expected one — see docs/e2e-testing.md for why.

Practical consequence when writing a new fixture: the card has to satisfy
every rule that FormalValidator.get_rules() hands to the prompt for its
visit type and age group, not only the rule under test. The rules that
apply to *all* visit types are the easy ones to trip by accident:

  ОТСУТСТВУЮТ_МЕТАДАННЫЕ_ВИЗИТА   — date, age, sex, service must be present
  ОБНАРУЖЕНЫ_ЗАГЛУШКИ             — no "-", "уточнить" stand-ins outside the target field
  ДУБЛИРОВАНИЕ_СМЫСЛОВЫХ_БЛОКОВ   — do not repeat a block under two Параметр's
  ОРФОГРАФИЧЕСКИЕ_ОШИБКИ          — proofread the Russian
  ОТСУТСТВУЕТ_ИНФОРМАЦИЯ_О_СОПРОВОЖДАЮЩЕМ — children need a named legal representative
                                     mentioned in the exam narrative

Dates are fixed, never datetime.now(): fixtures are pure data, nothing is
written to the database, and a stable date keeps runs reproducible and log
lines greppable.
"""

from __future__ import annotations

from typing import Any


def dx(
    code: str,
    name: str,
    *,
    detail: str = "",
    first_time: bool = False,
) -> dict[str, Any]:
    """One entry for `Диагнозы`.

    `code` drives both DiagnosisValidator (guideline lookup) and, for
    Z11.1, the PROPHYLACTIC_TUBERCULIN branch of get_visit_types.
    """
    return {
        "КодМКБ": code,
        "НаименованиеМКБ": name,
        "Детализация": detail,
        "ВыявленВпервые": first_time,
    }


def base_visit(
    *,
    guid: str,
    service_code: str,
    service_name: str,
    specialty: str,
    age: int,
    inspection: list[tuple[str, str]],
    diagnoses: list[dict[str, Any]],
    gender: str = "Женский",
    visit_date: str = "20.08.2026",
) -> dict[str, Any]:
    """Assemble a complete visit card.

    `inspection` is a list of (Параметр, Значение) pairs — kept as tuples at
    the call site so a fixture reads as a medical record rather than as
    JSON. `service_code` goes into `КодЕГИСЗ`; it is what get_visit_types
    classifies, so the visit type is *derived by the system*, never
    declared by the test. Pass an empty string to leave the service
    unclassified (used by the tuberculin fixtures, whose type comes from
    the Z11.1 diagnosis alone).

    `Пациент.AGE` is written as an int under the key the validator reads —
    validator.py looks at `AGE` only, no fallback to `Возраст`.
    """
    return {
        "Прием": {
            "GUID": guid,
            "NUM": guid.rsplit("-", 1)[-1],
            "DATE": visit_date,
            "Врач_код": "00042",
            "Врач": "Иванова Анна Сергеевна",
        },
        "Врач": {"SPECIALIZATION": specialty},
        "Пациент": {"CODE": "P-000001", "GENDER": gender, "AGE": age},
        "Услуги": [{"КодЕГИСЗ": service_code, "Наименование": service_name}],
        "ДанныеОсмотра": [
            {"Параметр": param, "Значение": value} for param, value in inspection
        ],
        "Диагнозы": diagnoses,
    }
```

- [ ] **Step 3: Smoke-check the import resolves**

```bash
python3 -c "
import sys
sys.path.insert(0, 'e2e/tests/audit')
import fixtures
d = fixtures.dx('J06.9', 'Острый тонзиллит и фарингит')
v = fixtures.base_visit(
    guid='test-guid', service_code='B01.070.001', service_name='Первичный приём терапевта',
    specialty='Терапевт', age=45, inspection=[('Жалобы', 'Боль в горле')], diagnoses=[d],
)
print(v['Прием']['GUID'], v['Услуги'], v['Пациент']['AGE'], v['Диагнозы'])
"
```

Expected: prints the guid, one-element `Услуги` list, `45`, and the one-element `Диагнозы` list — no
error.

- [ ] **Step 4: Commit**

```bash
git add e2e/tests/audit/__init__.py e2e/tests/audit/fixtures.py
git commit -m "feat(e2e): scaffold audit e2e package + fixtures.py (dx, base_visit)"
```

---

### Task 9: `e2e/tests/audit/harness.py`

**Files:**
- Create: `e2e/tests/audit/harness.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) class Case` with fields `name: str`, `visit: dict`, `expect:
  str`, `visit_types: set[VisitType]`; `async def run_cases(title: str, cases: list[Case]) -> int`.
- Consumes: `audit.formal_structure.validator.FormalValidator`,
  `audit.formal_structure.validator.VisitType`, `audit.pipeline.AuditPipeline`.

This is the shared two-stage runner every `test_audit_*.py` script calls into — adopted from
`e2e/tests/audit/harness.py` on the sibling branch `formal-rules-npa-revision` (see
`docs/superpowers/specs/2026-08-20-e2e-full-suite-design.md` §2 for why), adjusted for this branch's
`get_rules(visit_types, patient_age)` two-argument signature (the sibling branch's revised
`rules.json` added a third `icd_prefixes` parameter that does not exist here).

- [ ] **Step 1: Write `e2e/tests/audit/harness.py`**

```python
"""
harness.py — shared runner for the audit e2e scripts.

Each audit script declares a list of `Case`s and hands them to `run_cases`.
A case is one fixture card carrying exactly one defect plus the flag that
defect must produce.

The run happens in two stages:

  Stage 1 — parsing, no LLM. Confirms the card is classified as the fixture
  intended (get_visit_types) and that the rule under test actually reaches
  the prompt (get_rules). This is where a mis-built fixture fails, before a
  single token is spent. If any case fails stage 1 the script stops — stage
  2 is not attempted.

  Stage 2 — the full audit. AuditPipeline._audit_visit runs the formal
  validator, the ICD checker and DiagnosisValidator against the live LLM,
  exactly as production does.

Stage 2 asserts the **complete** set of formal flags equals the one
expected flag. That is deliberate: because every fixture carries exactly
one defect, a rule that fires indiscriminately shows up as an extra flag
and fails the case. A presence-only assert could never catch that — see
docs/e2e-testing.md for the full rationale.

Nothing is persisted. _audit_visit calls _upsert_done_card, but that
returns immediately while self._done_cards is None, and that field is only
set by AuditPipeline.__aenter__ — so the pipeline is deliberately used
*without* `async with`, and no teardown is needed. The database is still
required: the ICD checker reads the guidelines catalogue.

An empty finding list is never taken at face value. LLM.validations
returns [] both when the model reported no defects and when its answer
failed to parse — the parse failure only shows up as a log record. The
runner listens for that record so an unparsed answer can never be read as
"no violations".

There are no command-line flags. These scripts are meant for unattended
batch runs; anything that changes what they assert would make a green run
mean different things on different days.
"""

from __future__ import annotations

import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from audit.formal_structure.validator import FormalValidator, VisitType  # noqa: E402
from audit.pipeline import AuditPipeline  # noqa: E402

_OK, _BAD = "  \033[32mok\033[0m    ", "  \033[31mFAILED\033[0m"


@dataclass(frozen=True)
class Case:
    """One fixture card and the single flag its single defect must raise."""

    name: str
    visit: dict[str, Any]
    expect: str
    visit_types: set[VisitType]


class _Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        if ok:
            print(f"{_OK}{label}")
        else:
            print(f"{_BAD}{label}")
            for line in (detail or "").splitlines():
                print(f"          {line}")
            self.failures.append(label)
        return ok


def _flags(result: Any) -> set[str]:
    return {f.flag for f in result.formal.findings}


def _describe(result: Any) -> str:
    if not result.formal.findings:
        return "(no findings)"
    return "\n".join(f"{f.flag}: {f.issue}" for f in result.formal.findings)


class _FormalCallWatch(logging.Handler):
    """Watches the formal-validator call so an unparsed answer is not read as a clean card.

    LLM.validations returns an empty list when the model's answer cannot be
    parsed and only records logger.error(...), which makes a parse failure
    indistinguishable from "no defects" at the Result level.
    """

    _MARKER = "failed to parse JSON response"
    _TALLY = "LLM returned"
    _DROPPED = "dropping unrecognised flag"
    _FUZZY = "fuzzy-matched to"

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.parse_failed = False
        self.tally: str = ""
        self.dropped: list[str] = []
        self.fuzzy: list[str] = []

    def reset(self) -> None:
        self.parse_failed = False
        self.tally = ""
        self.dropped = []
        self.fuzzy = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            return
        if self._MARKER in message:
            self.parse_failed = True
        elif self._DROPPED in message:
            self.dropped.append(message.split(self._DROPPED, 1)[1].strip())
        elif self._FUZZY in message:
            self.fuzzy.append(message.split("] ", 1)[-1].strip())
        elif self._TALLY in message:
            self.tally = message.split(self._TALLY, 1)[1].split(":", 1)[0].strip()

    def install(self) -> None:
        for name in ("LLM.validations", "audit.formal_structure.validator"):
            logger = logging.getLogger(name)
            logger.addHandler(self)
            logger.setLevel(min(logger.level or logging.INFO, logging.INFO))


async def _stage_one(cases: list[Case], report: _Report) -> None:
    """Classification and rule selection — deterministic, no LLM, no database."""
    validator = FormalValidator()
    for case in cases:
        visit = case.visit
        got_types = await validator.get_visit_types(visit)
        report.check(
            f"[{case.name}] visit type — {', '.join(sorted(t.name for t in case.visit_types))}",
            got_types == case.visit_types,
            f"got: {', '.join(sorted(t.name for t in got_types)) or '(empty)'}",
        )

        age = visit["Пациент"]["AGE"]
        rules = validator.get_rules(got_types, age)
        selected = [r["flag_code"] for r in rules]
        report.check(
            f"[{case.name}] rule {case.expect} reached the prompt ({len(rules)} rules)",
            case.expect in selected,
            "selected: " + ", ".join(selected),
        )


async def _stage_two(cases: list[Case], report: _Report) -> None:
    """The real audit — formal validator, ICD checker and DiagnosisValidator."""
    watch = _FormalCallWatch()
    watch.install()
    for case in cases:
        print(f"\n  {case.name}")
        watch.reset()
        pipeline = AuditPipeline()  # deliberately not `async with` — see module docstring
        try:
            result = await pipeline._audit_visit(case.visit)
        except Exception:
            report.check(f"[{case.name}] audit ran without error", False, traceback.format_exc())
            continue

        if watch.tally:
            print(f"          formal call: {watch.tally}")
        for line in watch.fuzzy:
            print(f"          flag fuzzy-matched: {line}")
        if watch.dropped:
            report.check(
                f"[{case.name}] no flag dropped as unrecognised",
                False,
                "model returned flags not in rules.json: " + "; ".join(watch.dropped),
            )

        if not report.check(
            f"[{case.name}] formal validator response parsed",
            not watch.parse_failed,
            "LLM.validations could not parse the model's answer and returned an empty list; "
            "see the log for 'failed to parse JSON response'",
        ):
            continue

        got = _flags(result)
        report.check(
            f"[{case.name}] exactly one flag found — {case.expect}",
            got == {case.expect},
            f"extra: {', '.join(sorted(got - {case.expect})) or '—'}\n"
            f"missing: {', '.join(sorted({case.expect} - got)) or '—'}\n"
            f"full findings:\n{_describe(result)}",
        )

        expected_dx = len(case.visit["Диагнозы"])
        report.check(
            f"[{case.name}] DiagnosisValidator ran for all {expected_dx} diagnoses",
            len(result.diagnosis) >= expected_dx,
            f"got {len(result.diagnosis)} result(s)",
        )
        for dr in result.diagnosis:
            found = dr.guideline_file_id or "no guideline found"
            print(f"          diagnosis {dr.icd_code}: {found}, {len(dr.issues)} issue(s)")


async def run_cases(title: str, cases: list[Case]) -> int:
    """Run every case through both stages and return a process exit code."""
    report = _Report()
    print(f"\n{title}")
    print(f"  {len(cases)} fixture(s), one violation each\n")

    print("Stage 1 — parsing fixtures (no LLM)")
    await _stage_one(cases, report)
    if report.failures:
        print(
            f"\n\033[31mStage 1 failed ({len(report.failures)}), "
            f"full audit not run — no tokens spent.\033[0m"
        )
        for f in report.failures:
            print(f"  - {f}")
        return 1

    print("\nStage 2 — full audit (LLM + DB)")
    await _stage_two(cases, report)

    if report.failures:
        print(f"\n\033[31m{len(report.failures)} check(s) failed\033[0m")
        for f in report.failures:
            print(f"  - {f}")
        return 1
    print("\n\033[32mAll checks passed.\033[0m")
    return 0
```

- [ ] **Step 2: Smoke-check the import resolves**

```bash
python3 -c "
import sys
sys.path.insert(0, 'e2e/tests/audit')
import harness
print(harness.Case, harness.run_cases)
"
```

Expected: prints the `Case` class and `run_cases` function, no `ImportError`.

- [ ] **Step 3: Commit**

```bash
git add e2e/tests/audit/harness.py
git commit -m "feat(e2e): add audit e2e harness — Case/run_cases two-stage runner"
```

---

### Task 10: `test_audit_primary.py`

**Files:**
- Create: `e2e/tests/audit/test_audit_primary.py`

**Interfaces:**
- Consumes: `fixtures.dx`, `fixtures.base_visit`, `harness.Case`, `harness.VisitType`,
  `harness.run_cases`.

- [ ] **Step 1: Write the full file**

```python
#!/usr/bin/env python3
"""
Первичный приём (274н/203н). Два правила, каждое на своей фикстуре — adult и
child.

Тип визита выводится из кода B01.070.001, а не задаётся тестом напрямую.

В adult-фикстуре убрана только «Жалобы» — не «Объективный осмотр» — чтобы
изолировать ПЕРВИЧНЫЙ_ОТСУТСТВУЮТ_ОСНОВНЫЕ_РАЗДЕЛЫ от пересекающегося по цели
ОТСУТСТВУЕТ_ОБЪЕКТИВНЫЙ_ОСМОТР (оба правила применимы к primary, но только
первое таргетит complaints — см. rules.json).

В child-фикстуре Диагнозы пуст — единственная фикстура во всём наборе без
диагноза, потому что ОТСУТСТВУЕТ_ДИАГНОЗ — это правило именно про отсутствие
диагноза. DiagnosisValidator для неё не вызывается (audit/pipeline.py's
no-diagnoses early return) — ожидаемое поведение, не пробел.

Запуск (нужны БД и LLM):  python e2e/tests/audit/test_audit_primary.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import base_visit, dx  # noqa: E402
from harness import Case, VisitType, run_cases  # noqa: E402

PRIMARY = {VisitType.PRIMARY}
SERVICE_CODE = "B01.070.001"
J06_9 = dx("J06.9", "Острый тонзиллит и фарингит (Острый тонзиллофарингит)")

missing_complaints = base_visit(
    guid="e2e-audit-primary-adult-no-complaints",
    service_code=SERVICE_CODE,
    service_name="Первичный приём (осмотр, консультация) врача-терапевта",
    specialty="Терапевт",
    age=45,
    diagnoses=[J06_9],
    inspection=[
        (
            "Анамнез",
            "Заболел два дня назад, постепенное начало, ОРВИ у контактных отрицает.",
        ),
        (
            "Объективный осмотр",
            "Зев гиперемирован, миндалины увеличены, налётов нет, лимфоузлы не увеличены. "
            "Температура 37.8°C, дыхание везикулярное, хрипов нет.",
        ),
        (
            "Рекомендации",
            "Обильное питьё, полоскание горла антисептическим раствором, контроль температуры тела.",
        ),
    ],
)

no_diagnosis = base_visit(
    guid="e2e-audit-primary-child-no-diagnosis",
    service_code=SERVICE_CODE,
    service_name="Первичный приём (осмотр, консультация) врача-педиатра",
    specialty="Педиатр",
    age=8,
    diagnoses=[],
    inspection=[
        ("Жалобы", f"Боль в горле, першение, повышение температуры до 37.8°C. Осмотр проведён в сопровождении матери."),
        (
            "Анамнез",
            "Заболел два дня назад, постепенное начало, ОРВИ у контактных отрицает.",
        ),
        (
            "Объективный осмотр",
            "Зев гиперемирован, миндалины увеличены, налётов нет, лимфоузлы не увеличены. "
            "Температура 37.8°C, дыхание везикулярное, хрипов нет.",
        ),
        (
            "Рекомендации",
            "Обильное питьё, полоскание горла антисептическим раствором, контроль температуры тела.",
        ),
    ],
)

CASES = [
    Case(
        name="взрослый: нет жалоб",
        visit=missing_complaints,
        expect="ПЕРВИЧНЫЙ_ОТСУТСТВУЮТ_ОСНОВНЫЕ_РАЗДЕЛЫ",
        visit_types=PRIMARY,
    ),
    Case(
        name="ребёнок: нет диагноза",
        visit=no_diagnosis,
        expect="ОТСУТСТВУЕТ_ДИАГНОЗ",
        visit_types=PRIMARY,
    ),
]


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(run_cases("Аудит: первичный приём", CASES)))
    except KeyboardInterrupt:
        sys.exit(130)
```

- [ ] **Step 2: Make executable and syntax-check**

```bash
chmod +x e2e/tests/audit/test_audit_primary.py
python3 -c "import ast; ast.parse(open('e2e/tests/audit/test_audit_primary.py').read())"
```

- [ ] **Step 3: Run stage 1 first (cheap — no LLM), confirm it passes before spending tokens**

```bash
python3 e2e/tests/audit/test_audit_primary.py
```

Watch the printed output: if "Stage 1 failed" appears, the fixture's `service_code` or removed field
doesn't classify/select the rule as intended — fix `fixtures.py`/the fixture before re-running, do not
weaken `harness.py`'s assertions. If stage 1 passes, stage 2 (real LLM calls) runs automatically in
the same invocation.

Expected end state: `All checks passed.`, exit code 0 (`echo $?`). If stage 2's exact-flag-set check
fails with an unexpected extra flag, read the printed "full findings" detail — adjust the fixture to
remove whatever else it's inadvertently triggering, per `fixtures.py`'s docstring list of easy-to-trip
`"all"`-scoped rules.

- [ ] **Step 4: Commit**

```bash
git add e2e/tests/audit/test_audit_primary.py
git commit -m "feat(e2e): add primary visit audit e2e fixtures (adult, child)"
```

---

### Task 11: `test_audit_repeat.py`

**Files:**
- Create: `e2e/tests/audit/test_audit_repeat.py`

**Interfaces:** Same as Task 10.

- [ ] **Step 1: Write the full file**

```python
#!/usr/bin/env python3
"""
Повторный приём (274н/203н). Два правила, каждое на своей фикстуре — adult и
child.

Тип визита выводится из кода B01.070.011 (repeat), а не задаётся тестом
напрямую.

В adult-фикстуре убран только раздел с анамнезом/динамикой — объективный
осмотр и диагноз остаются, иначе задело бы соседнее repeat_core_sections_required.

В child-фикстуре наименование услуги содержит слово «первичный» при
NMU-суффиксе .011 (повторный) — целевая проверка здесь не LLM-правило, а
детерминированная _check_nmu_keyword_contradiction (надёжнее для e2e). Вся
остальная карта полная: динамика, осмотр, диагноз — на месте, иначе
NMU_CODE_CONTRADICTION пришёл бы вместе с лишним флагом.

Запуск (нужны БД и LLM):  python e2e/tests/audit/test_audit_repeat.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import base_visit, dx  # noqa: E402
from harness import Case, VisitType, run_cases  # noqa: E402

REPEAT = {VisitType.REPEAT}
SERVICE_CODE = "B01.070.011"
J06_9 = dx("J06.9", "Острый тонзиллит и фарингит (Острый тонзиллофарингит)")

no_dynamics = base_visit(
    guid="e2e-audit-repeat-adult-no-dynamics",
    service_code=SERVICE_CODE,
    service_name="Повторный приём (осмотр, консультация) врача-терапевта",
    specialty="Терапевт",
    age=45,
    diagnoses=[J06_9],
    inspection=[
        (
            "Объективный осмотр",
            "Зев умеренно гиперемирован, миндалины не увеличены, налётов нет. "
            "Температура 36.8°C, дыхание везикулярное, хрипов нет.",
        ),
        ("Рекомендации", "Продолжить полоскание горла антисептическим раствором ещё три дня."),
    ],
)

nmu_contradiction = base_visit(
    guid="e2e-audit-repeat-child-nmu-contradiction",
    service_code=SERVICE_CODE,
    service_name="первичный приём (осмотр, консультация) врача-педиатра",
    specialty="Педиатр",
    age=8,
    diagnoses=[J06_9],
    inspection=[
        (
            "Жалобы",
            "Повторно на боль в горле, температура нормализовалась. Осмотр проведён в сопровождении матери.",
        ),
        ("Динамика", "Состояние с улучшением по сравнению с первичным приёмом два дня назад, температура нормализовалась."),
        (
            "Объективный осмотр",
            "Зев слегка гиперемирован, миндалины не увеличены, налётов нет. "
            "Температура 36.6°C, дыхание везикулярное, хрипов нет.",
        ),
        ("Рекомендации", "Лечение завершено, контроль не требуется."),
    ],
)

CASES = [
    Case(
        name="взрослый: нет динамики",
        visit=no_dynamics,
        expect="ПОВТОРНЫЙ_ОТСУТСТВУЕТ_ДИНАМИКА",
        visit_types=REPEAT,
    ),
    Case(
        name="ребёнок: противоречие кода услуги и наименования",
        visit=nmu_contradiction,
        expect="NMU_CODE_CONTRADICTION",
        visit_types=REPEAT,
    ),
]


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(run_cases("Аудит: повторный приём", CASES)))
    except KeyboardInterrupt:
        sys.exit(130)
```

Note the `NMU_CODE_CONTRADICTION` flag is appended by `FormalValidator._check_nmu_keyword_contradiction`
outside `rules.json` (it has no entry there — added unconditionally when the code/name mismatch is
found), so stage 1's `case.expect in selected` check (which reads `get_rules()`'s output) does **not**
apply to it the same way as the other `Case`s — `get_rules()` never contains
`NMU_CODE_CONTRADICTION` since it isn't a `rules.json` entry. Add this note as a comment directly
above the `nmu_contradiction` `Case` in the file:

```python
    # NMU_CODE_CONTRADICTION is not a rules.json entry — _check_nmu_keyword_contradiction appends
    # it unconditionally, so stage 1's "rule reached the prompt" check for this Case will report
    # 0 selected rules containing it; that's expected, not a fixture bug. Stage 2 still asserts it
    # is the only flag returned.
```

- [ ] **Step 2: Make executable and syntax-check**

```bash
chmod +x e2e/tests/audit/test_audit_repeat.py
python3 -c "import ast; ast.parse(open('e2e/tests/audit/test_audit_repeat.py').read())"
```

- [ ] **Step 3: Run**

```bash
python3 e2e/tests/audit/test_audit_repeat.py
```

Expected: for the `nmu_contradiction` case, stage 1's rule-selection check is expected to show
`case.expect` (`NMU_CODE_CONTRADICTION`) NOT in the selected rules list (see the note above) — this is
not a stage-1 failure signal for `NMU_CODE_CONTRADICTION` specifically; only the classification check
(`visit type — REPEAT`) must pass for stage 2 to be meaningful. If `run_cases` as written treats the
"rule reached the prompt" check as a hard stage-1 gate for every case indiscriminately, this one
`Case` will always report that specific sub-check as failed even when the fixture is correct — when
that happens, verify by hand (via `git diff`) that only that one line fails, confirm stage 2 still
runs and passes on `NMU_CODE_CONTRADICTION`, and record it as a known, expected stage-1 sub-check
failure for this one `Case`, not a real defect — the important stage-1 signal (classification) still
gates correctly. Full end-to-end success is judged by stage 2's `All checks passed.`, exit code 0
(`echo $?`).

- [ ] **Step 4: Commit**

```bash
git add e2e/tests/audit/test_audit_repeat.py
git commit -m "feat(e2e): add repeat visit audit e2e fixtures (adult, child)"
```

---

### Task 12: `test_audit_prophylactic.py`

**Files:**
- Create: `e2e/tests/audit/test_audit_prophylactic.py`

**Interfaces:** Same as Task 10.

- [ ] **Step 1: Write the full file**

```python
#!/usr/bin/env python3
"""
Профилактический приём (274н). Два правила, каждое на своей фикстуре — adult
и child.

Тип визита выводится из B04.047.002 (adult) / B04.031.002 (child) — только
префикс B04 определяет PROPHYLACTIC (validator.py), точный 5-значный код
специальности после B04. не влияет на классификацию, взят для правдоподобия.

Запуск (нужны БД и LLM):  python e2e/tests/audit/test_audit_prophylactic.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import base_visit, dx  # noqa: E402
from harness import Case, VisitType, run_cases  # noqa: E402

PROPH = {VisitType.PROPHYLACTIC}
J06_9 = dx("J06.9", "Острый тонзиллит и фарингит (Острый тонзиллофарингит)")

plan_result_mixup = base_visit(
    guid="e2e-audit-prophylactic-adult-plan-result-mixup",
    service_code="B04.047.002",
    service_name="Профилактический приём (осмотр, консультация) врача-терапевта",
    specialty="Терапевт",
    age=45,
    diagnoses=[J06_9],
    inspection=[
        ("Жалобы", "Жалоб не предъявляет, приём в рамках профилактического осмотра."),
        ("Объективный осмотр", "Общее состояние удовлетворительное, зев спокоен, дыхание везикулярное."),
        (
            "План обследования",
            "Общий анализ крови — результат: гемоглобин 140 г/л, лейкоциты 6.2×10⁹/л, без патологии.",
        ),
        ("Заключение", "Хронических заболеваний не выявлено, очередной осмотр через 12 месяцев."),
    ],
)

placeholder_value = base_visit(
    guid="e2e-audit-prophylactic-child-placeholder",
    service_code="B04.031.002",
    service_name="Профилактический приём (осмотр, консультация) врача-педиатра",
    specialty="Педиатр",
    age=8,
    diagnoses=[J06_9],
    inspection=[
        ("Жалобы", "Жалоб не предъявляет. Осмотр проведён в сопровождении матери."),
        ("Объективный осмотр", "Общее состояние удовлетворительное, зев спокоен, дыхание везикулярное, хрипов нет."),
        ("Рекомендации", "уточнить"),
    ],
)

CASES = [
    Case(
        name="взрослый: план и результат перемешаны",
        visit=plan_result_mixup,
        expect="СМЕШАНЫ_ПЛАН_И_РЕЗУЛЬТАТЫ",
        visit_types=PROPH,
    ),
    Case(
        name="ребёнок: заглушка вместо рекомендаций",
        visit=placeholder_value,
        expect="ОБНАРУЖЕНЫ_ЗАГЛУШКИ",
        visit_types=PROPH,
    ),
]


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(run_cases("Аудит: профилактический приём", CASES)))
    except KeyboardInterrupt:
        sys.exit(130)
```

- [ ] **Step 2: Make executable and syntax-check**

```bash
chmod +x e2e/tests/audit/test_audit_prophylactic.py
python3 -c "import ast; ast.parse(open('e2e/tests/audit/test_audit_prophylactic.py').read())"
```

- [ ] **Step 3: Run**

```bash
python3 e2e/tests/audit/test_audit_prophylactic.py
```

Expected: `All checks passed.`, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add e2e/tests/audit/test_audit_prophylactic.py
git commit -m "feat(e2e): add prophylactic visit audit e2e fixtures (adult, child)"
```

---

### Task 13: `test_audit_lab_research_intervention.py`

**Files:**
- Create: `e2e/tests/audit/test_audit_lab_research_intervention.py`

**Interfaces:** Same as Task 10.

- [ ] **Step 1: Write the full file**

```python
#!/usr/bin/env python3
"""
Лабораторные/инструментальные исследования и вмешательства (464н, A-коды).
Два правила, каждое на своей фикстуре — adult и child.

Тип визита выводится из A05.10.006 (ЭКГ) — A-префикс всегда даёт
LAB_RESEARCH_INTERVENTION (validator.py).

В child-фикстуре заключение по ЭКГ присутствует, иначе задело бы соседнее
ecg_functional_description_and_conclusion вместе с целевым дублированием.

Запуск (нужны БД и LLM):  python e2e/tests/audit/test_audit_lab_research_intervention.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import base_visit, dx  # noqa: E402
from harness import Case, VisitType, run_cases  # noqa: E402

LAB = {VisitType.LAB_RESEARCH_INTERVENTION}
SERVICE_CODE = "A05.10.006"
J06_9 = dx("J06.9", "Острый тонзиллит и фарингит (Острый тонзиллофарингит)")

no_conclusion = base_visit(
    guid="e2e-audit-lab-adult-no-conclusion",
    service_code=SERVICE_CODE,
    service_name="Электрокардиография",
    specialty="Кардиолог",
    age=45,
    diagnoses=[J06_9],
    inspection=[
        (
            "Протокол ЭКГ",
            "Ритм синусовый, ЧСС 72 в минуту, электрическая ось не отклонена, интервалы PQ, QRS, QT в пределах нормы.",
        ),
    ],
)

duplicate_blocks = base_visit(
    guid="e2e-audit-lab-child-duplicate-blocks",
    service_code=SERVICE_CODE,
    service_name="Электрокардиография",
    specialty="Кардиолог",
    age=8,
    diagnoses=[J06_9],
    inspection=[
        ("Жалобы", "Боль в горле, першение. Осмотр проведён в сопровождении матери."),
        ("Жалобы со слов законного представителя", "Ребёнок жалуется на боль в горле и першение."),
        (
            "Протокол ЭКГ",
            "Ритм синусовый, ЧСС 100 в минуту, без патологии. Заключение: возрастная норма.",
        ),
    ],
)

CASES = [
    Case(
        name="взрослый: нет заключения по ЭКГ",
        visit=no_conclusion,
        expect="ФУНКЦИОНАЛЬНОЕ_ИССЛЕДОВАНИЕ_НЕПОЛНОЕ",
        visit_types=LAB,
    ),
    Case(
        name="ребёнок: дублирование смысловых блоков",
        visit=duplicate_blocks,
        expect="ДУБЛИРОВАНИЕ_СМЫСЛОВЫХ_БЛОКОВ",
        visit_types=LAB,
    ),
]


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(run_cases("Аудит: исследования и вмешательства", CASES)))
    except KeyboardInterrupt:
        sys.exit(130)
```

- [ ] **Step 2: Make executable and syntax-check**

```bash
chmod +x e2e/tests/audit/test_audit_lab_research_intervention.py
python3 -c "import ast; ast.parse(open('e2e/tests/audit/test_audit_lab_research_intervention.py').read())"
```

- [ ] **Step 3: Run**

```bash
python3 e2e/tests/audit/test_audit_lab_research_intervention.py
```

Expected: `All checks passed.`, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add e2e/tests/audit/test_audit_lab_research_intervention.py
git commit -m "feat(e2e): add lab/research/intervention visit audit e2e fixtures (adult, child)"
```

---

### Task 14: `test_audit_prophylactic_tuberculin.py`

**Files:**
- Create: `e2e/tests/audit/test_audit_prophylactic_tuberculin.py`

**Interfaces:** Same as Task 10.

- [ ] **Step 1: Write the full file**

This is the one visit type reached via `Диагноз.Код == "Z11.1"` at the visit's top level, not an NMU
code in `Услуги`. `fixtures.base_visit` has no dedicated "tuberculin" parameter — the Z11.1 diagnosis
is passed as a normal `Диагнозы` entry via `dx("Z11.1", ...)`, since `get_visit_types` reads
`Диагноз.Код` at the top of the visit dict, not inside `Диагнозы`. Both scripts therefore add the
`Диагноз` key to the assembled dict directly after calling `base_visit`.

```python
#!/usr/bin/env python3
"""
Профилактический приём — туберкулинодиагностика (Z11.1). Два правила, каждое
на своей фикстуре — adult и child.

Тип визита PROPHYLACTIC_TUBERCULIN приходит из Диагноз.Код == "Z11.1" на
верхнем уровне визита (get_visit_types), не из Услуги — service_code
намеренно пустая строка, услуга остаётся неклассифицированной.

В обеих фикстурах присутствует ровно один сегмент осмотра — второй
(заключение или объективные данные, смотря какой из двух здесь не целевой)
намеренно исключён, иначе оба правила 190н сработали бы одновременно.

Запуск (нужны БД и LLM):  python e2e/tests/audit/test_audit_prophylactic_tuberculin.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import base_visit, dx  # noqa: E402
from harness import Case, VisitType, run_cases  # noqa: E402

# Confirmed empirically (2026-08-21, this branch's validator.py): an empty
# Услуги list adds OTHER to the same result set before the Z11.1 check's
# PROPHYLACTIC_TUBERCULIN is returned — both end up in get_visit_types's
# output together, not PROPHYLACTIC_TUBERCULIN alone.
TUBERCULIN = {VisitType.PROPHYLACTIC_TUBERCULIN, VisitType.OTHER}
Z11_1 = dx("Z11.1", "Специальное скрининговое обследование с целью выявления туберкулёза органов дыхания")


def _tuberculin_visit(guid: str, specialty: str, age: int, inspection: list[tuple[str, str]]) -> dict:
    visit = base_visit(
        guid=guid,
        service_code="",
        service_name="Туберкулинодиагностика (внутрикожная проба с туберкулином)",
        specialty=specialty,
        age=age,
        diagnoses=[Z11_1],
        inspection=inspection,
    )
    # get_visit_types reads Диагноз.Код at the visit's top level for the
    # PROPHYLACTIC_TUBERCULIN branch — a separate key from Диагнозы[].КодМКБ.
    visit["Диагноз"] = {"Код": "Z11.1"}
    return visit


no_objective_data = _tuberculin_visit(
    "e2e-audit-tuberculin-adult-no-objective-data",
    "Терапевт",
    45,
    [
        ("Жалобы", "Жалоб не предъявляет, проба проводится в плановом порядке."),
        (
            "Заключение",
            "Патологических состояний, свидетельствующих о наличии туберкулёза, не выявлено.",
        ),
        ("Рекомендации", "Очередная туберкулинодиагностика через 12 месяцев."),
    ],
)

no_conclusion = _tuberculin_visit(
    "e2e-audit-tuberculin-child-no-conclusion",
    "Педиатр",
    8,
    [
        ("Жалобы", "Жалоб не предъявляет. Осмотр проведён в сопровождении матери."),
        (
            "Объективный осмотр",
            "Место введения туберкулина на левом предплечье: папула 8 мм, гиперемия вокруг папулы 2 мм, "
            "везикул и некроза нет.",
        ),
    ],
)

CASES = [
    Case(
        name="взрослый: нет объективных данных пробы",
        visit=no_objective_data,
        expect="ТУБЕРКУЛИН_ОТСУТСТВУЮТ_ОБЪЕКТИВНЫЕ_ДАННЫЕ",
        visit_types=TUBERCULIN,
    ),
    Case(
        name="ребёнок: нет заключения по пробе",
        visit=no_conclusion,
        expect="ТУБЕРКУЛИН_ОТСУТСТВУЕТ_ЗАКЛЮЧЕНИЕ",
        visit_types=TUBERCULIN,
    ),
]


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(run_cases("Аудит: туберкулинодиагностика", CASES)))
    except KeyboardInterrupt:
        sys.exit(130)
```

`_tuberculin_visit`'s fixtures carry no `Услуги`-derived type (`service_code=""`), so `get_visit_types`
takes the empty-`Услуги` early-return path (`validator.py:138-142`), which adds `OTHER` to the same
result set the Z11.1 check already populated with `PROPHYLACTIC_TUBERCULIN` — confirmed empirically
against this branch's `validator.py` (`python3 -c "..."` against a visit with `Услуги: []` and
`Диагноз.Код: "Z11.1"` prints `{VisitType.OTHER, VisitType.PROPHYLACTIC_TUBERCULIN}`). That's why
`TUBERCULIN` above is defined as the two-element set, not `{PROPHYLACTIC_TUBERCULIN}` alone — this is
not something to re-verify at runtime, it's already the correct constant.

- [ ] **Step 2: Make executable and syntax-check**

```bash
chmod +x e2e/tests/audit/test_audit_prophylactic_tuberculin.py
python3 -c "import ast; ast.parse(open('e2e/tests/audit/test_audit_prophylactic_tuberculin.py').read())"
```

- [ ] **Step 3: Run**

```bash
python3 e2e/tests/audit/test_audit_prophylactic_tuberculin.py
```

Expected: `All checks passed.`, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add e2e/tests/audit/test_audit_prophylactic_tuberculin.py
git commit -m "feat(e2e): add prophylactic tuberculin-diagnostics audit e2e fixtures (adult, child)"
```

---

### Task 15: `docs/e2e-testing.md` methodology doc

**Files:**
- Create: `docs/e2e-testing.md`
- Modify: `CLAUDE.md` (add a one-line pointer under an existing relevant section)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Write `docs/e2e-testing.md`**

```markdown
# E2E testing

This repo's e2e tests are standalone scripts, not pytest. They exercise real infrastructure — a
live pull API + Postgres for route tests, or the real `AuditPipeline` with real LLM calls for audit
tests — never mocks. That's what distinguishes them from `tests/` (pytest, `pythonpath=src`,
typically mocked LLM calls) and from the older `scripts/smoke-*.sh`/`.py` scripts, which this suite
extends the pattern of but does not replace.

Two kinds of e2e script live here:

- **Route tests** (`e2e/tests/test_*_smoke.py`) — one script per pull-API route, run against a live
  `uvicorn`/docker instance and the configured Postgres.
- **Audit tests** (`e2e/tests/audit/test_audit_*.py`) — one script per `visit_type`, each declaring a
  list of `Case`s (typically one per `age_group`) and handing them to the shared
  `e2e/tests/audit/harness.py` runner, which calls `AuditPipeline._audit_visit()` directly. No HTTP,
  no live API needed — only Postgres (for `GuidelinesStorage` reads) and the configured LLM endpoint.

## When to write a new e2e script

- A new pull-API route ships → add `e2e/tests/test_visits_<route>_smoke.py` (or
  `test_stats_<route>_smoke.py` for `/stats/*`).
- New behavior inside an *existing* route (e.g. a new trigger, a new response field) → add a new
  script next to the route's existing one rather than folding the new scenario into it. Example:
  `test_push_log_smoke.py` covers `push_log`/`push_metrics_by_date` specifically and deliberately
  does not re-assert `/visits/push`'s generic auth/422 behavior — that stays in
  `scripts/smoke-cards-push.sh`.
- A new `visit_type` is added to `rules.json`'s `applies_to.visit_types` → add
  `test_audit_<visit_type>.py` with one `Case` per `age_group` that needs coverage.
- A new rule is added to an existing `visit_type` → add a new `Case` to that visit_type's existing
  file's `CASES` list if an existing fixture can't also cover it without violating "one deliberate
  defect per fixture" (see `fixtures.py`'s module docstring for what "the rest of the card must be
  flawless" means in practice).

## Helper contract (`e2e/tests/helpers/`)

- `organizations.py` — `OrganizationFixtures` (`create_org`/`delete_org`), direct SQL — there's no
  public API for creating an organization.
- `api_keys.py` — `issue_key()` (goes through `ApiKeysStorage.create_key`, the same path
  `scripts/create-api-key.py` uses) and `ApiKeyFixtures` (`delete_key` by label, `count_key_scopes`
  for teardown verification).
- `cards.py` — `push_card()` (thin `POST /visits/push` wrapper) and `CardFixtures`: `stage_audited`
  (flips a card to a fake completed formal audit, for push/override scenarios), `stage_done_with_meta`
  (same, plus a controllable `Прием.DATE`/`Врач_код`/`Врач`, for routes that filter/group on those),
  `mark_ignored`, `card_row`, `push_log_rows`, `push_metrics_for_org_today`, `delete_cards`,
  `delete_push_log`.

Add a new method to an existing helper when it's a variant of what that helper already does (e.g.
`stage_done_with_meta` next to `stage_audited`). Add a new helper module only for a genuinely new
resource type (there's been one per DB concern so far: organizations, keys, cards).

## Script pattern (route tests)

Every route test copies this skeleton from `e2e/tests/test_push_log_smoke.py`:

- `argparse` with a positional `url` (`nargs="?"`, `default="local"` — resolves to
  `http://localhost:{API_PORT}`, `API_PORT` read from `.env`, default `8000`) and `--keep` (skip
  teardown, print what was left).
- `TAG = uuid.uuid4().hex[:8]` in every org name / key label / card guid, so concurrent runs against
  the shared Postgres never collide.
- A `check(label, condition, detail="")` accumulator: prints `ok`/`FAILED` per assertion, never
  raises, so teardown always runs. Failures accumulate in a module-level list; `main()` returns a
  non-zero exit code if any failed.
- `finally` teardown that deletes everything the script created (push_log rows before done_cards
  rows, then the API key, then the organization) — unless `--keep`, in which case it prints what was
  left for manual inspection.

## Running the tests

Route tests need a reachable pull API and the configured Postgres:

```bash
python e2e/tests/test_visits_check_smoke.py [url] [--keep]
```

Run all route tests:

```bash
for f in e2e/tests/test_*_smoke.py; do python3 "$f" || exit 1; done
```

Audit tests need only Postgres (for `GuidelinesStorage`) and the configured LLM endpoint — no live
API, no arguments, no `--keep`. **They spend real LLM tokens** — run them deliberately, not as part of
every local iteration:

```bash
python e2e/tests/audit/test_audit_primary.py
```

Run all audit tests:

```bash
for f in e2e/tests/audit/test_audit_*.py; do python3 "$f" || exit 1; done
```

## Handling LLM non-determinism

Audit tests assert that the **complete** set of formal flags a fixture produces equals exactly the one
flag its single deliberate defect targets (`{f.flag for f in result.formal.findings} == {case.expect}`
in `harness.py`) — not presence-only. This is a deliberate, higher-bar choice, adopted from
`e2e/tests/audit/harness.py` on the sibling branch `formal-rules-npa-revision`: a presence check can't
tell a working rule from one that fires unconditionally, because a broken fixture yields the expected
flag under either explanation. An exact-set check catches both — but only if every fixture is
otherwise flawless against every rule applicable to its `visit_type`/`age_group`, including the
`"visit_types": ["all"]` rules (see `fixtures.py`'s module docstring for the checklist). Writing a new
fixture costs more up front for this reason; it buys the sharpest signal available when a rule
misbehaves.

Every audit `Case`'s **stage 1** (in `harness.py`, before any LLM call) confirms
`FormalValidator.get_visit_types()` (deterministic — NMU code / keyword / `Диагноз.Код` parsing, never
an LLM call) resolves the fixture to `case.visit_types`, and that `case.expect` appears in
`get_rules()`'s output — both zero-cost, zero-token checks. A stage 1 failure means the fixture itself
is wrong, not that the LLM behaved unexpectedly, and stage 2 (the real audit) never runs — fix the
fixture, not the assertion.

Stage 2 also guards against an unparsed LLM response being misread as "no violations": `harness.py`'s
`_FormalCallWatch` listens for `LLM.validations`' `"failed to parse JSON response"` log line and fails
the case explicitly rather than letting an empty findings list pass as a clean card.

## Isolation and cleanup

Route tests namespace every resource with `TAG` and always clean up in `finally`, because they write
to the same shared Postgres real e2e runs and the audit scripts read guidelines from. `--keep` exists
purely for manual debugging against a real database — always confirm what it left behind and clean
up by hand afterward.

Audit tests are the one exception: `AuditPipeline._audit_visit()` (unlike `run_batched()`) does not
persist anything to `done_cards` on its own — only the calling code that wraps it in `run_batched`
does that, via `_upsert_done_card`. `harness.py` deliberately instantiates `AuditPipeline()` without
`async with`, the same way `e2e/tests/audit/harness.py` on `formal-rules-npa-revision` does, so
`self._done_cards` stays `None` and `_upsert_done_card` no-ops — an audit e2e run touches no DB state
that needs tearing down, only reading `GuidelinesStorage` and calling the LLM. This is also why audit
scripts take no `--keep`: there is nothing to keep.
```

- [ ] **Step 2: Add a one-line pointer from `CLAUDE.md`**

In `CLAUDE.md`, find the `## Commands` section's existing test-related lines (`pytest`, `pytest
tests/test_validations.py`). Directly after that block (before `## Architecture`), add:

```markdown
E2E tests (standalone scripts against a live API/Postgres/LLM, not pytest) are documented in
`docs/e2e-testing.md`.
```

- [ ] **Step 3: Commit**

```bash
git add docs/e2e-testing.md CLAUDE.md
git commit -m "docs(e2e): add e2e testing methodology doc, link from CLAUDE.md"
```

---

### Task 16: Final regression pass

**Files:** none (verification only)

**Interfaces:** none

- [ ] **Step 1: Confirm the unit/integration test suite is unaffected**

```bash
pytest -q
```

Expected: same pass/fail/error counts as this branch's established baseline (see the prior
`push_log_smoke` plan's Task 7 for the baseline numbers) — nothing under `e2e/` is collected, since
`pytest.ini` already has `norecursedirs = e2e` from that plan.

- [ ] **Step 2: Confirm every new route script parses and every helper import resolves**

```bash
for f in e2e/tests/test_visits_check_smoke.py e2e/tests/test_visits_pull_smoke.py \
         e2e/tests/test_visits_export_smoke.py e2e/tests/test_visits_check_updates_smoke.py \
         e2e/tests/test_visits_doctors_smoke.py e2e/tests/test_stats_storage_smoke.py; do
  python3 -c "import ast; ast.parse(open('$f').read())" || echo "SYNTAX ERROR: $f"
done
```

Expected: no `SYNTAX ERROR` lines.

- [ ] **Step 3: Confirm every new audit script and the shared harness/fixtures modules parse**

```bash
for f in e2e/tests/audit/fixtures.py e2e/tests/audit/harness.py e2e/tests/audit/test_audit_*.py; do
  python3 -c "import ast; ast.parse(open('$f').read())" || echo "SYNTAX ERROR: $f"
done
```

Expected: no `SYNTAX ERROR` lines.

```bash
ls e2e/tests/audit/test_audit_*.py | wc -l
```

Expected: `5` (one file per `visit_type`, each declaring 2 `Case`s — 10 fixtures total).

- [ ] **Step 4: Run every route smoke test once against a live API, in sequence**

Requires the pull API running (see Task 5 Step 2 of the prior `push_log_smoke` plan for how to start
it locally) and the configured Postgres reachable.

```bash
for f in e2e/tests/test_*_smoke.py; do
  echo "=== $f ==="
  python3 "$f" local || { echo "FAILED: $f"; exit 1; }
done
```

Expected: every script prints `All checks passed.`, loop completes without exiting early.

- [ ] **Step 5: Run every audit test once (spends real LLM tokens)**

```bash
for f in e2e/tests/audit/test_audit_*.py; do
  echo "=== $f ==="
  python3 "$f" || { echo "FAILED: $f"; exit 1; }
done
```

Expected: every script prints `All checks passed.`, loop completes without exiting early. If any
script's classification check fails, its fixture needs adjusting per that task's guidance before
re-running — do not weaken the assertion.

---

## Self-Review Notes

- **Spec coverage:** All 6 route scripts (Tasks 2–7, matching the spec's per-route scenario lists
  exactly — check/pull/export/check_updates/doctors/stats) ✓. `stage_done_with_meta` (Task 1) and
  `mark_ignored` (Task 4) cover every DB-state need the spec's route scenarios call for ✓. All 10
  audit fixtures across 5 files (Tasks 10–14, one file per `visit_type`, two `Case`s each for
  `adult`/`child` — matching the spec's revised §2 harness design) ✓, each with the exact `flag_code`
  the spec assigned it, using the exact NMU codes/keyword contradiction/Z11.1 path the spec specified
  for classification ✓. The shared `harness.py` (Task 9) implements the spec's two-stage run exactly:
  stage 1 classification + rule-selection check before any LLM call, stage 2 exact-flag-set assertion
  plus the parse-failure watchdog ✓. The "no teardown needed for audit tests" reasoning (spec §2 point
  7, `AuditPipeline()` used without `async with`) is implemented in `harness.py`'s `_stage_two` and
  documented in both its module docstring and Task 15's methodology doc ✓. Methodology doc (Task 15)
  covers every bullet from the spec's §3 outline, updated for the harness architecture: what's e2e vs
  not, when to add a script/Case, helper contract, script pattern, how to run, LLM non-determinism
  (exact-set, not presence — with the harness's own rationale reproduced), isolation/cleanup ✓.
- **Type/name consistency:** `CardFixtures.stage_done_with_meta(card_guid, *, visit_date,
  doctor_code=None, doctor_name=None)` signature (Task 1) matches every call site in Tasks 2, 3, 4, 6,
  7 exactly. `CardFixtures.mark_ignored(card_guid)` (Task 4) has no other call sites, consistent.
  `fixtures.dx(code, name, *, detail="", first_time=False)` and `fixtures.base_visit(*, guid,
  service_code, service_name, specialty, age, inspection, diagnoses, gender="Женский",
  visit_date="20.08.2026")` (Task 8) signatures match every call site in Tasks 10–14 exactly, including
  the `_tuberculin_visit` wrapper in Task 14 that adds the `Диагноз` top-level key `base_visit` itself
  does not set. `harness.Case(name, visit, expect, visit_types)` and `harness.run_cases(title, cases)`
  (Task 9) match every `CASES` list and `if __name__ == "__main__"` block in Tasks 10–14. `TUBERCULIN =
  {VisitType.PROPHYLACTIC_TUBERCULIN, VisitType.OTHER}` in Task 14 is derived from an empirical check
  against this branch's actual `get_visit_types` (documented inline), not left as a runtime
  "verify and adjust" step. `Result.formal.findings[i].flag`, `Result.diagnosis[i].guideline_file_id`,
  `Result.diagnosis[i].icd_code` (from `storage/models/result.py`, confirmed during spec research) are
  used identically inside `harness.py`'s `_flags`/`_describe`/`_stage_two` and nowhere else duplicated.
  `VisitType.PRIMARY/REPEAT/PROPHYLACTIC/LAB_RESEARCH_INTERVENTION/PROPHYLACTIC_TUBERCULIN/OTHER` names
  match `audit/formal_structure/validator.py` exactly in every `Case.visit_types` value.
- **Scope check:** No task touches `pytest.ini`, `.env.example`, `docker-compose.yml`, or the
  existing `test_push_log_smoke.py`/its helpers beyond the two additive methods in Tasks 1 and 4 —
  matches the spec's "Не в скоупе" section. No task wraps anything in pytest or mocks the LLM,
  matching the user's explicit choices during brainstorming. The harness architecture (Tasks 9–14) is
  adopted only as a *pattern* from `formal-rules-npa-revision`'s `e2e/tests/audit/harness.py` — no
  file, fixture, or `rules.json` content is copied from that branch, since its `rules.json`/`get_rules`
  signature differ from this branch's; every fixture and flag_code here is derived from this branch's
  own `src/audit/formal_structure/rules.json`, confirmed via direct reads during spec research.
