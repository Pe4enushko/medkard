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

`Case` (`harness.py`) has an `exact: bool = True` field, so exact-set is the default for every case and
must be deliberately opted out of, not the other way around. Of the suite's 10 cases, 4 set
`exact=False`: both cases in `test_audit_primary.py` and both cases in `test_audit_repeat.py`.
Each of those was only downgraded after fixture wording alone was proven, empirically and across
multiple rewrite attempts (documented in each file's module docstring), unable to reach a clean
exact-set result — typically because two rules are worded near-synonymously in `rules.json` and any
fixture text removing one rule's target content also trips the other. `exact=False` is a documented
escape hatch for a confirmed fixture-text ceiling, not a default and not a way to paper over a rule
that hasn't been investigated.

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
