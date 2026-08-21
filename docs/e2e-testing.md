# E2E testing

This repo's e2e tests are standalone scripts, not pytest. They exercise real infrastructure — a
live pull API + Postgres for route tests, or the real `AuditPipeline` with real LLM calls for audit
tests — never mocks. That's what distinguishes them from `tests/` (pytest, `pythonpath=src`,
typically mocked LLM calls) and from the older `scripts/smoke-*.sh`/`.py` scripts, which this suite
extends the pattern of but does not replace.

Test scripts are grouped into subfolders by subject, one folder per topic:

- **`e2e/tests/routes/`** — one script per pull-API route, run against a live `uvicorn`/docker
  instance and the configured Postgres. Documented in full below.
- **`e2e/tests/audit/`** — fixture cards with a known defect run through the real
  `AuditPipeline`, real LLM calls included. This suite has its own methodology doc at
  `e2e/tests/audit/README.md` — read that instead of duplicating it here; it covers the two-stage
  run, why the full flag set is asserted rather than just presence of the expected flag, how to add
  a fixture, and the runner script (`e2e/run-diagnosis-graph-tests.sh`).

Add a new subfolder only for a genuinely new test subject (route tests vs. audit-fixture tests are
the two so far) — a new script within an existing subject goes next to its siblings, not into a new
folder of its own.

## When to write a new route test script

- A new pull-API route ships → add `e2e/tests/routes/test_visits_<route>_smoke.py` (or
  `test_stats_<route>_smoke.py` for `/stats/*`).
- New behavior inside an *existing* route (e.g. a new trigger, a new response field) → add a new
  script next to the route's existing one rather than folding the new scenario into it. Example:
  `test_push_log_smoke.py` covers `push_log`/`push_metrics_by_date` specifically and deliberately
  does not re-assert `/visits/push`'s generic auth/422 behavior — that stays in
  `scripts/smoke-cards-push.sh`.

For when to add an audit fixture, see `e2e/tests/audit/README.md`'s "Как добавить фикстуру" section.

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

Every route test copies this skeleton from `e2e/tests/routes/test_push_log_smoke.py`:

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

## Running the route tests

```bash
python e2e/tests/routes/test_visits_check_smoke.py [url] [--keep]
```

Run all route tests:

```bash
for f in e2e/tests/routes/test_*_smoke.py; do python3 "$f" || exit 1; done
```

For running the audit suite, see `e2e/tests/audit/README.md`.

## Isolation and cleanup

Route tests namespace every resource with `TAG` and always clean up in `finally`, because they write
to the same shared Postgres real e2e runs use. `--keep` exists purely for manual debugging against a
real database — always confirm what it left behind and clean up by hand afterward.
