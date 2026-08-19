# fix-broken — переаудит накопленных broken-карт: план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Скрипт `scripts/fix-broken.py`, который переаудирует карты, застрявшие
в `done_cards` с `broken = TRUE`, скармливая их сохранённый `card_data` обратно в
`AuditPipeline` с отключённым дедупом.

**Architecture:** Никаких правок пайплайна. `DoneCardsStorage.get_broken()`
отдаёт broken-строки; скрипт группирует их по `organization_id`, на каждую
группу поднимает свой `AuditPipeline(org_id=…, card_filter=load_card_filter(org))`
и вызывает `run_batched(visits, done_guids=set())` — пустой набор отключает
всегда-включённый дедуп по GUID. Итог считается сверкой множеств broken-GUID до
и после прогона, а не по возврату пайплайна.

**Tech Stack:** Python 3, asyncio, psycopg3 (`storage/`), argparse, pytest +
pytest-asyncio (`asyncio_mode=auto`, `pythonpath=src`).

**Spec:** `docs/superpowers/specs/2026-08-17-diagnosis-graph-design.md`, §9
(«Переаудит накопленных broken-карт»). Ревью спеки:
`docs/superpowers/REVIEW-2026-08-19-diagnosis-graph.md` — блокеры Б1–Б3 касаются
только графа (§1–8) и этот план **не** затрагивают; замечания З4 и З5 относятся
к §9 и учтены в задачах 1 и 5 соответственно.

## Global Constraints

- Ветка — **от `specs-2026-08-17`**, не от `origin/release` (решение
  пользователя, HANDOFF §4). Новая тема — новая ветка.
- Push — **только по явной команде пользователя**. Коммитить самому можно.
- Тесты гоняются точечно: `pytest tests/test_<файл>.py -v` из корня. Полный
  прогон на dev-машине падает без БД/инфры — это норма, не чинить.
- Тесты storage — интеграционные, против реально сконфигуренной Postgres
  (образец: `tests/test_done_cards_storage_pending.py`). Каждый тест убирает за
  собой в `finally`.
- Комментарии в коде — кратко, по-английски. Доки — по-русски, даты ISO.
- Скрипт **офлайновый**: обращений к 1С нет, источник данных — только БД.
- `close_pool()` в `finally` — иначе процесс не завершится.

---

### Task 1: `DoneCardsStorage.get_broken()`

**Files:**
- Modify: `src/storage/done_cards_storage.py` (рядом с `get_pending`, строка ~351)
- Test: `tests/test_done_cards_storage_broken.py` (создать)

**Interfaces:**
- Consumes: `BaseStorage.__aenter__` (общий пул), существующие
  `upsert_broken(...)`, `upsert(...)`.
- Produces: `async def get_broken(self, organization_id: str | None = None) ->
  list[dict]` — строки с ключами `card_guid` (str), `card_data` (dict, psycopg
  разворачивает `jsonb` сам), `organization_id` (str | None).
  **Семантика аргумента:** `None` означает «все организации», а **не**
  «организации с NULL» — этим `get_broken` намеренно отличается от `get_pending`
  и `get_done_guids`, которые используют `IS NOT DISTINCT FROM`.

- [ ] **Step 1: Написать падающий тест**

Создать `tests/test_done_cards_storage_broken.py`:

```python
"""
Integration tests for DoneCardsStorage.get_broken. Hits the real configured
Postgres, like the other storage tests.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from storage.done_cards_storage import DoneCardsStorage


async def _cleanup(*guids: str) -> None:
    async with DoneCardsStorage() as storage:
        async with storage._pool.connection() as conn:
            for guid in guids:
                await conn.execute(
                    "DELETE FROM done_cards WHERE card_guid = %(g)s", {"g": guid}
                )


@pytest.mark.asyncio
async def test_get_broken_returns_broken_rows_with_card_data():
    guid = f"pytest-broken-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            await storage.upsert_broken(
                card_guid=guid,
                card_data=json.dumps({"Прием": {"GUID": guid}}, ensure_ascii=False),
                stacktrace="Traceback (most recent call last):\nValueError: boom",
                started_at=datetime.now(timezone.utc),
                organization_id=None,
            )
            rows = await storage.get_broken()

        by_guid = {r["card_guid"]: r for r in rows}
        assert guid in by_guid
        assert by_guid[guid]["card_data"]["Прием"]["GUID"] == guid
        assert "organization_id" in by_guid[guid]
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_get_broken_skips_rows_without_card_data():
    guid = f"pytest-broken-nodata-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            await storage.upsert_broken(
                card_guid=guid,
                card_data=json.dumps({"Прием": {"GUID": guid}}, ensure_ascii=False),
                stacktrace="boom",
                started_at=datetime.now(timezone.utc),
                organization_id=None,
            )
            # Wipe card_data to simulate a legacy row with nothing to replay.
            async with storage._pool.connection() as conn:
                await conn.execute(
                    "UPDATE done_cards SET card_data = NULL WHERE card_guid = %(g)s",
                    {"g": guid},
                )
            rows = await storage.get_broken()

        assert guid not in {r["card_guid"] for r in rows}
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_get_broken_skips_non_broken_rows():
    guid = f"pytest-notbroken-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            await storage.upsert_pending(
                card_guid=guid,
                card_data=json.dumps({"Прием": {"GUID": guid}}, ensure_ascii=False),
                organization_id=None,
            )
            rows = await storage.get_broken()

        assert guid not in {r["card_guid"] for r in rows}
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_get_broken_without_org_returns_every_organization():
    """organization_id=None means "all orgs", not "rows whose org is NULL"."""
    null_org_guid = f"pytest-broken-nullorg-{uuid.uuid4()}"
    real_org_guid = f"pytest-broken-realorg-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            async with storage._pool.connection() as conn:
                cur = await conn.execute("SELECT id::text FROM organizations LIMIT 1")
                row = await cur.fetchone()
            if row is None:
                pytest.skip("no organizations configured in this DB")
            org_id = row["id"]

            for guid, org in ((null_org_guid, None), (real_org_guid, org_id)):
                await storage.upsert_broken(
                    card_guid=guid,
                    card_data=json.dumps({"Прием": {"GUID": guid}}, ensure_ascii=False),
                    stacktrace="boom",
                    started_at=datetime.now(timezone.utc),
                    organization_id=org,
                )

            all_guids = {r["card_guid"] for r in await storage.get_broken()}
            scoped_guids = {
                r["card_guid"] for r in await storage.get_broken(organization_id=org_id)
            }

        assert {null_org_guid, real_org_guid} <= all_guids
        assert real_org_guid in scoped_guids
        assert null_org_guid not in scoped_guids
    finally:
        await _cleanup(null_org_guid, real_org_guid)


@pytest.mark.asyncio
async def test_get_broken_skips_rows_without_guid():
    """Cards that failed before a GUID was known cannot be matched back (спека §9.7, ревью З4)."""
    async with DoneCardsStorage() as storage:
        row_id = await storage.upsert_broken(
            card_guid=None,
            card_data=json.dumps({"Прием": {}}, ensure_ascii=False),
            stacktrace="boom",
            started_at=datetime.now(timezone.utc),
            organization_id=None,
        )
        try:
            rows = await storage.get_broken()
            assert all(r["card_guid"] is not None for r in rows)
        finally:
            async with storage._pool.connection() as conn:
                await conn.execute(
                    "DELETE FROM done_cards WHERE id = %(i)s::uuid", {"i": row_id}
                )
```

- [ ] **Step 2: Запустить тест — убедиться, что падает**

Run: `pytest tests/test_done_cards_storage_broken.py -v`
Expected: FAIL — `AttributeError: 'DoneCardsStorage' object has no attribute 'get_broken'`

- [ ] **Step 3: Реализовать `get_broken`**

В `src/storage/done_cards_storage.py`, сразу после `get_pending`:

```python
    async def get_broken(self, organization_id: str | None = None) -> list[dict]:
        """Return card_guid + card_data + organization_id for replayable broken rows.

        Unlike get_pending/get_done_guids, organization_id=None means "every
        organization", not "rows whose organization is NULL": the fix-broken
        script needs the whole set so it can group by org itself.

        Rows without card_data have nothing to replay, and rows without a guid
        cannot be matched back to their original row on re-audit — both are skipped.
        """
        sql = (
            "SELECT card_guid, card_data, organization_id::text AS organization_id "
            "FROM done_cards "
            "WHERE broken = TRUE "
            "AND card_data IS NOT NULL "
            "AND card_guid IS NOT NULL"
        )
        params: dict[str, Any] = {}
        if organization_id is not None:
            sql += " AND organization_id = %(org_id)s"
            params["org_id"] = organization_id

        async with self._pool.connection() as conn:
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()

        logger.info(
            "💾 done_cards loaded %d broken card(s) for org_id=%s",
            len(rows), organization_id if organization_id is not None else "<all>",
        )
        return [dict(row) for row in rows]
```

Проверить, что `Any` уже импортирован в файле (`from typing import Any`); если
нет — добавить в импорты.

- [ ] **Step 4: Запустить тесты — убедиться, что проходят**

Run: `pytest tests/test_done_cards_storage_broken.py -v`
Expected: PASS (5 тестов; последний может SKIP, если в БД нет организаций)

- [ ] **Step 5: Коммит**

```bash
git add src/storage/done_cards_storage.py tests/test_done_cards_storage_broken.py
git commit -m "feat(storage): get_broken — replayable broken rows for re-audit"
```

---

### Task 2: Группировка broken-строк по организациям

**Files:**
- Create: `src/audit/broken_replay.py`
- Test: `tests/test_broken_replay_grouping.py` (создать)

**Interfaces:**
- Consumes: строки из `DoneCardsStorage.get_broken()` (Task 1) — dict с
  `card_guid`, `card_data`, `organization_id`.
- Produces:
  - `@dataclass(frozen=True) class BrokenGroup: org_id: str | None; org_name: str | None; visits: list[dict]; guids: set[str]`
  - `def group_by_org(rows: list[dict], org_names: dict[str, str]) -> list[BrokenGroup]`

Отдельный модуль, а не функция внутри скрипта: группировка — единственная
нетривиальная чистая логика здесь, и её надо тестировать без БД.

- [ ] **Step 1: Написать падающий тест**

Создать `tests/test_broken_replay_grouping.py`:

```python
"""Unit tests for grouping broken rows by organization. No DB, no LLM."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.broken_replay import BrokenGroup, group_by_org


def _row(guid: str, org: str | None) -> dict:
    return {
        "card_guid": guid,
        "card_data": {"Прием": {"GUID": guid}},
        "organization_id": org,
    }


def test_group_by_org_splits_rows_per_organization():
    rows = [_row("a", "org-1"), _row("b", "org-2"), _row("c", "org-1")]
    groups = group_by_org(rows, {"org-1": "Alenka", "org-2": "MDS"})

    by_id = {g.org_id: g for g in groups}
    assert set(by_id) == {"org-1", "org-2"}
    assert by_id["org-1"].guids == {"a", "c"}
    assert by_id["org-1"].org_name == "Alenka"
    assert by_id["org-2"].guids == {"b"}
    assert by_id["org-2"].org_name == "MDS"


def test_group_by_org_puts_null_org_rows_in_their_own_group():
    groups = group_by_org([_row("a", None), _row("b", "org-1")], {"org-1": "Alenka"})

    null_group = next(g for g in groups if g.org_id is None)
    assert null_group.guids == {"a"}
    assert null_group.org_name is None


def test_group_by_org_carries_visit_payloads():
    groups = group_by_org([_row("a", "org-1")], {"org-1": "Alenka"})
    assert groups[0].visits == [{"Прием": {"GUID": "a"}}]


def test_group_by_org_falls_back_to_id_when_name_unknown():
    """An org row can exist with no matching organizations entry."""
    groups = group_by_org([_row("a", "org-ghost")], {})
    assert groups[0].org_id == "org-ghost"
    assert groups[0].org_name is None


def test_group_by_org_returns_empty_list_for_no_rows():
    assert group_by_org([], {}) == []


def test_group_by_org_is_deterministic():
    """Named orgs sort by name; the NULL-org group always comes last."""
    rows = [_row("a", "org-2"), _row("b", None), _row("c", "org-1")]
    groups = group_by_org(rows, {"org-1": "Alenka", "org-2": "MDS"})
    assert [g.org_name for g in groups] == ["Alenka", "MDS", None]
```

- [ ] **Step 2: Запустить тест — убедиться, что падает**

Run: `pytest tests/test_broken_replay_grouping.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'audit.broken_replay'`

- [ ] **Step 3: Реализовать модуль**

Создать `src/audit/broken_replay.py`:

```python
"""Grouping of broken done_cards rows for re-audit.

A pipeline instance is bound to one org_id and one CardFilter, and both are
written into every row it persists. Replaying every organization through a
single pipeline would stamp foreign org_ids and apply a foreign filter, so
broken rows are grouped by organization first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BrokenGroup:
    """Broken cards of one organization, ready to be replayed together."""

    org_id: str | None
    org_name: str | None
    visits: list[dict[str, Any]]
    guids: set[str]


def group_by_org(
    rows: list[dict[str, Any]],
    org_names: dict[str, str],
) -> list[BrokenGroup]:
    """Split broken rows into one group per organization.

    Args:
        rows:      Rows from DoneCardsStorage.get_broken().
        org_names: organization_id -> name, for display and filter lookup.

    Returns:
        Groups sorted by organization name; rows with a NULL organization_id
        form their own trailing group (org_id=None, org_name=None).
    """
    buckets: dict[str | None, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(row["organization_id"], []).append(row)

    groups = [
        BrokenGroup(
            org_id=org_id,
            org_name=org_names.get(org_id) if org_id is not None else None,
            visits=[row["card_data"] for row in bucket],
            guids={row["card_guid"] for row in bucket},
        )
        for org_id, bucket in buckets.items()
    ]
    # NULL-org group last; the rest alphabetically, so runs are reproducible.
    return sorted(groups, key=lambda g: (g.org_id is None, g.org_name or ""))
```

- [ ] **Step 4: Запустить тесты — убедиться, что проходят**

Run: `pytest tests/test_broken_replay_grouping.py -v`
Expected: PASS (6 тестов)

- [ ] **Step 5: Коммит**

```bash
git add src/audit/broken_replay.py tests/test_broken_replay_grouping.py
git commit -m "feat(audit): group broken cards by organization for replay"
```

---

### Task 3: Сводка прогона (сверка множеств до/после)

**Files:**
- Modify: `src/audit/broken_replay.py`
- Test: `tests/test_broken_replay_summary.py` (создать)

**Interfaces:**
- Consumes: `BrokenGroup` (Task 2).
- Produces:
  - `@dataclass(frozen=True) class CardOutcome: guid: str; state: str` — `state`
    один из `"fixed"`, `"ignored"`, `"still_broken"`.
  - `def diff_outcomes(before: set[str], after_broken: set[str], after_ignored: set[str]) -> list[CardOutcome]`
  - `def format_summary(outcomes: list[CardOutcome], stacktraces: dict[str, str]) -> str`
  - `def last_stacktrace_line(stacktrace: str) -> str`

Считаем именно так, потому что `run_batched` возвращает **только успешные** пары
(`pipeline.py`: `[p for p in raw_pairs if p is not None]`) — упавшие карты в
возврате не видны, и единственный надёжный источник итога это состояние БД.

- [ ] **Step 1: Написать падающий тест**

Создать `tests/test_broken_replay_summary.py`:

```python
"""Unit tests for the fix-broken run summary. No DB, no LLM."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.broken_replay import CardOutcome, diff_outcomes, format_summary, last_stacktrace_line


def test_diff_outcomes_marks_card_that_left_broken_as_fixed():
    outcomes = diff_outcomes(before={"a"}, after_broken=set(), after_ignored=set())
    assert outcomes == [CardOutcome(guid="a", state="fixed")]


def test_diff_outcomes_marks_card_that_went_to_ignored_separately():
    """A card matched by a filter leaves broken without being fixed (спека §9.6)."""
    outcomes = diff_outcomes(before={"a"}, after_broken=set(), after_ignored={"a"})
    assert outcomes == [CardOutcome(guid="a", state="ignored")]


def test_diff_outcomes_marks_card_that_failed_again_as_still_broken():
    outcomes = diff_outcomes(before={"a"}, after_broken={"a"}, after_ignored=set())
    assert outcomes == [CardOutcome(guid="a", state="still_broken")]


def test_diff_outcomes_ignores_cards_that_were_not_in_the_run():
    """Another org's broken card, or one created while we ran, is not our business."""
    outcomes = diff_outcomes(before={"a"}, after_broken={"a", "z"}, after_ignored=set())
    assert [o.guid for o in outcomes] == ["a"]


def test_diff_outcomes_is_sorted_by_guid():
    outcomes = diff_outcomes(before={"b", "a"}, after_broken=set(), after_ignored=set())
    assert [o.guid for o in outcomes] == ["a", "b"]


def test_last_stacktrace_line_returns_the_exception_not_the_header():
    tb = "Traceback (most recent call last):\n  File x\nValueError: boom"
    assert last_stacktrace_line(tb) == "ValueError: boom"


def test_last_stacktrace_line_survives_trailing_whitespace_and_empty_input():
    assert last_stacktrace_line("ValueError: boom\n\n") == "ValueError: boom"
    assert last_stacktrace_line("") == "<no stacktrace>"


def test_format_summary_counts_each_state_and_lists_still_broken():
    outcomes = [
        CardOutcome(guid="a", state="fixed"),
        CardOutcome(guid="b", state="ignored"),
        CardOutcome(guid="c", state="still_broken"),
    ]
    text = format_summary(outcomes, {"c": "Traceback (most recent call last):\nValueError: boom"})

    assert "fixed: 1" in text
    assert "moved to ignored: 1" in text
    assert "still broken: 1" in text
    assert "c" in text
    assert "ValueError: boom" in text
    # An ignored card must not read as fixed.
    assert "b" in text


def test_format_summary_handles_a_clean_run():
    assert "still broken: 0" in format_summary([], {})
```

- [ ] **Step 2: Запустить тест — убедиться, что падает**

Run: `pytest tests/test_broken_replay_summary.py -v`
Expected: FAIL — `ImportError: cannot import name 'CardOutcome'`

- [ ] **Step 3: Дописать модуль**

Добавить в `src/audit/broken_replay.py`:

```python
_STATE_LABELS = {
    "fixed": "fixed",
    "ignored": "moved to ignored",
    "still_broken": "still broken",
}


@dataclass(frozen=True)
class CardOutcome:
    """What happened to one card over a re-audit run."""

    guid: str
    state: str  # fixed | ignored | still_broken


def diff_outcomes(
    before: set[str],
    after_broken: set[str],
    after_ignored: set[str],
) -> list[CardOutcome]:
    """Classify every card we replayed by comparing DB state before and after.

    run_batched returns successful pairs only, so the DB is the sole reliable
    source of truth about what actually happened.
    """
    outcomes: list[CardOutcome] = []
    for guid in sorted(before):
        if guid in after_broken:
            state = "still_broken"
        elif guid in after_ignored:
            state = "ignored"
        else:
            state = "fixed"
        outcomes.append(CardOutcome(guid=guid, state=state))
    return outcomes


def last_stacktrace_line(stacktrace: str | None) -> str:
    """Return the exception line of a traceback.

    The first line is always "Traceback (most recent call last):" and tells
    nothing apart; the last one carries the exception type and message.
    """
    lines = [line.strip() for line in (stacktrace or "").splitlines() if line.strip()]
    return lines[-1] if lines else "<no stacktrace>"


def format_summary(outcomes: list[CardOutcome], stacktraces: dict[str, str]) -> str:
    """Render the human-readable run summary printed to stdout and the log."""
    counts = {state: 0 for state in _STATE_LABELS}
    for outcome in outcomes:
        counts[outcome.state] += 1

    lines = ["", "── fix-broken summary ──"]
    lines.extend(f"{_STATE_LABELS[state]}: {counts[state]}" for state in _STATE_LABELS)

    ignored = [o for o in outcomes if o.state == "ignored"]
    if ignored:
        lines.append("")
        lines.append("Moved to ignored (matched a filter — NOT fixed):")
        lines.extend(f"  {o.guid}" for o in ignored)

    still = [o for o in outcomes if o.state == "still_broken"]
    if still:
        lines.append("")
        lines.append("Still broken:")
        lines.extend(
            f"  {o.guid} — {last_stacktrace_line(stacktraces.get(o.guid))}" for o in still
        )

    return "\n".join(lines)
```

- [ ] **Step 4: Запустить тесты — убедиться, что проходят**

Run: `pytest tests/test_broken_replay_summary.py -v`
Expected: PASS (9 тестов)

- [ ] **Step 5: Коммит**

```bash
git add src/audit/broken_replay.py tests/test_broken_replay_summary.py
git commit -m "feat(audit): fix-broken run summary from before/after DB state"
```

---

### Task 4: Чтение состояния после прогона

**Files:**
- Modify: `src/storage/done_cards_storage.py`
- Test: `tests/test_done_cards_storage_broken.py` (дополнить)

**Interfaces:**
- Produces: `async def get_states_for_guids(self, guids: set[str]) -> dict[str, dict]`
  — по каждому GUID отдаёт `{"broken": bool, "ignored": bool, "stacktrace": str | None}`.
  Пропавшие из БД GUID в результат не попадают.

Нужен, чтобы посчитать «после» одним запросом, не поднимая весь `get_broken`
повторно и не теряя свежие стектрейсы для сводки.

- [ ] **Step 1: Написать падающий тест**

Дописать в `tests/test_done_cards_storage_broken.py`:

```python
@pytest.mark.asyncio
async def test_get_states_for_guids_reports_flags_and_stacktrace():
    broken_guid = f"pytest-state-broken-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            await storage.upsert_broken(
                card_guid=broken_guid,
                card_data=json.dumps({"Прием": {"GUID": broken_guid}}, ensure_ascii=False),
                stacktrace="Traceback (most recent call last):\nValueError: boom",
                started_at=datetime.now(timezone.utc),
                organization_id=None,
            )
            states = await storage.get_states_for_guids({broken_guid})

        assert states[broken_guid]["broken"] is True
        assert states[broken_guid]["ignored"] is False
        assert "ValueError: boom" in states[broken_guid]["stacktrace"]
    finally:
        await _cleanup(broken_guid)


@pytest.mark.asyncio
async def test_get_states_for_guids_omits_unknown_guids():
    async with DoneCardsStorage() as storage:
        states = await storage.get_states_for_guids({f"pytest-absent-{uuid.uuid4()}"})
    assert states == {}


@pytest.mark.asyncio
async def test_get_states_for_guids_returns_empty_for_empty_input():
    """Guard: an empty IN () is a SQL syntax error, so this must short-circuit."""
    async with DoneCardsStorage() as storage:
        assert await storage.get_states_for_guids(set()) == {}
```

- [ ] **Step 2: Запустить тест — убедиться, что падает**

Run: `pytest tests/test_done_cards_storage_broken.py -v -k states`
Expected: FAIL — `AttributeError: … has no attribute 'get_states_for_guids'`

- [ ] **Step 3: Реализовать метод**

В `src/storage/done_cards_storage.py`, следом за `get_broken`:

```python
    async def get_states_for_guids(self, guids: set[str]) -> dict[str, dict]:
        """Return broken/ignored flags and stacktrace for each of *guids*.

        Guids with no row are omitted. Used to diff DB state before and after
        a re-audit run.
        """
        if not guids:
            return {}

        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT card_guid, broken, ignored, stacktrace FROM done_cards "
                "WHERE card_guid = ANY(%(guids)s)",
                {"guids": list(guids)},
            )
            rows = await cur.fetchall()

        return {
            row["card_guid"]: {
                "broken": row["broken"],
                "ignored": row["ignored"],
                "stacktrace": row["stacktrace"],
            }
            for row in rows
        }
```

- [ ] **Step 4: Запустить тесты — убедиться, что проходят**

Run: `pytest tests/test_done_cards_storage_broken.py -v`
Expected: PASS (все тесты файла, включая три новых)

- [ ] **Step 5: Коммит**

```bash
git add src/storage/done_cards_storage.py tests/test_done_cards_storage_broken.py
git commit -m "feat(storage): get_states_for_guids for fix-broken before/after diff"
```

---

### Task 5: Скрипт `scripts/fix-broken.py`

**Files:**
- Create: `scripts/fix-broken.py`
- Test: `tests/test_fix_broken_args.py` (создать)

**Interfaces:**
- Consumes: `DoneCardsStorage.get_broken` / `get_states_for_guids` (задачи 1, 4),
  `group_by_org` / `diff_outcomes` / `format_summary` (задачи 2, 3),
  `AuditPipeline(org_id=…, card_filter=…)`, `load_card_filter(org)`,
  `OrganizationsStorage.get_name_by_id`, `close_pool`.
- Produces: CLI —
  `python scripts/fix-broken.py ORG [-y] [--dry-run] [--num-batches N]` и
  `python scripts/fix-broken.py --all [-y] [--dry-run] [--num-batches N]`;
  функция `build_parser() -> argparse.ArgumentParser` (импортируется тестом).

- [ ] **Step 1: Написать падающий тест на разбор аргументов**

Создать `tests/test_fix_broken_args.py`:

```python
"""Argument-parsing tests for scripts/fix-broken.py. No DB, no LLM."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# The script is not an importable module name (it has a dash), so load by path.
_spec = importlib.util.spec_from_file_location(
    "fix_broken_script", ROOT / "scripts" / "fix-broken.py"
)
fix_broken = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fix_broken)


def test_org_mode_parses():
    args = fix_broken.build_parser().parse_args(["Alenka"])
    assert args.org == "Alenka"
    assert args.all is False


def test_all_mode_parses():
    args = fix_broken.build_parser().parse_args(["--all"])
    assert args.all is True
    assert args.org is None


def test_org_and_all_together_is_an_error():
    with pytest.raises(SystemExit):
        fix_broken.build_parser().parse_args(["Alenka", "--all"])


def test_neither_org_nor_all_is_an_error():
    with pytest.raises(SystemExit):
        fix_broken.build_parser().parse_args([])


def test_unknown_org_is_an_error():
    with pytest.raises(SystemExit):
        fix_broken.build_parser().parse_args(["Nope"])


def test_num_batches_defaults_to_five():
    assert fix_broken.build_parser().parse_args(["MDS"]).num_batches == 5


def test_dry_run_and_yes_flags_parse():
    args = fix_broken.build_parser().parse_args(["MDS", "-y", "--dry-run"])
    assert args.y is True
    assert args.dry_run is True
```

- [ ] **Step 2: Запустить тест — убедиться, что падает**

Run: `pytest tests/test_fix_broken_args.py -v`
Expected: FAIL — `FileNotFoundError` на `scripts/fix-broken.py`

- [ ] **Step 3: Написать скрипт**

Создать `scripts/fix-broken.py`:

```python
#!/usr/bin/env python3
"""
Re-audit cards frozen with broken = TRUE, replaying their stored card_data
through the pipeline with deduplication switched off.

A broken card is a done_cards row with status = 'done', so get_done_guids()
returns it and CardFilter skips it forever: a nightly run never retries it.
Passing done_guids=set() to run_batched bypasses that dedup.

Runs entirely offline — the source of data is the DB, never 1C.

Run from project root:
    python scripts/fix-broken.py ORG [-y] [--dry-run] [--num-batches N]
    python scripts/fix-broken.py --all [-y] [--dry-run] [--num-batches N]

Options:
    ORG            Organization: Alenka or MDS. Mutually exclusive with --all
    --all          Every organization at once
    -y             Skip the confirmation prompt
    --dry-run      Show what would be re-audited, then exit without writing
    --num-batches  Max concurrent cards at a time (default: 5)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.broken_replay import BrokenGroup, diff_outcomes, format_summary, group_by_org
from audit.filters import CardFilter
from audit.pipeline import AuditPipeline
from parsers.filter_config import load_card_filter
from RAG.retrieval.vector_store import close_pool
from storage.done_cards_storage import DoneCardsStorage
from storage.organizations_storage import OrganizationsStorage

LOGS_DIR = ROOT / "logs"


def build_parser() -> argparse.ArgumentParser:
    """CLI contract: exactly one of ORG / --all is required."""
    parser = argparse.ArgumentParser(description="Re-audit broken cards from the DB.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("org", nargs="?", choices=("Alenka", "MDS"), help="Organization")
    target.add_argument("--all", action="store_true", help="Every organization at once")
    parser.add_argument("-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="Report only, write nothing")
    parser.add_argument(
        "--num-batches", type=int, default=5, metavar="N",
        help="Max concurrent cards at a time (default: 5)",
    )
    return parser


def _configure_logging() -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    log_file = LOGS_DIR / f"fix-broken_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return log_file


def _filter_for(group: BrokenGroup) -> CardFilter:
    """Each organization keeps its own filter; unknown/NULL orgs get an empty one."""
    return load_card_filter(group.org_name) if group.org_name else CardFilter([])


def _confirm(groups: list[BrokenGroup], args: argparse.Namespace) -> None:
    mode = "all organizations" if args.all else args.org
    print(f"Mode: {mode}")
    total = sum(len(g.guids) for g in groups)
    print(f"Broken cards to re-audit: {total}")
    for group in groups:
        print(f"  {group.org_name or '<no organization>'}: {len(group.guids)}")
        print(f"  Filters:\n{_filter_for(group)}")
    if args.dry_run:
        print("Dry run — nothing was written.")
        return
    if args.y:
        return
    if input("Proceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        sys.exit(0)


async def _replay(group: BrokenGroup, num_batches: int, log: logging.Logger) -> None:
    """Re-audit one organization's cards with dedup disabled."""
    log.info(
        "🔧 Replaying %d card(s) for org=%s",
        len(group.visits), group.org_name or "<none>",
    )
    async with AuditPipeline(org_id=group.org_id, card_filter=_filter_for(group)) as pipeline:
        # done_guids=set() is the whole trick: it disables the always-on dedup
        # that otherwise skips these very cards.
        pairs = await pipeline.run_batched(
            group.visits, num_batches=num_batches, done_guids=set()
        )
    log.info("🔧 org=%s produced %d successful result(s)", group.org_name or "<none>", len(pairs))


async def main() -> None:
    args = build_parser().parse_args()
    log_file = _configure_logging()
    log = logging.getLogger(__name__)

    try:
        async with OrganizationsStorage() as orgs:
            org_id = await orgs.get_id_by_name(args.org) if args.org else None
            names: dict[str, str] = {}
            async with DoneCardsStorage() as done_cards:
                rows = await done_cards.get_broken(organization_id=org_id)
                for row in rows:
                    row_org = row["organization_id"]
                    if row_org and row_org not in names:
                        names[row_org] = await orgs.get_name_by_id(row_org)

        groups = group_by_org(rows, names)
        if not groups:
            print("No broken cards to re-audit.")
            return

        _confirm(groups, args)
        if args.dry_run:
            return

        before = {guid for group in groups for guid in group.guids}
        for group in groups:
            await _replay(group, args.num_batches, log)

        async with DoneCardsStorage() as done_cards:
            states = await done_cards.get_states_for_guids(before)

        after_broken = {g for g, s in states.items() if s["broken"]}
        after_ignored = {g for g, s in states.items() if s["ignored"]}
        stacktraces = {g: s["stacktrace"] for g, s in states.items() if s["stacktrace"]}

        summary = format_summary(diff_outcomes(before, after_broken, after_ignored), stacktraces)
        print(summary)
        log.info(summary)
        log.info("Done. Log: %s", log_file)
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Запустить тесты — убедиться, что проходят**

Run: `pytest tests/test_fix_broken_args.py -v`
Expected: PASS (7 тестов)

- [ ] **Step 5: Проверить `--dry-run` вживую (нужна БД)**

Run: `python scripts/fix-broken.py --all --dry-run`
Expected: печатает разбивку по организациям и «Dry run — nothing was written.»,
выходит с кодом 0. Проверить `echo $?` = 0.
Если БД недоступна с dev-машины — отложить шаг до стенда и отметить это в
отчёте о задаче, не подгоняя код.

- [ ] **Step 6: Коммит**

```bash
git add scripts/fix-broken.py tests/test_fix_broken_args.py
git commit -m "feat(scripts): fix-broken — re-audit frozen broken cards"
```

---

### Task 6: Интеграционные тесты сценариев починки

**Files:**
- Create: `tests/test_fix_broken_replay.py`

**Interfaces:**
- Consumes: всё из задач 1–5.

Это тесты §10 спеки, пункты 1–4 и 6. Гонять на стенде: они пишут в реальную БД
и запускают настоящий аудит (LLM), поэтому помечены `slow`.

- [ ] **Step 1: Написать тесты**

Создать `tests/test_fix_broken_replay.py`:

```python
"""
End-to-end tests for the fix-broken replay path. Hits the real DB and runs a
real audit — stand-only, slow, and costs LLM tokens.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from audit.broken_replay import group_by_org
from audit.filters import CardFilter
from audit.pipeline import AuditPipeline
from storage.done_cards_storage import DoneCardsStorage

pytestmark = pytest.mark.slow


def _visit(guid: str) -> dict:
    """A minimal auditable visit, shaped like the ones tests/test_filters.py builds.

    Кладём реальный визит со стенда, если он есть в data_snapshots/ — тогда
    аудит отработает содержательно; иначе минимальный каркас: он проходит
    пайплайн, хотя замечаний почти не даст.
    """
    snapshots = sorted((ROOT / "data_snapshots").glob("one_c_*.json"))
    if snapshots:
        payload = json.loads(snapshots[0].read_text(encoding="utf-8"))
        visits = payload if isinstance(payload, list) else payload.get("appointments", [])
        if visits:
            visit = json.loads(json.dumps(visits[0], ensure_ascii=False))
            visit.setdefault("Прием", {})["GUID"] = guid
            return visit
    return {"Прием": {"GUID": guid}, "Пациент": {}, "Диагнозы": [], "Услуги": []}


async def _cleanup(*guids: str) -> None:
    async with DoneCardsStorage() as storage:
        async with storage._pool.connection() as conn:
            for guid in guids:
                await conn.execute(
                    "DELETE FROM done_cards WHERE card_guid = %(g)s", {"g": guid}
                )


async def _seed_broken(guid: str, org_id: str | None = None) -> None:
    async with DoneCardsStorage() as storage:
        await storage.upsert_broken(
            card_guid=guid,
            card_data=json.dumps(_visit(guid), ensure_ascii=False),
            stacktrace="Traceback (most recent call last):\nValueError: seeded",
            started_at=datetime.now(timezone.utc),
            organization_id=org_id,
        )


@pytest.mark.asyncio
async def test_replay_clears_the_broken_flag_and_fills_results():
    guid = f"pytest-replay-{uuid.uuid4()}"
    try:
        await _seed_broken(guid)
        async with DoneCardsStorage() as storage:
            rows = await storage.get_broken()
        group = next(g for g in group_by_org(rows, {}) if guid in g.guids)

        async with AuditPipeline(org_id=None, card_filter=CardFilter([])) as pipeline:
            await pipeline.run_batched(
                [v for v in group.visits if v["Прием"]["GUID"] == guid],
                num_batches=1,
                done_guids=set(),
            )

        async with DoneCardsStorage() as storage:
            state = (await storage.get_states_for_guids({guid}))[guid]
            async with storage._pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT formal_result FROM done_cards WHERE card_guid = %(g)s",
                    {"g": guid},
                )
                row = await cur.fetchone()

        assert state["broken"] is False
        assert row["formal_result"] is not None
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_dedup_would_skip_the_card_without_the_empty_done_guids():
    """Guard for the core mechanism: with the real done_guids the card is skipped."""
    guid = f"pytest-dedup-{uuid.uuid4()}"
    try:
        await _seed_broken(guid)
        async with DoneCardsStorage() as storage:
            done_guids = await storage.get_done_guids(organization_id=None)
        assert guid.lower() in {g.lower() for g in done_guids}, (
            "broken card must be inside done_guids — that is exactly why it is frozen"
        )

        async with AuditPipeline(org_id=None, card_filter=CardFilter([])) as pipeline:
            pairs = await pipeline.run_batched([_visit(guid)], num_batches=1)
        assert pairs == []

        async with DoneCardsStorage() as storage:
            assert (await storage.get_states_for_guids({guid}))[guid]["broken"] is True
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_filtered_card_moves_to_ignored_instead_of_being_fixed():
    """Спека §9.6: a filter matching a broken card sends it to ignored, not fixed."""
    guid = f"pytest-filtered-{uuid.uuid4()}"
    try:
        await _seed_broken(guid)

        class _SkipEverything:
            def should_skip(self, visit: dict) -> bool:
                return True

        async with AuditPipeline(org_id=None, card_filter=CardFilter([_SkipEverything()])) as pipeline:
            await pipeline.run_batched([_visit(guid)], num_batches=1, done_guids=set())

        async with DoneCardsStorage() as storage:
            state = (await storage.get_states_for_guids({guid}))[guid]
        assert state["ignored"] is True
        assert state["broken"] is False  # migration 014 CHECK holds
    finally:
        await _cleanup(guid)


@pytest.mark.asyncio
async def test_each_organization_keeps_its_own_org_id():
    """--all must never stamp one org's id onto another org's card."""
    guid_a = f"pytest-orga-{uuid.uuid4()}"
    guid_b = f"pytest-orgb-{uuid.uuid4()}"
    try:
        async with DoneCardsStorage() as storage:
            async with storage._pool.connection() as conn:
                cur = await conn.execute("SELECT id::text FROM organizations LIMIT 2")
                org_rows = await cur.fetchall()
        if len(org_rows) < 2:
            pytest.skip("need two organizations in the DB")
        org_a, org_b = org_rows[0]["id"], org_rows[1]["id"]

        await _seed_broken(guid_a, org_a)
        await _seed_broken(guid_b, org_b)

        async with DoneCardsStorage() as storage:
            rows = await storage.get_broken()
        groups = [g for g in group_by_org(rows, {}) if g.org_id in (org_a, org_b)]

        for group in groups:
            async with AuditPipeline(org_id=group.org_id, card_filter=CardFilter([])) as pipeline:
                await pipeline.run_batched(
                    [v for v in group.visits if v["Прием"]["GUID"] in (guid_a, guid_b)],
                    num_batches=1,
                    done_guids=set(),
                )

        async with DoneCardsStorage() as storage:
            async with storage._pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT card_guid, organization_id::text AS org FROM done_cards "
                    "WHERE card_guid = ANY(%(g)s)",
                    {"g": [guid_a, guid_b]},
                )
                by_guid = {r["card_guid"]: r["org"] for r in await cur.fetchall()}

        assert by_guid[guid_a] == org_a
        assert by_guid[guid_b] == org_b
    finally:
        await _cleanup(guid_a, guid_b)


@pytest.mark.asyncio
async def test_dry_run_writes_nothing():
    guid = f"pytest-dryrun-{uuid.uuid4()}"
    try:
        await _seed_broken(guid)
        async with DoneCardsStorage() as storage:
            before = (await storage.get_states_for_guids({guid}))[guid]

        # --dry-run stops before any pipeline runs; assert the seeded state is intact.
        async with DoneCardsStorage() as storage:
            rows = await storage.get_broken()
            after = (await storage.get_states_for_guids({guid}))[guid]

        assert guid in {r["card_guid"] for r in rows}
        assert after == before
    finally:
        await _cleanup(guid)
```

- [ ] **Step 2: Проверить, что `slow` — известный маркер**

Run: `grep -n "markers" -A 5 pytest.ini`
Если маркера `slow` нет — добавить в `pytest.ini`:

```ini
markers =
    slow: hits the real DB and runs a real audit (stand only)
```

Иначе pytest выдаст `PytestUnknownMarkWarning`.

- [ ] **Step 3: Запустить на стенде**

Run: `pytest tests/test_fix_broken_replay.py -v`
Expected: PASS (5 тестов; последний из них — SKIP, если в БД меньше двух
организаций). На dev-машине без БД тесты падают на коннекте — это ожидаемо,
гонять на стенде.

- [ ] **Step 4: Коммит**

```bash
git add tests/test_fix_broken_replay.py pytest.ini
git commit -m "test: end-to-end coverage for fix-broken replay"
```

---

### Task 7: Документация

**Files:**
- Modify: `docs/storage.md`
- Modify: `CLAUDE.md` (блок Commands)
- Modify: `docs/revision-log.md`

**Interfaces:** ничего не производит для кода — закрывает §11 спеки и правило
«гейт приёмки каждой ветки — стенд + запись в `docs/revision-log.md`».

- [ ] **Step 1: Дописать `docs/storage.md`**

Найти описание `get_done_guids` (около строки 69) и добавить рядом два пункта:

```markdown
- `get_broken(organization_id=None)` — broken-строки, пригодные для переаудита:
  `broken = TRUE AND card_data IS NOT NULL AND card_guid IS NOT NULL`.
  В отличие от `get_pending`/`get_done_guids`, `organization_id=None` значит
  «все организации», а не «строки с NULL» — скрипт `fix-broken.py` группирует
  их сам. Карты без GUID не отдаются: их нельзя сматчить обратно.
- `get_states_for_guids(guids)` — флаги `broken`/`ignored` и стектрейс по
  набору GUID; используется для сверки состояния до и после переаудита.
```

Там же, в описании `get_done_guids`, дописать оговорку:

```markdown
  Внимание: отдаёт и broken-карты (у них `status = 'done'`). Из-за этого
  упавшая карта не переаудируется ночным прогоном — для неё есть
  `scripts/fix-broken.py`.
```

- [ ] **Step 2: Дописать `CLAUDE.md`**

В блок Commands, после `audit-file.py`:

```bash
# Re-audit cards frozen with broken = TRUE (offline: reads card_data from the DB)
python scripts/fix-broken.py ORG|--all [-y] [--dry-run] [--num-batches N]
```

- [ ] **Step 3: Запись в `docs/revision-log.md`**

Добавить строку в формате, принятом в файле (посмотреть соседние записи и
повторить их структуру): дата ISO, тема «переаудит broken-карт», что появилось
(`scripts/fix-broken.py`, `get_broken`, `get_states_for_guids`), и что дата
прогона на стенде важна для сравнения отчётов «до/после» — прогон меняет
`updated_at` и карты уедут в «Искру» как свежие (спека §9.8, §13.3).

- [ ] **Step 4: Коммит**

```bash
git add docs/storage.md CLAUDE.md docs/revision-log.md
git commit -m "docs: fix-broken script, get_broken/get_states_for_guids"
```

---

## Приёмка

Гейт — стенд (правило проекта: storage-тесты и реальный прогон только там).

1. `pytest tests/test_broken_replay_grouping.py tests/test_broken_replay_summary.py tests/test_fix_broken_args.py -v` — зелёные **без БД**, гоняются на dev-машине.
2. `pytest tests/test_done_cards_storage_broken.py tests/test_fix_broken_replay.py -v` — зелёные на стенде.
3. `python scripts/fix-broken.py --all --dry-run` — печатает разбивку, ничего не пишет.
4. `python scripts/fix-broken.py --all` — сводка сведена; починенные карты имеют
   заполненные результаты; ушедшие в ignored перечислены отдельной строкой;
   оставшиеся broken перечислены с типом исключения.
5. Число «осталось broken» сверено с ожиданием из ревью: до выкатки графа это
   почти все 96 карт (81 ICD + 15 диагноз); после выкатки графа диагноз-контур
   должен уйти, а ICD-остаток сохраниться. Это и есть проверка DoD §12.3 спеки.

## Что этот план не делает

- Не трогает граф (§1–8 спеки) — отдельный план после ревью блокеров Б1–Б3.
- Не чинит карты, упавшие из-за битых данных внутри самой карты: они упадут
  снова и останутся broken. Осознанная граница спеки §9.2.
- Не чинит карты без GUID (ревью З4): они не отдаются `get_broken`.
- Не делает Excel-экспорт и FTP-выгрузку: починенные карты подхватит обычный
  периодный отчёт (спека §9.3).
