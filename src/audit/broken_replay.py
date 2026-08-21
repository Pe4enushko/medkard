"""Pure grouping and outcome reporting for broken-card re-audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

OutcomeState = Literal["fixed", "ignored", "still_broken"]

_STATE_LABELS: dict[OutcomeState, str] = {
    "fixed": "fixed",
    "ignored": "moved to ignored",
    "still_broken": "still broken",
}


@dataclass(frozen=True)
class BrokenGroup:
    """Broken cards belonging to one organization and pipeline configuration."""

    org_id: str | None
    org_name: str | None
    visits: list[dict[str, Any]]
    guids: set[str]


@dataclass(frozen=True)
class CardOutcome:
    """State of one card after a replay run."""

    guid: str
    state: OutcomeState


def group_by_org(
    rows: list[dict[str, Any]],
    org_names: dict[str, str],
) -> list[BrokenGroup]:
    """Split storage rows into deterministic organization-bound batches."""
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
    return sorted(
        groups,
        key=lambda group: (
            group.org_id is None,
            group.org_name or group.org_id or "",
        ),
    )


def diff_outcomes(
    before: set[str],
    after_broken: set[str],
    after_ignored: set[str],
) -> list[CardOutcome]:
    """Classify replayed cards using their persisted state after the run."""
    outcomes: list[CardOutcome] = []
    for guid in sorted(before):
        if guid in after_broken:
            state: OutcomeState = "still_broken"
        elif guid in after_ignored:
            state = "ignored"
        else:
            state = "fixed"
        outcomes.append(CardOutcome(guid=guid, state=state))
    return outcomes


def last_stacktrace_line(stacktrace: str | None) -> str:
    """Return the exception line rather than the generic traceback header."""
    lines = [line.strip() for line in (stacktrace or "").splitlines() if line.strip()]
    return lines[-1] if lines else "<no stacktrace>"


def format_summary(
    outcomes: list[CardOutcome],
    stacktraces: dict[str, str],
) -> str:
    """Render the before/after reconciliation for stdout and the run log."""
    counts = {state: 0 for state in _STATE_LABELS}
    for outcome in outcomes:
        counts[outcome.state] += 1

    lines = ["", "── fix-broken summary ──"]
    lines.extend(f"{_STATE_LABELS[state]}: {counts[state]}" for state in _STATE_LABELS)

    ignored = [outcome for outcome in outcomes if outcome.state == "ignored"]
    if ignored:
        lines.extend(["", "Moved to ignored (matched a filter — NOT fixed):"])
        lines.extend(f"  {outcome.guid}" for outcome in ignored)

    still_broken = [outcome for outcome in outcomes if outcome.state == "still_broken"]
    if still_broken:
        lines.extend(["", "Still broken:"])
        lines.extend(
            f"  {outcome.guid} — {last_stacktrace_line(stacktraces.get(outcome.guid))}"
            for outcome in still_broken
        )

    return "\n".join(lines)
