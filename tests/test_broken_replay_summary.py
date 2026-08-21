"""Unit tests for the fix-broken before/after summary."""

from audit.broken_replay import (
    CardOutcome,
    diff_outcomes,
    format_summary,
    last_stacktrace_line,
)


def test_diff_outcomes_classifies_and_sorts_only_replayed_cards() -> None:
    outcomes = diff_outcomes(
        before={"c", "a", "b"},
        after_broken={"c", "not-in-run"},
        after_ignored={"b"},
    )

    assert outcomes == [
        CardOutcome(guid="a", state="fixed"),
        CardOutcome(guid="b", state="ignored"),
        CardOutcome(guid="c", state="still_broken"),
    ]


def test_last_stacktrace_line_returns_exception_and_handles_empty() -> None:
    traceback = "Traceback (most recent call last):\n  File x\nValueError: boom\n"
    assert last_stacktrace_line(traceback) == "ValueError: boom"
    assert last_stacktrace_line(None) == "<no stacktrace>"


def test_format_summary_counts_and_explains_non_fixed_cards() -> None:
    outcomes = [
        CardOutcome(guid="a", state="fixed"),
        CardOutcome(guid="b", state="ignored"),
        CardOutcome(guid="c", state="still_broken"),
    ]

    rendered = format_summary(
        outcomes,
        {"c": "Traceback (most recent call last):\nValueError: boom"},
    )

    assert "fixed: 1" in rendered
    assert "moved to ignored: 1" in rendered
    assert "still broken: 1" in rendered
    assert "b" in rendered
    assert "c — ValueError: boom" in rendered


def test_format_summary_handles_empty_run() -> None:
    rendered = format_summary([], {})
    assert "fixed: 0" in rendered
    assert "still broken: 0" in rendered
