"""An atomic rule parse failure must never pass a negative e2e assertion."""
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e2e/tests/audit"))
from harness import _FormalCallWatch


@pytest.mark.parametrize("message", [
    "failed to parse JSON response: invalid JSON",
    "failed to parse rule verdict: invalid JSON",
])
def test_parse_failure_is_observed(message):
    watch = _FormalCallWatch()
    watch.emit(logging.LogRecord("LLM.validations", logging.ERROR, "", 0, message, (), None))
    assert watch.parse_failed
    watch.reset()
    assert not watch.parse_failed


def test_successful_formal_tally_is_not_a_parse_failure():
    watch = _FormalCallWatch()
    watch.emit(logging.LogRecord(
        "audit.formal_structure.validator", logging.INFO, "", 0,
        "[formal] LLM returned 0 finding(s), tokens=100: []", (), None,
    ))
    assert not watch.parse_failed
    assert watch.tally == "0 finding(s), tokens=100"
