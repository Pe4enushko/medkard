"""Shared runner for diagnosis-graph regressions from ``eval_broken_cards``.

This is a component e2e boundary: real ClinicalRecs, Postgres, embeddings,
reranker (when configured), LLM calls, and the compiled diagnosis graph. The
unchanged ICD ReAct checker is deliberately excluded, otherwise historical
ICD failures could hide whether the new diagnosis graph itself completed.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from time import monotonic
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env")

from audit.diagnosis.validator import DiagnosisValidator
from RAG.retrieval.vector_store import close_pool

CASES_PATH = ROOT / "e2e" / "fixtures" / "eval_broken_cards" / "cases.json"
VALID_ASPECTS = {"anamnesis", "inspection", "treatment", "criteria"}
TIMEOUT_SECONDS = float(os.environ.get("E2E_DIAG_GRAPH_TIMEOUT_SECONDS", "900"))

_PASS = "  \033[32mok\033[0m"
_FAIL = "  \033[31mFAILED\033[0m"


def _case(case_id: str) -> dict[str, Any]:
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    try:
        return next(case for case in payload["cases"] if case["id"] == case_id)
    except StopIteration as exc:
        raise ValueError(f"Unknown eval_broken_cards case: {case_id}") from exc


async def _run_case(case: dict[str, Any]) -> list[str]:
    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        if condition:
            print(f"{_PASS}  {label}")
            return
        print(f"{_FAIL}  {label}{(' — ' + detail) if detail else ''}")
        failures.append(label)

    visit = case["visit"]
    diagnoses = visit.get("Диагнозы") or []
    eval_guid = (visit.get("Прием") or {}).get("GUID")
    check(
        "fixture keeps the selected eval GUID",
        eval_guid == case["eval_guid"],
        str(eval_guid),
    )
    check("fixture has diagnoses to pass through the graph", bool(diagnoses))
    if not diagnoses:
        return failures

    validator = DiagnosisValidator(visit)
    results = []
    started = monotonic()
    try:
        async with asyncio.timeout(TIMEOUT_SECONDS):
            for diagnosis in diagnoses:
                result, tokens = await validator.validate_diagnosis(diagnosis)
                results.append((result, tokens))
    except Exception as exc:  # noqa: BLE001 - e2e must report and exit non-zero
        check("diagnosis graph completes without an exception", False, repr(exc))
        traceback.print_exc()
        return failures

    elapsed = monotonic() - started
    check(
        "every diagnosis returned a result",
        len(results) == len(diagnoses),
        f"returned={len(results)} expected={len(diagnoses)}",
    )
    check(
        "card completed within the graph timeout",
        elapsed < TIMEOUT_SECONDS,
        f"elapsed={elapsed:.1f}s timeout={TIMEOUT_SECONDS:.1f}s",
    )

    graph_results = [
        (result, tokens) for result, tokens in results if result.guideline_file_id
    ]
    check(
        "at least one diagnosis reached the compiled graph",
        bool(graph_results),
        "no guideline_file_id was resolved on the configured stand",
    )
    check(
        "real graph calls produced token usage",
        any(tokens > 0 for _result, tokens in graph_results),
        str([tokens for _result, tokens in graph_results]),
    )
    check(
        "graph exposed the guideline chunks shown to judges",
        any(result.guideline_sources for result, _tokens in graph_results),
        "all guideline_sources collections are empty",
    )

    issues = [issue for result, _tokens in graph_results for issue in result.all_issues]
    invalid_aspects = sorted(
        {str(issue.aspect) for issue in issues if issue.aspect not in VALID_ASPECTS}
    )
    check(
        "every returned issue has a graph aspect",
        not invalid_aspects,
        str(invalid_aspects),
    )

    print(
        f"\n  elapsed={elapsed:.1f}s diagnoses={len(diagnoses)} graph_results={len(graph_results)}"
    )
    for result, tokens in results:
        print(
            f"  [{result.icd_code}] guideline={result.guideline_file_id or '—'} "
            f"issues={len(result.all_issues)} sources={len(result.guideline_sources)} "
            f"errors={len(result.errors)} tokens={tokens}"
        )
        for error in result.errors:
            print(f"    degradation: {error}")
    return failures


async def _main(case_id: str) -> int:
    case = _case(case_id)
    print(f"Diagnosis graph e2e: {case['id']}")
    print(f"  eval_guid={case['eval_guid']} original_guid={case['original_guid']}")
    print(f"  dataset_class={case['dataset_failure_class']}")
    print(f"  historical_reason={case['historical_failure_reason']}")
    print(f"  selected_because={case['selection']}")

    try:
        failures = await _run_case(case)
    finally:
        await close_pool()

    if failures:
        print(f"\n\033[31m{len(failures)} check(s) FAILED:\033[0m")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\n\033[32mAll checks passed.\033[0m")
    return 0


def run_case(case_id: str) -> None:
    """Run one standalone e2e case and terminate with its result code."""
    try:
        raise SystemExit(asyncio.run(_main(case_id)))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
