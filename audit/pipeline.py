"""
audit/pipeline.py — top-level audit pipeline.

Accepts a raw JSON payload from 1C (a list of visits, or a wrapper dict
containing such a list), audits every visit through the formal-structure
and diagnosis validators, and writes one xlsx row per (visit, diagnosis).

Usage::
    import asyncio, json
    from audit.pipeline import AuditPipeline

    pipeline = AuditPipeline(excel_path="audit_results.xlsx")
    results  = asyncio.run(pipeline.run(json.load(open("input.json"))))
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from audit.diagnosis.validator import DiagnosisValidator
from parsers.excel import AuditExcelWriter
from audit.formal_structure.validator import FormalValidator
from audit.models import DiagnosisAuditResult, FormalFinding, FormalStructureResult
from storage.models.result import Result

logger = logging.getLogger(__name__)

_APPOINTMENTS_KEY = "appointments"


def _split_appointments(raw: Any) -> list[dict[str, Any]]:
    """Extract the appointments list from *raw*.

    Accepts a wrapper dict with the ``"appointments"`` key, a bare list,
    or a raw JSON string of either shape.
    """
    if isinstance(raw, str):
        raw = json.loads(raw)

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        return raw[_APPOINTMENTS_KEY]

    raise ValueError(f"Cannot extract appointments from input of type {type(raw).__name__!r}")


class AuditPipeline:
    """Run the full audit pipeline over a batch of visits.

    One xlsx row is written per **(visit, diagnosis)** pair so that each
    diagnosis can be inspected independently alongside its formal findings.

    Args:
        excel_path: Path to the output xlsx file (created if absent).
    """

    def __init__(self, excel_path: str | Path) -> None:
        self._excel = AuditExcelWriter(excel_path)

    async def run(
        self,
        raw_input: dict | list | str,
    ) -> list[Result]:
        """Audit all appointments in *raw_input* and return their Results.

        Args:
            raw_input: JSON payload — a list of visit dicts, a wrapper dict,
                       or a raw JSON string of either shape.

        Returns:
            One :class:`~storage.models.result.Result` per (visit, diagnosis).
            If a visit has no diagnoses, a single result with an empty issues
            list is still returned (formal findings only).
        """
        appointments = _split_appointments(raw_input)
        results: list[Result] = []

        for idx, visit in enumerate(appointments):
            priem: dict = visit.get("Прием") or {}
            visit_id = priem.get("GUID") or priem.get("DATE") or f"#{idx + 1}"
            logger.info("Auditing visit %s (%d/%d)", visit_id, idx + 1, len(appointments))

            visit_results = await self._audit_visit(visit)
            results.extend(visit_results)

        return results

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _audit_visit(self, visit: dict[str, Any]) -> list[Result]:
        """Audit a single visit; returns one Result per diagnosis."""

        # ── Formal structure (once per visit) ─────────────────────────────────
        formal_raw = FormalValidator().validate(visit)
        formal_result = FormalStructureResult(
            findings=[
                FormalFinding(flag=f["flag"], issue=f["issue"]) for f in formal_raw
            ]
        )

        diagnoses: list[dict] = visit.get("Диагнозы", [])

        if not diagnoses:
            # No diagnoses — log formal findings only.
            empty_diag = DiagnosisAuditResult()
            self._excel.append(visit=visit, formal=formal_result, diagnosis=empty_diag)
            return [Result(input=visit, flags=formal_result.flags, issues=[])]

        # ── Diagnosis check (once per diagnosis) ──────────────────────────────
        diag_validator = DiagnosisValidator(visit)
        results: list[Result] = []

        for diagnosis in diagnoses:
            diag_result = await diag_validator.validate_diagnosis(diagnosis)

            self._excel.append(
                visit=visit,
                formal=formal_result,
                diagnosis=diag_result,
            )

            results.append(
                Result(
                    input=visit,
                    flags=formal_result.flags,
                    issues=diag_result.all_issues,
                )
            )

        return results
