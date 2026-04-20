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
from storage.models.result import DiagnosisResult, Result

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
            One :class:`~storage.models.result.Result` per visit, containing
            a :class:`~storage.models.result.FormalStructureResult` and a list
            of :class:`~storage.models.result.DiagnosisResult` (one per ICD entry).
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
        priem = visit.get("Прием") or {}
        visit_id = priem.get("GUID") or priem.get("DATE") or "unknown"
        logger.debug("[pipeline] _audit_visit START — visit_id=%s", visit_id)
        logger.debug("[pipeline] visit input:\n%s", json.dumps(visit, ensure_ascii=False, indent=2))

        # ── Formal structure (once per visit) ─────────────────────────────────
        logger.info("[pipeline] running FormalValidator for visit %s", visit_id)
        formal_raw = await FormalValidator().validate(visit)
        formal_result = FormalStructureResult(
            findings=[
                FormalFinding(flag=f["flag"], issue=f["issue"]) for f in formal_raw
            ]
        )
        logger.info(
            "[pipeline] FormalValidator done:\n%s",
            formal_result.pretty_format(),
        )

        diagnoses: list[dict] = visit.get("Диагнозы", [])
        logger.debug("[pipeline] diagnoses found: %d", len(diagnoses))

        if not diagnoses:
            logger.info("[pipeline] visit %s has no diagnoses — skipping DiagnosisValidator", visit_id)
            empty_diag = DiagnosisAuditResult()
            self._excel.append(visit=visit, formal=formal_result, diagnosis=empty_diag)
            return [Result(input=visit, formal=formal_result, diagnosis=[])]

        # ── Diagnosis check (once per diagnosis) ──────────────────────────────
        diag_validator = DiagnosisValidator(visit)
        diagnosis_results: list[DiagnosisResult] = []

        for dx_idx, diagnosis in enumerate(diagnoses):
            dx_code = diagnosis.get("КодМКБ", f"#{dx_idx + 1}")
            logger.info(
                "[pipeline] DiagnosisValidator — visit %s, diagnosis %d/%d (%s)",
                visit_id, dx_idx + 1, len(diagnoses), dx_code,
            )
            logger.debug(
                "[pipeline] diagnosis input:\n%s",
                json.dumps(diagnosis, ensure_ascii=False, indent=2),
            )
            diag_result = await diag_validator.validate_diagnosis(diagnosis)
            logger.info(
                "[pipeline] DiagnosisValidator done — visit %s dx %s: "
                "anamnesis=%d inspection=%d treatment=%d",
                visit_id, dx_code,
                len(diag_result.anamnesis_issues),
                len(diag_result.inspection_issues),
                len(diag_result.treatment_issues),
            )
            logger.debug(
                "[pipeline] DiagnosisAuditResult:\n%s",
                diag_result.pretty_format(),
            )

            self._excel.append(
                visit=visit,
                formal=formal_result,
                diagnosis=diag_result,
            )

            diagnosis_results.append(
                DiagnosisResult(
                    icd_code=dx_code,
                    issues=diag_result.all_issues,
                )
            )

        result = Result(
            input=visit,
            formal=formal_result,
            diagnosis=diagnosis_results,
        )
        logger.debug("[pipeline] _audit_visit END — visit_id=%s\n%s", visit_id, result.pretty_format())
        return [result]
