"""
audit/pipeline.py — top-level audit pipeline.

Accepts a raw JSON payload from 1C (a list of visits, or a wrapper dict
containing such a list), audits every visit through the formal-structure
and diagnosis validators, and writes one xlsx row per visit.

Usage::
    import asyncio, json
    from audit.pipeline import AuditPipeline

    pipeline = AuditPipeline(excel_path="audit_results.xlsx")
    results  = asyncio.run(pipeline.run(json.load(open("input.json"))))
"""

from __future__ import annotations

import asyncio
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


def _visit_guid(visit: dict[str, Any]) -> str | None:
    priem = visit.get("Прием") or {}
    guid = priem.get("GUID")
    return str(guid).lower() if guid else None



class AuditPipeline:
    """Run the full audit pipeline over a batch of visits.

    One xlsx row is written per visit, with all diagnosis checks for that
    appointment rendered together in the diagnosis column.

    Args:
        excel_path: Path to the output xlsx file (created if absent).
    """

    def __init__(self, excel_path: str | Path) -> None:
        self._excel = AuditExcelWriter(excel_path)
        self._excel_lock = asyncio.Lock()

    async def run(
        self,
        raw_input: dict | list | str,
        done_guids: set[str] | None = None,
    ) -> list[Result]:
        """Audit all appointments in *raw_input* and return their Results.

        Args:
            raw_input: JSON payload — a list of visit dicts, a wrapper dict,
                       or a raw JSON string of either shape.
            done_guids: Optional set of visit GUIDs already audited. Matching
                        visits are skipped before any LLM/checker work runs.

        Returns:
            One :class:`~storage.models.result.Result` per visit, containing
            a :class:`~storage.models.result.FormalStructureResult` and a list
            of :class:`~storage.models.result.DiagnosisResult` (one per ICD entry).
        """
        appointments = _split_appointments(raw_input)
        pending, skipped = self._filter_pending_appointments(appointments, done_guids)
        results: list[Result] = []

        for idx, visit in pending:
            priem: dict = visit.get("Прием") or {}
            visit_id = priem.get("GUID") or priem.get("DATE") or f"#{idx + 1}"
            logger.info("🩺 Auditing visit %s (%d/%d)", visit_id, idx + 1, len(appointments))
            visit_result = await self._audit_visit(visit)
            results.append(visit_result)

        self._log_queue_summary(appointments, done_guids, skipped, len(results))
        return results

    async def run_batched(
        self,
        raw_input: dict | list | str,
        num_batches: int,
        done_guids: set[str] | None = None,
    ) -> list[Result]:
        """Audit appointments like :meth:`run`, with at most *num_batches* running simultaneously.

        Args:
            raw_input: JSON payload — a list of visit dicts, a wrapper dict,
                       or a raw JSON string of either shape.
            num_batches: Maximum number of visits processed concurrently.
            done_guids: Optional set of visit GUIDs already audited. Matching
                        visits are filtered out before batch processing starts.
        """
        appointments = _split_appointments(raw_input)
        pending, skipped = self._filter_pending_appointments(appointments, done_guids)

        sem = asyncio.Semaphore(num_batches)

        async def _audit_with_sem(idx: int, visit: dict[str, Any]) -> Result:
            priem: dict = visit.get("Прием") or {}
            visit_id = priem.get("GUID") or priem.get("DATE") or f"#{idx + 1}"
            async with sem:
                logger.info("🩺 Auditing visit %s (%d/%d)", visit_id, idx + 1, len(appointments))
                return await self._audit_visit(visit)

        results: list[Result] = list(
            await asyncio.gather(*[_audit_with_sem(idx, visit) for idx, visit in pending])
        )

        self._log_queue_summary(appointments, done_guids, skipped, len(results))
        return results

    # ── Internal ──────────────────────────────────────────────────────────────

    def _filter_pending_appointments(
        self,
        appointments: list[dict[str, Any]],
        done_guids: set[str] | None,
    ) -> tuple[list[tuple[int, dict[str, Any]]], int]:
        normalized_done_guids = {str(guid).lower() for guid in (done_guids or set())}
        pending: list[tuple[int, dict[str, Any]]] = []
        skipped = 0

        for idx, visit in enumerate(appointments):
            priem: dict = visit.get("Прием") or {}
            visit_id = priem.get("GUID") or priem.get("DATE") or f"#{idx + 1}"
            visit_guid = _visit_guid(visit)

            if visit_guid and visit_guid in normalized_done_guids:
                skipped += 1
                logger.info(
                    "🩺 Skipping already audited visit %s (%d/%d)",
                    visit_guid, idx + 1, len(appointments),
                )
                continue

            if normalized_done_guids and not visit_guid:
                logger.warning(
                    "🩺 Visit %s has no Прием.GUID; auditing it because it cannot be matched to done_guids",
                    visit_id,
                )

            pending.append((idx, visit))

        return pending, skipped

    def _log_queue_summary(
        self,
        appointments: list[dict[str, Any]],
        done_guids: set[str] | None,
        skipped: int,
        audited: int,
    ) -> None:
        if done_guids:
            logger.info(
                "🩺 Pipeline audit queue complete: total=%d skipped=%d audited=%d",
                len(appointments),
                skipped,
                audited,
            )

    async def _append_excel(
        self,
        visit: dict[str, Any],
        formal: FormalStructureResult,
        diagnosis: DiagnosisAuditResult | list[DiagnosisAuditResult],
    ) -> None:
        async with self._excel_lock:
            self._excel.append(visit=visit, formal=formal, diagnosis=diagnosis)

    async def _audit_visit(self, visit: dict[str, Any]) -> Result:
        """Audit a single visit and return one Result object."""
        priem = visit.get("Прием") or {}
        visit_id = priem.get("GUID") or priem.get("DATE") or "unknown"
        logger.debug("[pipeline] _audit_visit START — visit_id=%s", visit_id)

        # ── Formal structure (once per visit) ─────────────────────────────────
        logger.info("📋 [pipeline] running FormalValidator for visit %s", visit_id)
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
            logger.info("🧬 [pipeline] visit %s has no diagnoses — skipping DiagnosisValidator", visit_id)
            empty_diag = DiagnosisAuditResult()
            await self._append_excel(visit=visit, formal=formal_result, diagnosis=empty_diag)
            return Result(input=visit, formal=formal_result, diagnosis=[])

        # ── Diagnosis check (once per diagnosis) ──────────────────────────────
        diag_validator = DiagnosisValidator(visit)
        diagnosis_results: list[DiagnosisResult] = []
        diagnosis_audit_results: list[DiagnosisAuditResult] = []

        for dx_idx, diagnosis in enumerate(diagnoses):
            dx_code = diagnosis.get("КодМКБ", f"#{dx_idx + 1}")
            logger.info(
                "🧬 [pipeline] DiagnosisValidator — visit %s, diagnosis %d/%d (%s)",
                visit_id, dx_idx + 1, len(diagnoses), dx_code,
            )
            logger.debug(
                "[pipeline] diagnosis input code=%s name=%s",
                dx_code,
                diagnosis.get("НаименованиеМКБ"),
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
            diagnosis_audit_results.append(diag_result)

            diagnosis_results.append(
                DiagnosisResult(
                    icd_code=dx_code,
                    issues=diag_result.all_issues,
                )
            )

        await self._append_excel(
            visit=visit,
            formal=formal_result,
            diagnosis=diagnosis_audit_results,
        )

        result = Result(
            input=visit,
            formal=formal_result,
            diagnosis=diagnosis_results,
        )
        logger.debug("[pipeline] _audit_visit END — visit_id=%s", visit_id)
        return result
