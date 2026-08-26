"""
audit/pipeline.py — top-level audit pipeline.

Accepts a raw JSON payload from 1C (a list of visits, or a wrapper dict
containing such a list), audits every visit through the formal-structure
and diagnosis validators, and persists one done_cards row per visit.

Usage::
    import asyncio, json
    from audit.pipeline import AuditPipeline

    async with AuditPipeline() as pipeline:
        results = await pipeline.run_batched(json.load(open("input.json")))
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from datetime import datetime, timezone
from functools import wraps
from typing import Any

from audit.diagnosis.clinic_recs import _is_age_eligible, _patient_age
from audit.diagnosis.validator import DiagnosisValidator
from audit.filters import CardFilter
from audit.formal_structure.validator import FormalValidator
from audit.graph_trace import (
    current_correlation_id,
    new_correlation_id,
    trace_context,
)
from audit.graph_trace import (
    emit as trace_emit,
)
from audit.icd_check.validator import check_icd_codes
from audit.models import FormalFinding, FormalStructureResult
from parsers.inspection_labels import normalize_visit_labels
from parsers.json_parser import AppointmentParser
from storage.done_cards_storage import DoneCardsStorage
from storage.guidelines_storage import GuidelinesStorage
from storage.models.result import (
    DiagnosisIssue,
    DiagnosisResult,
    IcdCodingIssue,
    Result,
)

logger = logging.getLogger(__name__)


def _visit_guid(visit: dict[str, Any]) -> str | None:
    priem = visit.get("Прием") or {}
    guid = priem.get("GUID")
    return str(guid).lower() if guid else None


def _traced_card_audit(method):
    """Give every direct or batched card audit one end-to-end trace identity."""

    @wraps(method)
    async def wrapper(self, visit: dict[str, Any]):
        priem = visit.get("Прием") or {}
        card_guid = priem.get("GUID") or None
        correlation_id = current_correlation_id() or new_correlation_id()
        started = time.monotonic()
        with trace_context(correlation_id, card_guid):
            trace_emit(
                "audit.started",
                organization_id=self._org_id,
                card_data_priem=priem,
            )
            try:
                result = await method(self, visit)
            except BaseException as exc:
                trace_emit(
                    "audit.failed",
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    exception=exc,
                    traceback=traceback.format_exc(),
                )
                raise
            trace_emit(
                "audit.completed",
                elapsed_ms=int((time.monotonic() - started) * 1000),
                token_count=result.token_count,
                output=result,
            )
            return result

    return wrapper


class AuditPipeline:
    """Run the full audit pipeline over a batch of visits.

    Persists one done_cards DB row per visit. Excel export is handled
    separately by audit.excel_formatter.
    """

    def __init__(
        self, org_id: str | None = None, card_filter: CardFilter | None = None
    ) -> None:
        self._org_id = org_id
        self._card_filter = card_filter or CardFilter([])
        self._done_cards: DoneCardsStorage | None = None

    async def __aenter__(self) -> AuditPipeline:  # noqa: PYI034
        self._done_cards = DoneCardsStorage()
        await self._done_cards.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._done_cards is not None:
            await self._done_cards.__aexit__(*args)
            self._done_cards = None

    async def run(
        self,
        raw_input: dict | list | str,
        done_guids: set[str] | None = None,
    ) -> list[tuple[Result, int]]:
        """Deprecated sequential wrapper — delegates to :meth:`run_batched` with num_batches=1."""
        return await self.run_batched(raw_input, num_batches=1, done_guids=done_guids)

    async def run_batched(
        self,
        raw_input: dict | list | str,
        num_batches: int = 5,
        done_guids: set[str] | None = None,
    ) -> list[tuple[Result, int]]:
        """Audit all appointments concurrently, up to *num_batches* at a time.

        Args:
            raw_input:   JSON payload — a list of visit dicts, a wrapper dict,
                         or a raw JSON string of either shape.
            num_batches: Maximum number of visits processed concurrently (default 5).
            done_guids:  Set of visit GUIDs already audited. If omitted, loaded
                         from the DB automatically.

        Returns:
            List of ``(Result, elapsed_ms)`` pairs — one per audited visit,
            where ``elapsed_ms`` is the wall-clock time for that card in milliseconds.
        """
        if done_guids is None:
            done_guids = await self._done_cards.get_done_guids(
                organization_id=self._org_id
            )

        appointments = AppointmentParser.split(raw_input)
        pending, ignored_visits, skipped_done, skipped_strategy = (
            self._card_filter.filter(appointments, done_guids)
        )

        if ignored_visits and self._done_cards is not None:
            await asyncio.gather(
                *[
                    self._done_cards.upsert_ignored(
                        card_guid=_visit_guid(v),
                        card_data=json.dumps(v, ensure_ascii=False),
                        organization_id=self._org_id,
                    )
                    for v in ignored_visits
                    if _visit_guid(v)
                ]
            )

        sem = asyncio.Semaphore(num_batches)

        async def _audit_with_sem(
            idx: int, visit: dict[str, Any]
        ) -> tuple[Result, int] | None:
            priem: dict = visit.get("Прием") or {}
            visit_id = priem.get("GUID") or priem.get("DATE") or f"#{idx + 1}"
            card_guid = priem.get("GUID") or None
            correlation_id = new_correlation_id()
            async with sem:
                with trace_context(correlation_id, card_guid):
                    logger.info(
                        "🩺 Auditing visit %s (%d/%d)",
                        visit_id,
                        idx + 1,
                        len(appointments),
                    )
                    t_start = time.monotonic()
                    dt_start = datetime.now(timezone.utc)
                    try:
                        result = await self._audit_visit(visit)
                        elapsed_ms = int((time.monotonic() - t_start) * 1000)
                        logger.info("🩺 Visit %s done in %d ms", visit_id, elapsed_ms)
                        return result, elapsed_ms
                    except Exception:
                        elapsed_ms = int((time.monotonic() - t_start) * 1000)
                        tb = traceback.format_exc()
                        logger.exception(
                            "🩺 Visit %s FAILED after %d ms", visit_id, elapsed_ms
                        )
                        if self._done_cards is not None:
                            trace_emit("audit.broken_persistence.started")
                            try:
                                await self._done_cards.upsert_broken(
                                    card_guid=card_guid,
                                    card_data=json.dumps(visit, ensure_ascii=False),
                                    stacktrace=tb,
                                    started_at=dt_start,
                                    organization_id=self._org_id,
                                )
                                trace_emit("audit.broken_persistence.completed")
                            except Exception as persistence_exc:
                                trace_emit(
                                    "audit.broken_persistence.failed",
                                    exception=persistence_exc,
                                    traceback=traceback.format_exc(),
                                )
                                logger.exception(
                                    "💾 Failed to persist broken card guid=%s",
                                    card_guid,
                                )
                        return None

        raw_pairs = await asyncio.gather(
            *[_audit_with_sem(idx, visit) for idx, visit in pending]
        )
        pairs: list[tuple[Result, int]] = [p for p in raw_pairs if p is not None]

        self._log_queue_summary(
            appointments, done_guids, skipped_done, skipped_strategy, len(pairs)
        )
        return pairs

    # ── Internal ──────────────────────────────────────────────────────────────

    def _log_queue_summary(
        self,
        appointments: list[dict[str, Any]],
        done_guids: set[str] | None,
        skipped_done: int,
        skipped_strategy: int,
        audited: int,
    ) -> None:
        if done_guids or skipped_strategy:
            logger.info(
                "🩺 Pipeline audit queue complete: total=%d skipped_done=%d skipped_strategy=%d audited=%d",
                len(appointments),
                skipped_done,
                skipped_strategy,
                audited,
            )

    @_traced_card_audit
    async def _audit_visit(self, visit: dict[str, Any]) -> Result:
        """Audit a single visit and return one Result object."""
        # единственная точка входа визита в аудит: дальше этот же dict уходит и
        # в проверяющие модели, и в карточку отчёта, поэтому имена полей осмотра
        # приводятся к врачебным здесь (см. parsers.inspection_labels)
        visit = normalize_visit_labels(visit)
        priem = visit.get("Прием") or {}
        visit_id = priem.get("GUID") or priem.get("DATE") or "unknown"
        card_guid = priem.get("GUID") or None
        logger.debug("[pipeline] _audit_visit START — visit_id=%s", visit_id)

        t_start = time.monotonic()
        dt_start = datetime.now(timezone.utc)

        # ── Formal structure (once per visit) ─────────────────────────────────
        logger.info("📋 [pipeline] running FormalValidator for visit %s", visit_id)
        trace_emit("checker.started", checker="formal")
        try:
            formal_raw, formal_tokens = await FormalValidator().validate(visit)
        except Exception as exc:
            trace_emit(
                "checker.failed",
                checker="formal",
                exception=exc,
                traceback=traceback.format_exc(),
            )
            raise
        formal_result = FormalStructureResult(
            findings=[
                FormalFinding(
                    flag=f["flag"],
                    issue=f["issue"],
                    source=f.get("source", ""),
                    comment=f.get("comment", ""),
                )
                for f in formal_raw
            ]
        )
        logger.info(
            "[pipeline] FormalValidator done (tokens=%d):\n%s",
            formal_tokens,
            formal_result.pretty_format(),
        )
        trace_emit(
            "checker.completed",
            checker="formal",
            output=formal_raw,
            tokens=formal_tokens,
        )

        patient: dict = visit.get("Пациент") or {}
        diagnoses: list[dict] = visit.get("Диагнозы") or []
        inspection_data: list[dict] = visit.get("ДанныеОсмотра") or []
        logger.debug("[pipeline] diagnoses found: %d", len(diagnoses))

        if not diagnoses:
            logger.info(
                "🧬 [pipeline] visit %s has no diagnoses — skipping ICD check and DiagnosisValidator",
                visit_id,
            )
            await self._upsert_done_card(
                visit=visit,
                card_guid=card_guid,
                formal=formal_result,
                diagnosis=[],
                icd_check=[],
                time_ms=int((time.monotonic() - t_start) * 1000),
                token_count=formal_tokens,
                started_at=dt_start,
                finished_at=datetime.now(timezone.utc),
            )
            trace_emit("checker.skipped", checker="icd", reason="no diagnoses")
            trace_emit("checker.skipped", checker="diagnosis", reason="no diagnoses")
            return Result(
                input=visit,
                formal=formal_result,
                diagnosis=[],
                token_count=formal_tokens,
            )

        # ── ICD coding check (once per visit, all diagnoses together) ─────────
        age = _patient_age(patient)
        # TODO(guidelines-sql): фильтрацию по возрасту вынести в SQL —
        # GuidelinesStorage.all_age_eligible(age) вместо загрузки всего
        # справочника и фильтра в Python. См. spec §5.
        async with GuidelinesStorage() as _store:
            manifest_rows = [g for g in await _store.all() if _is_age_eligible(g, age)]

        logger.info(
            "🔎 [pipeline] running ICD checker for visit %s (%d diagnoses)",
            visit_id,
            len(diagnoses),
        )
        trace_emit(
            "checker.started",
            checker="icd",
            diagnoses=diagnoses,
            manifest_rows=manifest_rows,
            inspection_data=inspection_data,
        )
        try:
            icd_issues, icd_tokens = await check_icd_codes(
                patient=patient,
                diagnoses=diagnoses,
                manifest_rows=manifest_rows,
                inspection_data=inspection_data,
                card_guid=card_guid,
            )
        except Exception as exc:
            trace_emit(
                "checker.failed",
                checker="icd",
                exception=exc,
                traceback=traceback.format_exc(),
            )
            raise
        if icd_issues is None:
            # Отказ ICD-чекера не отменяет уже сделанную формальную проверку и
            # не отменяет проверку по клин.рекомендациям: карта доезжает, а
            # icd_check_result остаётся NULL — «мнения нет» отличимо от
            # «замечаний нет» одним SQL-запросом.
            trace_emit("checker.failed", checker="icd", reason="checker did not complete")
            logger.error(
                "[pipeline] ICD checker did not complete for visit %s — card continues without it",
                visit_id,
            )
        else:
            logger.info(
                "[pipeline] ICD checker done — %d recommendation(s), tokens=%d",
                len(icd_issues),
                icd_tokens,
            )
            trace_emit(
                "checker.completed",
                checker="icd",
                output=icd_issues,
                tokens=icd_tokens,
            )

        # ── Diagnosis check (once per code врача) ─────────────────────────────
        # Рекомендованный чекером код здесь не аудируется: он гипотеза, а не
        # диагноз визита. Прогон по нему давал вторую порцию замечаний к тому
        # же приёму (24 частично совпадающих на I67.8/I69.3, до 200 тысяч
        # токенов на карту) и уходил в граф с наименованием, равным самому
        # коду. Решение от 2026-08-22.
        diag_validator = DiagnosisValidator(visit)
        diagnosis_results: list[DiagnosisResult] = []
        total_tokens = formal_tokens + icd_tokens

        for dx_idx, diagnosis in enumerate(diagnoses):
            dx_code = diagnosis.get("КодМКБ", f"#{dx_idx + 1}")
            logger.info(
                "🧬 [pipeline] DiagnosisValidator — visit %s, diagnosis %d/%d (%s)",
                visit_id,
                dx_idx + 1,
                len(diagnoses),
                dx_code,
            )
            trace_emit(
                "checker.started",
                checker="diagnosis",
                diagnosis_index=dx_idx,
                diagnosis=diagnosis,
                suggested=False,
            )

            if self._card_filter.should_skip_diagnosis(diagnosis):
                logger.info(
                    "🧬 [pipeline] Skipping diagnosis %s for visit %s — filtered by ICD filter",
                    dx_code,
                    visit_id,
                )
                diagnosis_results.append(
                    DiagnosisResult(
                        icd_code=dx_code,
                        issues=[
                            DiagnosisIssue(
                                issue="Диагноз пропущен фильтром МКБ", sources=[]
                            )
                        ],
                    )
                )
                trace_emit(
                    "checker.skipped",
                    checker="diagnosis",
                    diagnosis_index=dx_idx,
                    diagnosis=diagnosis,
                    reason="ICD filter",
                )
                continue

            # Audit original code
            try:
                diag_result, diag_tokens = await diag_validator.validate_diagnosis(
                    diagnosis
                )
            except Exception as exc:
                trace_emit(
                    "checker.failed",
                    checker="diagnosis",
                    diagnosis_index=dx_idx,
                    diagnosis=diagnosis,
                    suggested=False,
                    exception=exc,
                    traceback=traceback.format_exc(),
                )
                raise
            total_tokens += diag_tokens
            logger.info(
                "[pipeline] DiagnosisValidator done — visit %s dx %s: "
                "anamnesis=%d inspection=%d treatment=%d criteria=%d tokens=%d",
                visit_id,
                dx_code,
                len(diag_result.anamnesis_issues),
                len(diag_result.inspection_issues),
                len(diag_result.treatment_issues),
                len(diag_result.criteria_issues),
                diag_tokens,
            )
            diagnosis_results.append(
                DiagnosisResult(
                    icd_code=dx_code,
                    issues=diag_result.all_issues,
                    guideline_file_id=diag_result.guideline_file_id,
                    guideline_meta=diag_result.guideline_meta,
                    guideline_sources=diag_result.guideline_sources,
                    errors=diag_result.errors,
                )
            )
            trace_emit(
                "checker.completed",
                checker="diagnosis",
                diagnosis_index=dx_idx,
                diagnosis=diagnosis,
                suggested=False,
                output=diag_result.to_dict(),
                tokens=diag_tokens,
            )

        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        dt_finish = datetime.now(timezone.utc)

        await self._upsert_done_card(
            visit=visit,
            card_guid=card_guid,
            formal=formal_result,
            diagnosis=diagnosis_results,
            icd_check=icd_issues,
            time_ms=elapsed_ms,
            token_count=total_tokens,
            started_at=dt_start,
            finished_at=dt_finish,
        )

        result = Result(
            input=visit,
            formal=formal_result,
            diagnosis=diagnosis_results,
            token_count=total_tokens,
        )
        logger.info(
            "[pipeline] _audit_visit END — visit_id=%s total_tokens=%d",
            visit_id,
            total_tokens,
        )
        return result

    async def _upsert_done_card(
        self,
        *,
        visit: dict[str, Any],
        card_guid: str | None,
        formal: FormalStructureResult,
        diagnosis: list[DiagnosisResult],
        icd_check: list[IcdCodingIssue] | None,
        time_ms: int,
        started_at: datetime,
        finished_at: datetime,
        token_count: int = 0,
    ) -> None:
        if self._done_cards is None:
            trace_emit("audit.persistence.skipped", reason="storage is not configured")
            return
        card_data = json.dumps(visit, ensure_ascii=False)
        trace_emit(
            "audit.persistence.started",
            output={
                "formal": formal,
                "diagnosis": diagnosis,
                "icd_check": icd_check,
                "token_count": token_count,
                "time_ms": time_ms,
            },
        )
        try:
            await self._done_cards.upsert(
                card_guid=card_guid,
                card_data=card_data,
                formal=formal,
                diagnosis=diagnosis,
                icd_check=icd_check,
                token_count=token_count,
                time_ms=time_ms,
                started_at=started_at,
                finished_at=finished_at,
                organization_id=self._org_id,
            )
        except Exception as exc:
            trace_emit(
                "audit.persistence.failed",
                exception=exc,
                traceback=traceback.format_exc(),
            )
            raise
        trace_emit("audit.persistence.completed")
