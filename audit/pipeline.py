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
from typing import Any

from audit.diagnosis.validator import DiagnosisValidator
from audit.formal_structure.validator import FormalValidator
from audit.models import FormalFinding, FormalStructureResult
from storage.done_cards_storage import DoneCardsStorage
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

    Persists one done_cards DB row per visit. Excel export is handled
    separately by audit.excel_formatter.
    """

    def __init__(self) -> None:
        self._done_cards: DoneCardsStorage | None = None

    async def __aenter__(self) -> "AuditPipeline":
        self._done_cards = DoneCardsStorage()
        await self._done_cards.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._done_cards is not None:
            await self._done_cards.__aexit__(*args)
            self._done_cards = None

    async def run(
        self,
        raw_input: dict | list | str,
        done_guids: set[str] | None = None,
        ignore_icd: list[str] | None = None,
    ) -> list[tuple[Result, int]]:
        """Deprecated sequential wrapper — delegates to :meth:`run_batched` with num_batches=1."""
        return await self.run_batched(raw_input, num_batches=1, done_guids=done_guids, ignore_icd=ignore_icd)

    async def run_batched(
        self,
        raw_input: dict | list | str,
        num_batches: int = 5,
        done_guids: set[str] | None = None,
        ignore_icd: list[str] | None = None,
    ) -> list[tuple[Result, int]]:
        """Audit all appointments concurrently, up to *num_batches* at a time.

        Args:
            raw_input:   JSON payload — a list of visit dicts, a wrapper dict,
                         or a raw JSON string of either shape.
            num_batches: Maximum number of visits processed concurrently (default 5).
            done_guids:  Optional set of visit GUIDs already audited. Matching
                         visits are filtered out before processing starts.
            ignore_icd:  Optional list of ICD codes. A visit is skipped only if
                         every one of its diagnoses is in this list.

        Returns:
            List of ``(Result, elapsed_ms)`` pairs — one per audited visit,
            where ``elapsed_ms`` is the wall-clock time for that card in milliseconds.
        """
        appointments = _split_appointments(raw_input)
        pending, skipped_done, skipped_icd = self._filter_pending_appointments(appointments, done_guids, ignore_icd)

        sem = asyncio.Semaphore(num_batches)

        async def _audit_with_sem(idx: int, visit: dict[str, Any]) -> tuple[Result, int]:
            priem: dict = visit.get("Прием") or {}
            visit_id = priem.get("GUID") or priem.get("DATE") or f"#{idx + 1}"
            async with sem:
                logger.info("🩺 Auditing visit %s (%d/%d)", visit_id, idx + 1, len(appointments))
                t_start = time.monotonic()
                result = await self._audit_visit(visit)
                elapsed_ms = int((time.monotonic() - t_start) * 1000)
                logger.info("🩺 Visit %s done in %d ms", visit_id, elapsed_ms)
                return result, elapsed_ms

        pairs: list[tuple[Result, int]] = list(
            await asyncio.gather(*[_audit_with_sem(idx, visit) for idx, visit in pending])
        )

        self._log_queue_summary(appointments, done_guids, skipped_done, skipped_icd, len(pairs))
        return pairs

    # ── Internal ──────────────────────────────────────────────────────────────

    def _filter_pending_appointments(
        self,
        appointments: list[dict[str, Any]],
        done_guids: set[str] | None,
        ignore_icd: list[str] | None = None,
    ) -> tuple[list[tuple[int, dict[str, Any]]], int, int]:
        """Return (pending, skipped_done, skipped_icd)."""
        normalized_done_guids = {str(guid).lower() for guid in (done_guids or set())}
        normalized_ignore_icd = {code.upper() for code in (ignore_icd or [])}
        pending: list[tuple[int, dict[str, Any]]] = []
        skipped_done = 0
        skipped_icd = 0

        for idx, visit in enumerate(appointments):
            priem: dict = visit.get("Прием") or {}
            visit_id = priem.get("GUID") or priem.get("DATE") or f"#{idx + 1}"
            visit_guid = _visit_guid(visit)

            if visit_guid and visit_guid in normalized_done_guids:
                skipped_done += 1
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

            if normalized_ignore_icd:
                diagnoses: list[dict] = visit.get("Диагнозы") or []
                dx_codes = {(d.get("КодМКБ") or "").upper() for d in diagnoses}
                if dx_codes and dx_codes.issubset(normalized_ignore_icd):
                    skipped_icd += 1
                    logger.info(
                        "🩺 Skipping visit %s — all diagnoses %s are in ignore_icd list",
                        visit_id, dx_codes,
                    )
                    continue

            pending.append((idx, visit))

        return pending, skipped_done, skipped_icd

    def _log_queue_summary(
        self,
        appointments: list[dict[str, Any]],
        done_guids: set[str] | None,
        skipped_done: int,
        skipped_icd: int,
        audited: int,
    ) -> None:
        if done_guids or skipped_icd:
            logger.info(
                "🩺 Pipeline audit queue complete: total=%d skipped_done=%d skipped_icd=%d audited=%d",
                len(appointments),
                skipped_done,
                skipped_icd,
                audited,
            )

    async def _audit_visit(self, visit: dict[str, Any]) -> Result:
        """Audit a single visit and return one Result object."""
        priem = visit.get("Прием") or {}
        visit_id = priem.get("GUID") or priem.get("DATE") or "unknown"
        card_guid = priem.get("GUID") or None
        logger.debug("[pipeline] _audit_visit START — visit_id=%s", visit_id)

        t_start = time.monotonic()

        # ── Formal structure (once per visit) ─────────────────────────────────
        logger.info("📋 [pipeline] running FormalValidator for visit %s", visit_id)
        formal_raw, formal_tokens = await FormalValidator().validate(visit)
        formal_result = FormalStructureResult(
            findings=[
                FormalFinding(flag=f["flag"], issue=f["issue"]) for f in formal_raw
            ]
        )
        logger.info(
            "[pipeline] FormalValidator done (tokens=%d):\n%s",
            formal_tokens,
            formal_result.pretty_format(),
        )

        diagnoses: list[dict] = visit.get("Диагнозы", [])
        logger.debug("[pipeline] diagnoses found: %d", len(diagnoses))

        if not diagnoses:
            logger.info("🧬 [pipeline] visit %s has no diagnoses — skipping DiagnosisValidator", visit_id)
            await self._upsert_done_card(
                visit=visit, card_guid=card_guid,
                formal=formal_result, diagnosis=[],
                time_ms=int((time.monotonic() - t_start) * 1000),
                token_count=formal_tokens,
            )
            return Result(input=visit, formal=formal_result, diagnosis=[], token_count=formal_tokens)

        # ── Diagnosis check (once per diagnosis) ──────────────────────────────
        diag_validator = DiagnosisValidator(visit)
        diagnosis_results: list[DiagnosisResult] = []
        total_tokens = formal_tokens

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
            diag_result, diag_tokens = await diag_validator.validate_diagnosis(diagnosis)
            total_tokens += diag_tokens
            logger.info(
                "[pipeline] DiagnosisValidator done — visit %s dx %s: "
                "anamnesis=%d inspection=%d treatment=%d tokens=%d",
                visit_id, dx_code,
                len(diag_result.anamnesis_issues),
                len(diag_result.inspection_issues),
                len(diag_result.treatment_issues),
                diag_tokens,
            )
            diagnosis_results.append(
                DiagnosisResult(
                    icd_code=dx_code,
                    issues=diag_result.all_issues,
                )
            )

        elapsed_ms = int((time.monotonic() - t_start) * 1000)

        await self._upsert_done_card(
            visit=visit, card_guid=card_guid,
            formal=formal_result, diagnosis=diagnosis_results,
            time_ms=elapsed_ms,
            token_count=total_tokens,
        )

        result = Result(
            input=visit,
            formal=formal_result,
            diagnosis=diagnosis_results,
            token_count=total_tokens,
        )
        logger.info("[pipeline] _audit_visit END — visit_id=%s total_tokens=%d", visit_id, total_tokens)
        return result

    async def _upsert_done_card(
        self,
        *,
        visit: dict[str, Any],
        card_guid: str | None,
        formal: FormalStructureResult,
        diagnosis: list[DiagnosisResult],
        time_ms: int,
        token_count: int = 0,
    ) -> None:
        if self._done_cards is None:
            return
        card_data = json.dumps(visit, ensure_ascii=False)
        await self._done_cards.upsert(
            card_guid=card_guid,
            card_data=card_data,
            formal=formal,
            diagnosis=diagnosis,
            token_count=token_count,
            time_ms=time_ms,
        )
