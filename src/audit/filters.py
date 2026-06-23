"""
audit/filters.py — card-level and diagnosis-level filter strategies.

Each strategy implements two methods:
  - should_skip(visit)     → bool: True means skip the entire card from auditing
  - skip_diagnosis(diag)   → bool: True means skip clinical-rec check for this diagnosis

CardFilter holds a list of strategies and applies them during pipeline.run_batched().
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_УСЛУГИ = "Услуги"
_ДИАГНОЗЫ = "Диагнозы"
_КОД_МКБ = "КодМКБ"
_НАИМЕНОВАНИЕ = "Наименование"
_АРТИКУЛ = "Артикул"
_КОД_ЕГИСЗ = "КодЕГИСЗ"
_УИД_ЕГИСЗ = "УИДЕГИСЗ"

_SERVICE_CODE_FIELDS = (_АРТИКУЛ, _КОД_ЕГИСЗ, _УИД_ЕГИСЗ)


class _BaseFilter:
    def skip_diagnosis(self, diagnosis: dict[str, Any]) -> bool:
        return False


class IcdFilter(_BaseFilter):
    """Skip card if ALL diagnoses are in the codes list.
    Skip a single diagnosis in the clinical-rec stage if its code is in the list.
    """

    def __init__(self, codes: list[str]) -> None:
        self._codes = {c.upper() for c in codes}

    def __str__(self) -> str:
        codes = ", ".join(sorted(self._codes)) if self._codes else "(none)"
        return f"IcdFilter: skip visits/diagnoses with ICD codes [{codes}]"

    def should_skip(self, visit: dict[str, Any]) -> bool:
        if not self._codes:
            return False
        diagnoses: list[dict] = visit.get(_ДИАГНОЗЫ) or []
        if not diagnoses:
            return False
        dx_codes = {(d.get(_КОД_МКБ) or "").upper() for d in diagnoses}
        return bool(dx_codes) and dx_codes.issubset(self._codes)

    def skip_diagnosis(self, diagnosis: dict[str, Any]) -> bool:
        if not self._codes:
            return False
        return (diagnosis.get(_КОД_МКБ) or "").upper() in self._codes


class KDLFilter(_BaseFilter):
    """Skip card if ALL services each satisfy:
      Наименование ends with "(КДЛ)"  OR  any code field matches the 6.2.A5.101-style pattern.
    """

    _KDL_SUFFIX = "(КДЛ)"
    _CODE_RE = re.compile(r"\d+\.\d+\.[A-Za-zА-Яа-яЁё]\d+\.\d+")

    def __str__(self) -> str:
        return "KDLFilter: skip KDL-only visits"

    def _service_matches(self, service: dict[str, Any]) -> bool:
        name: str = service.get(_НАИМЕНОВАНИЕ) or ""
        if name.endswith(self._KDL_SUFFIX):
            return True
        for field in _SERVICE_CODE_FIELDS:
            value: str = service.get(field) or ""
            if self._CODE_RE.search(value):
                return True
        return False

    def should_skip(self, visit: dict[str, Any]) -> bool:
        services: list[dict] = visit.get(_УСЛУГИ) or []
        if not services:
            return False
        return all(self._service_matches(s) for s in services)


class AnalysisFilter(_BaseFilter):
    """Skip card if ALL services each have any code field matching the A04.16.001-style pattern
    (starts with Latin or Cyrillic A).
    """

    _CODE_RE = re.compile(r"[AА]\d{2,}\.\d{2,}\.\d{3,}")

    def __str__(self) -> str:
        return "AnalysisFilter: skip analysis-only visits"

    def _service_matches(self, service: dict[str, Any]) -> bool:
        for field in _SERVICE_CODE_FIELDS:
            value: str = service.get(field) or ""
            if self._CODE_RE.search(value):
                return True
        return False

    def should_skip(self, visit: dict[str, Any]) -> bool:
        services: list[dict] = visit.get(_УСЛУГИ) or []
        if not services:
            return False
        return all(self._service_matches(s) for s in services)


class CardFilter:
    """Holds a list of filter strategies and applies them to appointments.

    Card-level: a visit is skipped if ANY strategy's should_skip returns True.
    Diagnosis-level: a diagnosis is skipped if ANY strategy's skip_diagnosis returns True.
    """

    def __init__(self, strategies: list[_BaseFilter]) -> None:
        self.strategies = strategies

    def __str__(self) -> str:
        if not self.strategies:
            return "(none)"
        return "\n".join(f"  {s}" for s in self.strategies)

    def filter(
        self,
        appointments: list[dict[str, Any]],
        done_guids: set[str],
    ) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]], int, int]:
        """Filter appointments into pending and ignored lists.

        Args:
            appointments: Raw visit list from 1C.
            done_guids:   Set of already-audited GUIDs (always-on dedup).

        Returns:
            (pending, ignored_visits, skipped_done, skipped_strategy)
            - pending:          [(original_index, visit), ...] — visits to audit
            - ignored_visits:   visits skipped by a strategy (for upsert_ignored)
            - skipped_done:     count of GUID-dedup skips
            - skipped_strategy: count of strategy skips
        """
        normalized_done = {str(g).lower() for g in done_guids}
        pending: list[tuple[int, dict[str, Any]]] = []
        ignored: list[dict[str, Any]] = []
        skipped_done = 0
        skipped_strategy = 0

        for idx, visit in enumerate(appointments):
            priem: dict = visit.get("Прием") or {}
            guid_raw = priem.get("GUID")
            visit_guid = str(guid_raw).lower() if guid_raw else None
            visit_id = guid_raw or priem.get("DATE") or f"#{idx + 1}"

            # Always-on dedup
            if visit_guid and visit_guid in normalized_done:
                skipped_done += 1
                logger.info(
                    "🩺 Skipping already audited visit %s (%d/%d)",
                    visit_guid, idx + 1, len(appointments),
                )
                continue

            if normalized_done and not visit_guid:
                logger.warning(
                    "🩺 Visit %s has no Прием.GUID; auditing because it cannot be matched to done_guids",
                    visit_id,
                )

            # Strategy-based filtering
            skip_reason: str | None = None
            for strategy in self.strategies:
                if strategy.should_skip(visit):
                    skip_reason = type(strategy).__name__
                    break

            if skip_reason:
                skipped_strategy += 1
                logger.info(
                    "🩺 Skipping visit %s (%d/%d) — filtered by %s",
                    visit_id, idx + 1, len(appointments), skip_reason,
                )
                ignored.append(visit)
                continue

            pending.append((idx, visit))

        return pending, ignored, skipped_done, skipped_strategy

    def should_skip_diagnosis(self, diagnosis: dict[str, Any]) -> bool:
        """Return True if any strategy wants to skip this specific diagnosis."""
        return any(s.skip_diagnosis(diagnosis) for s in self.strategies)
