"""
clinic_recs.py — map an МКБ-10 code to clinical-guideline file_ids via GuidelinesStorage.

Usage::
    from audit.diagnosis.clinic_recs import ClinicRecs

    recs = ClinicRecs()
    file_id = await recs.pick_recs(patient, diagnosis)  # str | None
"""

from __future__ import annotations

from typing import Any

from LLM.decider import decide_file_id
from LLM.icd_prefix_picker import IcdPrefixPicker
from storage.guidelines_storage import GuidelinesStorage
from storage.models.guideline import Guideline

# МКБ codes for which no guideline lookup is needed (e.g. routine checkup codes).
_SKIP_CODES: frozenset[str] = frozenset({"Z00.1"})

_ADULT_THRESHOLD = 15  # age > this → adult


def _is_age_eligible(guideline: "Guideline", age: int | None) -> bool:
    """Return False if the guideline's age category contradicts the patient's age."""
    if age is None:
        return True
    cats = {c.strip().lower() for c in guideline.age_category}
    is_child = age <= _ADULT_THRESHOLD
    has_child = "дети" in cats
    has_adult = "взрослые" in cats
    if has_child and not has_adult:
        return is_child
    if has_adult and not has_child:
        return not is_child
    return True  # оба или неизвестно — пропускаем


def _patient_age(patient: dict[str, Any]) -> int | None:
    raw = patient.get("AGE") or patient.get("Возраст")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


class ClinicRecs:
    """Look up a clinical-guideline file_id via GuidelinesStorage for the given diagnosis."""

    def __init__(self) -> None:
        self._prefix_picker = IcdPrefixPicker()

    async def pick_recs(
        self,
        patient: dict[str, Any],
        diagnosis: dict[str, Any],
    ) -> tuple[str | None, int]:
        """Return (file_id, tokens) for *diagnosis*.

        Args:
            patient:   Patient info dict (e.g. ``{"Возраст": ..., "Пол": ...}``).
            diagnosis: Diagnosis dict with at least ``КодМКБ`` key.

        Returns:
            A tuple of (manifest ID string or None, total LLM tokens spent).
        """
        icd_raw: str = diagnosis.get("КодМКБ", "")
        normalised = icd_raw.strip().upper()

        if not normalised or normalised in _SKIP_CODES:
            return None, 0

        age = _patient_age(patient)
        async with GuidelinesStorage() as store:
            matched = [g for g in await store.find_by_code(normalised) if _is_age_eligible(g, age)]

            if not matched:
                prefix = normalised.split(".")[0]
                if prefix != normalised:
                    candidates = [g for g in await store.find_by_prefix(prefix) if _is_age_eligible(g, age)]
                    if candidates:
                        return await self._prefix_picker.pick(patient, diagnosis, candidates)
                return None, 0

        if len(matched) == 1:
            return matched[0].file_id or None, 0

        diag_name: str = diagnosis.get("НаименованиеМКБ", "").lower()
        diag_tokens = set(diag_name.split())
        scores = [len(diag_tokens & set((g.name or "").lower().split())) for g in matched]
        best_score = max(scores)
        if best_score > 0:
            best = matched[scores.index(best_score)]
            return best.file_id or None, 0

        return await decide_file_id(patient, diagnosis, matched)
