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
from parsers.json_parser import patient_age as _patient_age  # noqa: F401 — re-export для audit.pipeline
from storage.guidelines_storage import GuidelinesStorage
from storage.models.guideline import Guideline

# МКБ codes for which no guideline lookup is needed (e.g. routine checkup codes).
_SKIP_CODES: frozenset[str] = frozenset({"Z00.1"})

# Граница «дети / взрослые». Рубрикатор относит к детям 0–17 лет
# (`manifest.csv` знает только «Дети», «Взрослые», «Взрослые, дети»), 404н
# задаёт объём ПМО для «граждан в возрасте 18 лет и старше». Прежние 15 лет
# пришли из коммита 720637d без обоснования и отсекали 16–17-летних от детских
# КР; 15 лет в 323-ФЗ — про самостоятельное информированное согласие, а не про
# применимость рекомендации. Та же константа в
# audit.formal_structure.validator._ADULT_AGE.
_ADULT_AGE = 18


def _is_age_eligible(guideline: "Guideline", age: int | None) -> bool:
    """Return False if the guideline's age category contradicts the patient's age."""
    if age is None:
        return True
    cats = {c.strip().lower() for c in guideline.age_category}
    is_child = age < _ADULT_AGE
    has_child = "дети" in cats
    has_adult = "взрослые" in cats
    if has_child and not has_adult:
        return is_child
    if has_adult and not has_child:
        return not is_child
    return True  # оба или неизвестно — пропускаем


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
