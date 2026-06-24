"""
clinic_recs.py — map an МКБ-10 code to clinical-guideline file_ids from manifest.csv.

Usage::
    from audit.diagnosis.clinic_recs import ClinicRecs

    recs = ClinicRecs()
    file_id = await recs.pick_recs(patient, diagnosis)  # str | None
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from LLM.decider import decide_file_id
from LLM.icd_prefix_picker import IcdPrefixPicker

# Path to manifest — resolved relative to the project root.
_MANIFEST_PATH: Path = Path(__file__).resolve().parent.parent.parent.parent / "resources" / "manifest.csv"

# МКБ codes for which no guideline lookup is needed (e.g. routine checkup codes).
_SKIP_CODES: frozenset[str] = frozenset({"Z00.1"})

_ICD_COLUMN = "МКБ-10"
_ID_COLUMN = "ID"
_NAME_COLUMN = "Наименование"
_AGE_COLUMN = "Возрастная категория"
_ADULT_THRESHOLD = 15  # age > this → adult


def _is_age_eligible(row: dict[str, str], age: int | None) -> bool:
    """Return False if the row's age category contradicts the patient's age."""
    if age is None:
        return True
    cat = row.get(_AGE_COLUMN, "").strip().lower()
    is_child = age <= _ADULT_THRESHOLD
    if "дети" in cat and "взрослые" not in cat:
        return is_child
    if "взрослые" in cat and "дети" not in cat:
        return not is_child
    return True  # "Взрослые, дети" or unknown — keep


def _patient_age(patient: dict[str, Any]) -> int | None:
    raw = patient.get("AGE") or patient.get("Возраст")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


class ClinicRecs:
    """Look up a clinical-guideline file_id in manifest.csv for the given diagnosis.

    The manifest may contain comma-separated codes in a single cell, e.g.
    ``"J06.0, J06.9"``.  Each cell is split and each code is normalised
    (stripped of whitespace, upper-cased) before comparison.
    """

    def __init__(self, manifest_path: Path = _MANIFEST_PATH) -> None:
        self._manifest_path = manifest_path
        self._prefix_picker = IcdPrefixPicker()

    def _load_manifest(self) -> list[dict[str, str]]:
        with open(self._manifest_path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def _find_matching_rows_by_prefix(self, prefix: str) -> list[dict[str, str]]:
        """Return manifest rows where any МКБ-10 code starts with *prefix*."""
        rows = self._load_manifest()
        matched: list[dict[str, str]] = []
        for row in rows:
            raw_codes: str = row.get(_ICD_COLUMN, "")
            cell_codes = [c.strip().upper() for c in raw_codes.split(",")]
            if any(c.split(".")[0] == prefix for c in cell_codes):
                fid = row.get(_ID_COLUMN, "").strip()
                if fid:
                    matched.append(row)
        return matched

    def _find_matching_rows(self, normalised_code: str) -> list[dict[str, str]]:
        """Return manifest rows whose МКБ-10 cell contains *normalised_code*."""
        rows = self._load_manifest()
        matched: list[dict[str, str]] = []
        for row in rows:
            raw_codes: str = row.get(_ICD_COLUMN, "")
            cell_codes = [c.strip().upper() for c in raw_codes.split(",")]
            if normalised_code in cell_codes:
                fid = row.get(_ID_COLUMN, "").strip()
                if fid:
                    matched.append(row)
        return matched

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
        matched = [r for r in self._find_matching_rows(normalised) if _is_age_eligible(r, age)]

        if not matched:
            # ── Prefix fallback: strip .(\d+) subcategory (J20.9 → J20) ─────
            prefix = normalised.split(".")[0]
            if prefix != normalised:
                candidates = [r for r in self._find_matching_rows_by_prefix(prefix) if _is_age_eligible(r, age)]
                if candidates:
                    return await self._prefix_picker.pick(patient, diagnosis, candidates)
            return None, 0

        if len(matched) == 1:
            return matched[0].get(_ID_COLUMN, "").strip(), 0

        # ── Multiple candidates: BM25 token-overlap first ─────────────────────
        diag_name: str = diagnosis.get("НаименованиеМКБ", "").lower()
        diag_tokens = set(diag_name.split())

        scores = [
            len(diag_tokens & set(row.get(_NAME_COLUMN, "").lower().split()))
            for row in matched
        ]
        best_score = max(scores)

        if best_score > 0:
            # Unique winner — return its file_id without an LLM call.
            best_row = matched[scores.index(best_score)]
            return best_row.get(_ID_COLUMN, "").strip() or None, 0

        # ── BM25 impossible (all zero) — fall back to LLM decider ────────────
        return await decide_file_id(patient, diagnosis, matched)
