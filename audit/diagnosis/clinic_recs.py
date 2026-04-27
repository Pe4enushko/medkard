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

# Path to manifest — resolved relative to the project root.
_MANIFEST_PATH: Path = Path(__file__).resolve().parent.parent.parent / "manifest.csv"

# МКБ codes for which no guideline lookup is needed (e.g. routine checkup codes).
_SKIP_CODES: frozenset[str] = frozenset({"Z00.1"})

_ICD_COLUMN = "МКБ-10"
_ID_COLUMN = "ID"
_NAME_COLUMN = "Наименование"


class ClinicRecs:
    """Look up a clinical-guideline file_id in manifest.csv for the given diagnosis.

    The manifest may contain comma-separated codes in a single cell, e.g.
    ``"J06.0, J06.9"``.  Each cell is split and each code is normalised
    (stripped of whitespace, upper-cased) before comparison.
    """

    def __init__(self, manifest_path: Path = _MANIFEST_PATH) -> None:
        self._manifest_path = manifest_path

    def _load_manifest(self) -> list[dict[str, str]]:
        with open(self._manifest_path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

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
    ) -> str | None:
        """Return the most relevant guideline file_id for *diagnosis*.

        Args:
            patient:   Patient info dict (e.g. ``{"Возраст": ..., "Пол": ...}``).
            diagnosis: Diagnosis dict with at least ``КодМКБ`` key.

        Returns:
            A single manifest ``ID`` string, or ``None`` when no match exists or
            when the ICD code is in the skip list (e.g. Z00.1).
        """
        icd_raw: str = diagnosis.get("КодМКБ", "")
        normalised = icd_raw.strip().upper()

        if not normalised or normalised in _SKIP_CODES:
            return None

        matched = self._find_matching_rows(normalised)

        if not matched:
            return None

        if len(matched) == 1:
            return matched[0].get(_ID_COLUMN, "").strip()

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
            return best_row.get(_ID_COLUMN, "").strip() or None

        # ── BM25 impossible (all zero) — fall back to LLM decider ────────────
        return await decide_file_id(patient, diagnosis, matched)
