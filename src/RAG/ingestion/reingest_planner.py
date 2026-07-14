"""reingest_planner.py — pure work-list classification + PDF hashing for reingest.

No DB and no fitz imports — unit-testable in isolation. See
docs/superpowers/specs/2026-07-09-reingest-with-resume-design.md (work-list).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from storage.models.guideline import Guideline

Decision = Literal["full", "metadata_only", "skip"]


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file's bytes (hex)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def classify(
    *,
    status: str | None,
    stored_hash: str | None,
    current_hash: str,
    stored_guideline: Guideline | None,
    new_guideline: Guideline,
) -> Decision:
    """Decide reingest action for one manifest file.

    - full          — no ingest_runs row, or status != 'done', or the PDF hash
                      differs from the last successful ('done') hash.
    - metadata_only — file is done and PDF unchanged, but the manifest row
                      differs from the stored guideline (normalized Guideline).
    - skip          — done, hash matches, metadata matches.
    """
    if status != "done" or stored_hash != current_hash:
        return "full"
    if new_guideline != stored_guideline:
        return "metadata_only"
    return "skip"


def build_worklist(manifest_rows, runs, guidelines_by_id, hash_of):
    """Pure work-list: -> list[(file_id, decision)] over manifest files present on disk.

    manifest_rows:     {file_id: raw csv row dict}
    runs:              {file_id: (status, content_hash)}   (ingest_runs snapshot)
    guidelines_by_id:  {file_id: Guideline}                (stored 'old' manifest snapshot)
    hash_of:           callable file_id -> current sha256 hex, or None if PDF missing
    """
    out: list[tuple[str, Decision]] = []
    for file_id, row in manifest_rows.items():
        current_hash = hash_of(file_id)
        if current_hash is None:
            continue  # PDF missing on disk; loader logs it too
        status, stored_hash = runs.get(file_id, (None, None))
        decision = classify(
            status=status,
            stored_hash=stored_hash,
            current_hash=current_hash,
            stored_guideline=guidelines_by_id.get(file_id),
            new_guideline=Guideline.from_manifest_row(row),
        )
        out.append((file_id, decision))
    return out
