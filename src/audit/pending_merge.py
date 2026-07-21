"""
audit/pending_merge.py — folds pending (pushed) done_cards rows into a 1C
payload's visit list before it's handed to AuditPipeline.run_batched.

Cards pushed via POST /cards/push are stored with status='pending' and never
re-appear from a normal 1C date-range pull if their visit date falls outside
that night's window — merging them in here is what lets a pushed update
actually get (re-)audited.

A card can appear in both sources in the same run (pushed, and also due for
tonight's normal 1C pull) — deduped by card_guid, keeping the first
occurrence, so it's only audited once per run.
"""

from __future__ import annotations

from typing import Any

from parsers.json_parser import AppointmentParser


def _card_guid(visit: dict[str, Any]) -> str | None:
    priem = visit.get("Прием") or {}
    guid = priem.get("GUID")
    return str(guid).lower() if guid else None


def merge_pending_cards(payload: dict | list | str, pending_rows: list[dict]) -> list[dict[str, Any]]:
    """Return the payload's visits plus every pending row's raw card_data, deduped by card_guid
    (first occurrence wins; visits without an extractable guid are never deduped against each other).
    """
    visits = AppointmentParser.split(payload)
    pending_visits = [row["card_data"] for row in pending_rows]

    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for visit in visits + pending_visits:
        guid = _card_guid(visit)
        if guid:
            if guid in seen:
                continue
            seen.add(guid)
        merged.append(visit)
    return merged
