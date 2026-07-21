"""
audit/pending_merge.py — folds pending (pushed) done_cards rows into a 1C
payload's visit list before it's handed to AuditPipeline.run_batched.

Cards pushed via POST /cards/push are stored with status='pending' and never
re-appear from a normal 1C date-range pull if their visit date falls outside
that night's window — merging them in here is what lets a pushed update
actually get (re-)audited.
"""

from __future__ import annotations

from typing import Any

from parsers.json_parser import AppointmentParser


def merge_pending_cards(payload: dict | list | str, pending_rows: list[dict]) -> list[dict[str, Any]]:
    """Return the payload's visits plus every pending row's raw card_data, as one flat list."""
    visits = AppointmentParser.split(payload)
    pending_visits = [row["card_data"] for row in pending_rows]
    return visits + pending_visits
