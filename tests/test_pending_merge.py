"""
Unit tests for audit.pending_merge.merge_pending_cards: folds pending
(pushed) done_cards rows into a 1C payload's visit list, regardless of
whether the payload is a bare list or an {"appointments": [...]} wrapper.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.pending_merge import merge_pending_cards


def test_merge_appends_pending_cards_to_bare_list_payload():
    payload = [{"Прием": {"GUID": "a"}}]
    pending = [{"card_guid": "b", "card_data": {"Прием": {"GUID": "b"}}}]
    merged = merge_pending_cards(payload, pending)
    guids = [v["Прием"]["GUID"] for v in merged]
    assert guids == ["a", "b"]


def test_merge_appends_pending_cards_to_wrapper_dict_payload():
    payload = {"appointments": [{"Прием": {"GUID": "a"}}]}
    pending = [{"card_guid": "b", "card_data": {"Прием": {"GUID": "b"}}}]
    merged = merge_pending_cards(payload, pending)
    guids = [v["Прием"]["GUID"] for v in merged]
    assert guids == ["a", "b"]


def test_merge_with_no_pending_cards_returns_payload_visits_unchanged():
    payload = [{"Прием": {"GUID": "a"}}]
    merged = merge_pending_cards(payload, [])
    assert merged == payload
