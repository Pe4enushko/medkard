import importlib.util
import json
from datetime import datetime
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "backfill_priem",
    Path(__file__).resolve().parent.parent / "scripts" / "operator" / "backfill-priem.py")
backfill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill)


def _d(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def test_date_range_single_day():
    assert list(backfill._date_range(_d("2026-07-01"), _d("2026-07-01"))) == [_d("2026-07-01")]


def test_date_range_spans_month_boundary():
    assert list(backfill._date_range(_d("2026-06-30"), _d("2026-07-02"))) == [
        _d("2026-06-30"), _d("2026-07-01"), _d("2026-07-02"),
    ]


def test_parse_date_accepts_iso_only():
    assert backfill._parse_date("2026-07-01", "--since") == _d("2026-07-01")
    try:
        backfill._parse_date("01.07.2026", "--since")
    except SystemExit as exc:
        assert "YYYY-MM-DD" in str(exc)
    else:
        raise AssertionError("expected SystemExit for non-ISO date")


def test_visit_priem_extracts_guid():
    guid, priem = backfill._visit_priem({"Прием": {"GUID": "AB-1", "DATE": "01.07.2026"}})
    assert guid == "AB-1"
    assert priem == {"GUID": "AB-1", "DATE": "01.07.2026"}


def test_visit_priem_missing_block():
    assert backfill._visit_priem({"Пациент": {}}) == (None, {})
    assert backfill._visit_priem({"Прием": {"DATE": "01.07.2026"}}) == (None, {"DATE": "01.07.2026"})


def test_diff_keys_reports_set_and_dropped():
    stored = {"GUID": "ab-1", "DATE": "01.07.2026", "Кабинет": "5"}
    fresh = {"GUID": "ab-1", "DATE": "02.07.2026", "Филиал": "Центр"}
    assert backfill._diff_keys(stored, fresh) == "set: DATE, Филиал; dropped: Кабинет"


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload
        self.requests = []

    def fetch_json_for_period(self, datebegin, dateend):
        self.requests.append((datebegin, dateend))
        return self._payload


class _FakeStorage:
    def __init__(self, stored):
        self._stored = stored
        self.replaced = {}

    async def get_priem(self, card_guid):
        return self._stored.get(card_guid)

    async def replace_priem(self, *, card_guid, priem):
        self.replaced[card_guid] = json.loads(priem)
        return True


def _visit(guid, **priem):
    return {"Прием": {"GUID": guid, **priem} if guid else priem}


def _totals():
    return {"visits": 0, "updated": 0, "unchanged": 0, "not_found": 0, "no_guid": 0}


async def test_process_day_replaces_only_differing_cards():
    client = _FakeClient([
        _visit("changed", DATE="02.07.2026"),
        _visit("same", DATE="01.07.2026"),
        _visit("missing", DATE="01.07.2026"),
        _visit(None, DATE="01.07.2026"),
    ])
    storage = _FakeStorage({
        "changed": {"GUID": "changed", "DATE": "01.07.2026"},
        "same": {"GUID": "same", "DATE": "01.07.2026"},
    })
    totals = _totals()

    await backfill._process_day(_d("2026-07-01"), client, storage, False, totals)

    assert client.requests == [("01.07.2026", "01.07.2026")]
    assert list(storage.replaced) == ["changed"]
    assert totals == {"visits": 4, "updated": 1, "unchanged": 1, "not_found": 1, "no_guid": 1}


async def test_process_day_replaces_block_wholesale_dropping_stale_keys():
    client = _FakeClient([_visit("g1", DATE="01.07.2026")])
    storage = _FakeStorage({
        "g1": {"GUID": "g1", "DATE": "01.07.2026", "Устаревший": "мусор"},
    })
    totals = _totals()

    await backfill._process_day(_d("2026-07-01"), client, storage, False, totals)

    assert storage.replaced["g1"] == {"GUID": "g1", "DATE": "01.07.2026"}
    assert totals["updated"] == 1


async def test_process_day_dry_run_writes_nothing():
    client = _FakeClient([_visit("changed", DATE="02.07.2026")])
    storage = _FakeStorage({"changed": {"GUID": "changed", "DATE": "01.07.2026"}})
    totals = _totals()

    await backfill._process_day(_d("2026-07-01"), client, storage, True, totals)

    assert storage.replaced == {}
    assert totals["updated"] == 1
