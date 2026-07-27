import importlib.util
from datetime import datetime
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "backfill_priem",
    Path(__file__).resolve().parent.parent / "scripts" / "backfill-priem.py")
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


def test_changed_keys_reports_new_and_differing_only():
    stored = {"GUID": "ab-1", "DATE": "01.07.2026", "Кабинет": "5"}
    fresh = {"GUID": "ab-1", "DATE": "02.07.2026", "Филиал": "Центр"}
    assert backfill._changed_keys(stored, fresh) == ["DATE", "Филиал"]
