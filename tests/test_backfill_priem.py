import importlib.util
from datetime import datetime
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "backfill_priem",
    Path(__file__).resolve().parent.parent / "scripts" / "backfill-priem.py")
backfill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill)


def _d(value: str) -> datetime:
    return datetime.strptime(value, "%d.%m.%Y")


def test_date_range_single_day():
    assert list(backfill._date_range(_d("01.07.2026"), _d("01.07.2026"))) == ["01.07.2026"]


def test_date_range_spans_month_boundary():
    assert list(backfill._date_range(_d("30.06.2026"), _d("02.07.2026"))) == [
        "30.06.2026", "01.07.2026", "02.07.2026",
    ]


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
