from datetime import date

import pytest

from grls import status as st
from grls.status import StatusAtVisit, status_at
from storage.models.grls_record import GrlsRecord


def _rec(status: str, expires_at=None, annulled_at=None) -> GrlsRecord:
    return GrlsRecord(status=status, reg_number="ЛП-000001", trade_name="Тест",
                      row_hash="h", expires_at=expires_at, annulled_at=annulled_at)


VISIT = date(2025, 3, 10)


def test_constants_are_consistent():
    assert len(st.ALL_STATUSES) == 7
    assert set(st.STATUS_RANK) == set(st.ALL_STATUSES)
    assert st.LIVE_STATUSES == {st.STATUS_ACTIVE, st.STATUS_EAEU,
                                st.STATUS_CONFIRMING, st.STATUS_FOREIGN_PACK}
    assert st.STATUS_CHANGED not in st.ALL_STATUSES


@pytest.mark.parametrize("status", [st.STATUS_ACTIVE, st.STATUS_EAEU])
def test_live_is_active_even_if_expires_in_past(status):
    # status wins over dates: registry has 87 such rows
    assert status_at(_rec(status, expires_at=date(2020, 1, 1)), VISIT) is StatusAtVisit.ACTIVE
    assert status_at(_rec(status), None) is StatusAtVisit.ACTIVE


@pytest.mark.parametrize("status", [st.STATUS_CONFIRMING, st.STATUS_FOREIGN_PACK, st.STATUS_SUSPENDED])
def test_note_statuses(status):
    assert status_at(_rec(status), VISIT) is StatusAtVisit.ACTIVE_WITH_NOTE


def test_expired_before_visit():
    assert status_at(_rec(st.STATUS_EXPIRED, expires_at=date(2025, 1, 1)), VISIT) is StatusAtVisit.EXPIRED


def test_expired_after_visit_is_valid_at_visit():
    assert status_at(_rec(st.STATUS_EXPIRED, expires_at=date(2025, 12, 31)), VISIT) is StatusAtVisit.VALID_AT_VISIT


def test_expired_on_visit_day_is_valid():
    assert status_at(_rec(st.STATUS_EXPIRED, expires_at=VISIT), VISIT) is StatusAtVisit.VALID_AT_VISIT


def test_expired_without_visit_date_is_expired():
    assert status_at(_rec(st.STATUS_EXPIRED, expires_at=date(2025, 12, 31)), None) is StatusAtVisit.EXPIRED


def test_expired_without_boundary_is_unknown():
    assert status_at(_rec(st.STATUS_EXPIRED), VISIT) is StatusAtVisit.UNKNOWN_END


def test_annulled_uses_annulled_at_first():
    r = _rec(st.STATUS_ANNULLED, expires_at=date(2030, 1, 1), annulled_at=date(2024, 2, 14))
    assert status_at(r, VISIT) is StatusAtVisit.ANNULLED


def test_annulled_falls_back_to_expires_at():
    r = _rec(st.STATUS_ANNULLED, expires_at=date(2025, 6, 1))
    assert status_at(r, VISIT) is StatusAtVisit.VALID_AT_VISIT


def test_annulled_without_any_date_is_unknown():
    assert status_at(_rec(st.STATUS_ANNULLED), VISIT) is StatusAtVisit.UNKNOWN_END


def test_unknown_status_raises():
    with pytest.raises(ValueError):
        status_at(_rec("Изменённый"), VISIT)
