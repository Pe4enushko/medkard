import zipfile
from datetime import date
from pathlib import Path

import pytest

from grls import status as st
from grls.parser import GrlsFormatError, build_record, read_archive, read_sheet
from tests.grls_fixtures import HEADERS, make_sheet, sample_row


def test_read_sheet_marker_date_and_rows(tmp_path: Path):
    p = make_sheet(tmp_path / "a.xlsx", st.STATUS_ACTIVE,
                   [sample_row(), sample_row(reg_number="ЛП-000002", trade_name="Другой")])
    res = read_sheet(p)
    assert res.status == st.STATUS_ACTIVE
    assert res.registry_date == date(2026, 8, 17)
    assert res.skipped is False
    assert [r.reg_number for r in res.records] == ["ЛП-000001", "ЛП-000002"]


def test_trailer_row_is_not_a_record(tmp_path: Path):
    p = make_sheet(tmp_path / "a.xlsx", st.STATUS_ACTIVE, [sample_row()], trailer=True)
    assert len(read_sheet(p).records) == 1


def test_record_fields_are_normalized(tmp_path: Path):
    p = make_sheet(tmp_path / "a.xlsx", st.STATUS_EXPIRED,
                   [sample_row(expires_at="31.12.2025", inn_name="~", is_vital="Нет")])
    rec = read_sheet(p).records[0]
    assert rec.status == st.STATUS_EXPIRED
    assert rec.registered_at == date(2020, 2, 1)
    assert rec.expires_at == date(2025, 12, 31)
    assert rec.annulled_at is None
    assert rec.inn_name is None
    assert rec.is_vital is False
    assert rec.narcotic_list is None
    assert rec.is_orphan is None
    assert rec.forms == ["таблетки, 5 мг, 10 шт. - блистеры - пачки картонные - По рецепту"]
    assert rec.dosage_forms == ["таблетки"]
    assert rec.dispensing == ["По рецепту"]
    assert rec.is_substance is False
    assert rec.normative_docs == "ЛП-000001-010220\nИзм. №1"
    assert len(rec.row_hash) == 64


def test_substance_row_flagged(tmp_path: Path):
    p = make_sheet(tmp_path / "a.xlsx", st.STATUS_ACTIVE, [sample_row(
        reg_number="ФС-000001", trade_name="Норфлоксацин",
        forms_raw="субстанция-порошок, ~, 25 кг - пакеты - барабаны - Не указано;")])
    rec = read_sheet(p).records[0]
    assert rec.is_substance is True
    assert rec.dosage_forms == ["субстанция-порошок"]
    assert rec.dispensing == ["Не указано"]


def test_changed_sheet_is_skipped(tmp_path: Path):
    p = make_sheet(tmp_path / "ch.xlsx", st.STATUS_CHANGED, [sample_row()])
    res = read_sheet(p)
    assert res.skipped is True
    assert res.records == []
    assert res.status == st.STATUS_CHANGED


def test_unknown_marker_raises(tmp_path: Path):
    p = make_sheet(tmp_path / "x.xlsx", "Неведомый", [sample_row()])
    with pytest.raises(GrlsFormatError):
        read_sheet(p)


def test_wrong_headers_raise(tmp_path: Path):
    bad = list(HEADERS)
    bad[0] = "Номер чего-то другого"
    p = make_sheet(tmp_path / "x.xlsx", st.STATUS_ACTIVE, [sample_row()], headers=tuple(bad))
    with pytest.raises(GrlsFormatError):
        read_sheet(p)


def test_missing_registry_date_raises(tmp_path: Path):
    p = make_sheet(tmp_path / "x.xlsx", st.STATUS_ACTIVE, [sample_row()],
                   title="Государственный реестр лекарственных средств")
    with pytest.raises(GrlsFormatError):
        read_sheet(p)


def test_build_record_skips_empty_and_nameless():
    assert build_record(st.STATUS_ACTIVE, ("",) * 15) is None
    assert build_record(st.STATUS_ACTIVE, ("17.08.2026 05:00:00",) + (None,) * 14) is None
    assert build_record(st.STATUS_ACTIVE, sample_row(trade_name="")) is None


def test_read_archive_zip_and_dir(tmp_path: Path):
    d = tmp_path / "xlsx"
    d.mkdir()
    make_sheet(d / "1.xlsx", st.STATUS_ACTIVE, [sample_row()])
    make_sheet(d / "2.xlsx", st.STATUS_CHANGED, [sample_row()])
    zpath = tmp_path / "grls.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for f in sorted(d.iterdir()):
            # long/garbled names like the real export must not matter
            z.write(f, arcname="grls2026-08-17-1-" + "Действующий" * 20 + f.name)
    from_dir = read_archive(d)
    from_zip = read_archive(zpath)
    for results in (from_dir, from_zip):
        assert sorted(r.status for r in results) == sorted([st.STATUS_ACTIVE, st.STATUS_CHANGED])
        assert sum(len(r.records) for r in results) == 1
