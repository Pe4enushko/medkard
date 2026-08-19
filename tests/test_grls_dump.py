import gzip
import json
from datetime import date
from pathlib import Path

import pytest

from grls import status as st
from grls.dump import read_dump, record_from_dict, record_to_dict, write_dump
from grls.parser import build_record
from tests.grls_fixtures import sample_row


def _records():
    return [build_record(st.STATUS_ACTIVE, sample_row()),
            build_record(st.STATUS_EXPIRED, sample_row(reg_number="ЛП-000002", expires_at="31.12.2025"))]


def test_record_dict_roundtrip_iso_dates_no_id():
    rec = _records()[1]
    d = record_to_dict(rec)
    assert "id" not in d and "imported_at" not in d
    assert d["expires_at"] == "2025-12-31"
    assert d["forms"] == rec.forms
    back = record_from_dict(d)
    assert back == rec


def test_record_from_dict_rejects_hash_mismatch():
    d = record_to_dict(_records()[0])
    d["trade_name"] = "Подмена"
    with pytest.raises(ValueError):
        record_from_dict(d)


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz"])
def test_write_and_read_dump(tmp_path: Path, suffix: str):
    p = tmp_path / f"grls{suffix}"
    n = write_dump(p, _records(), registry_date=date(2026, 8, 17), archive_name="grls2026-08-17-1.zip")
    assert n == 2
    opener = gzip.open if suffix.endswith(".gz") else open
    with opener(p, "rt", encoding="utf-8") as fh:
        first = json.loads(fh.readline())
    assert first == {"_meta": {"registry_date": "2026-08-17", "archive_name": "grls2026-08-17-1.zip", "row_count": 2}}
    meta, records = read_dump(p)
    assert meta["registry_date"] == "2026-08-17"
    assert [r.reg_number for r in records] == ["ЛП-000001", "ЛП-000002"]
