"""JSONL(.gz) dump of grls_registry — the sync format for engine (spec §7)."""
from __future__ import annotations

import gzip
import json
from dataclasses import fields
from datetime import date
from pathlib import Path
from typing import IO, Iterable

from grls.normalize import row_hash
from storage.models.grls_record import GrlsRecord

_EXCLUDED = {"id", "imported_at"}
_DATE_FIELDS = ("registered_at", "expires_at", "annulled_at")
_HASH_FIELDS = ("status", "reg_number", "registered_at", "expires_at", "annulled_at", "holder",
                "holder_country", "trade_name", "inn_name", "forms_raw", "production_stages",
                "normative_docs", "pharm_group", "is_vital", "narcotic_list", "is_orphan")


def record_to_dict(rec: GrlsRecord) -> dict:
    out = {}
    for f in fields(GrlsRecord):
        if f.name in _EXCLUDED:
            continue
        v = getattr(rec, f.name)
        out[f.name] = v.isoformat() if isinstance(v, date) else v
    return out


def record_from_dict(d: dict) -> GrlsRecord:
    data = {k: v for k, v in d.items() if k not in _EXCLUDED}
    for k in _DATE_FIELDS:
        if data.get(k):
            data[k] = date.fromisoformat(data[k])
        else:
            data[k] = None
    rec = GrlsRecord(**data)
    expected = row_hash(**{k: getattr(rec, k) for k in _HASH_FIELDS})
    if rec.row_hash != expected:
        raise ValueError(f"row_hash mismatch for {rec.reg_number} / {rec.trade_name}")
    return rec


def _open(path: Path, mode: str) -> IO[str]:
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def write_dump(path: Path, records: Iterable[GrlsRecord], *, registry_date: date, archive_name: str) -> int:
    records = list(records)
    with _open(Path(path), "w") as fh:
        fh.write(json.dumps({"_meta": {"registry_date": registry_date.isoformat(),
                                        "archive_name": archive_name,
                                        "row_count": len(records)}}, ensure_ascii=False) + "\n")
        for rec in records:
            fh.write(json.dumps(record_to_dict(rec), ensure_ascii=False) + "\n")
    return len(records)


def read_dump(path: Path) -> tuple[dict, list[GrlsRecord]]:
    with _open(Path(path), "r") as fh:
        first = json.loads(fh.readline())
        meta = first["_meta"]
        records = [record_from_dict(json.loads(line)) for line in fh if line.strip()]
    return meta, records
