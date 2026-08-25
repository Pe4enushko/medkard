"""JSONL(.gz) дамп реестра БАД — формат синка для движка.

Читает его `scripts/sgr/sgr_sync.py --from-dump` в движке; набор ключей обязан
совпадать с `references/supplements/sync/client.py::COLS`, иначе синк молча
положит в реестр строки без половины полей.

Шапки с датой выгрузки здесь нет, в отличие от дампа ГРЛС: даты реестра СГР в
источнике не существует вовсе — есть только дата оформления каждого документа.
Подставить дату выгрузки файла и выдать её за дату реестра значило бы соврать о
том, насколько данные свежие.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Iterable

from sgr.parser import Row

# Ровно те ключи, что читает движок. Список продублирован намеренно: молчаливое
# расхождение здесь обнаружилось бы только пустыми полями в справочнике у врача.
ENGINE_COLS = (
    "registration_number", "status", "product_name", "manufacturer_name",
    "country_of_manufacture", "scope_of_application", "label_info", "registered_at",
)


def row_to_dict(row: Row) -> dict:
    out = asdict(row)
    value = out.get("registered_at")
    out["registered_at"] = value.isoformat() if isinstance(value, date) else None
    return {k: out[k] for k in ENGINE_COLS}


def write_dump(path: Path, rows: Iterable[Row]) -> int:
    opener = gzip.open if str(path).endswith(".gz") else open
    n = 0
    with opener(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row_to_dict(row), ensure_ascii=False) + "\n")
            n += 1
    return n
