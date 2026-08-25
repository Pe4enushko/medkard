"""Разбор выгрузки реестра СГР ЕАЭС (БАД) — чистая логика, без БД.

Формат выгрузки: CSV с разделителем «;», шапка на первой строке, кодировка
Windows-1251. Колонок в выгрузке 2026-08-24 — 43; в предыдущей их было 39, и
именно на это упирался прежний загрузчик `scripts/knowledge/seed-reference-lists.sh`: у
него ширина staging-таблицы прибита в SQL (`col01 … col39`), и `\\COPY` падает с
«extra data after last expected column», как только регулятор добавит поле.

Поэтому здесь колонки берутся ПО ИМЕНИ ИЗ ШАПКИ, а не по позиции. Имена
устойчивее номеров, а если нужного всё-таки нет — ошибка называет пропавшее
поле, а не позицию, по которой ещё надо догадаться.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Колонка выгрузки → поле dietary_supplements. Восемь из сорока трёх: остальное
# либо служебное (коды организаций, серия бланка), либо не нужно поиску.
# Имена сверены с выгрузкой 2026-08-24; хвостовые пробелы у регулятора в шапке
# встречаются («Информация, наносимая на этикетку »), поэтому имена нормализуются.
COLUMNS: dict[str, str] = {
    "номер свидетельства": "registration_number",
    "статус": "status",
    "дата оформления документа": "registered_at",
    "наименование продукции": "product_name",
    "наименование изготовителя": "manufacturer_name",
    "страна изготовителя продукции": "country_of_manufacture",
    "область применения": "scope_of_application",
    "информация, наносимая на этикетку": "label_info",
}
# Без наименования запись бесполезна: искать по ней нечего и показать врачу
# нечего. Остальные семь могут быть пустыми — это законное состояние.
REQUIRED = ("наименование продукции",)

# Порядок проб. Регулятор отдаёт CP1251; utf-8-sig оставлен на случай, если
# выгрузку пересохранили руками, а не скачали.
ENCODINGS = ("cp1251", "utf-8-sig")

CSV_FIELD_LIMIT = 10 ** 8


@dataclass
class Row:
    """Строка реестра — ровно те поля, что переносятся в БД."""

    product_name: str
    registration_number: str | None = None
    status: str | None = None
    manufacturer_name: str | None = None
    country_of_manufacture: str | None = None
    scope_of_application: str | None = None
    label_info: str | None = None
    registered_at: date | None = None


@dataclass
class ParseResult:
    rows: list[Row]
    duplicates_dropped: int
    skipped_no_name: int
    encoding: str
    columns: int


class SgrFormatError(RuntimeError):
    """Выгрузка не похожа на реестр СГР. Скрипт печатает и выходит."""


def _norm_header(name: str) -> str:
    return " ".join(name.replace("﻿", "").strip().strip('"').lower().split())


def _parse_date(value: str) -> date | None:
    """Дата оформления — «24.08.2026». ISO принимается тоже: выгрузка держит оба
    вида (даты действия записи идут как «2026-08-24T00:00»), и разбирать их
    разными правилами значило бы завести второй источник ошибок."""
    text = (value or "").strip()
    if not text:
        return None
    text = text.split("T")[0]
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            from datetime import datetime
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _decode(path: Path) -> tuple[str, str]:
    """Текст файла и кодировка, в которой он прочитался.

    Кодировку определяем по ШАПКЕ, а не по всему файлу: кириллица в CP1251
    декодируется как utf-8 с ошибкой почти сразу, а гонять 140 МБ ради этого
    незачем.
    """
    head = path.read_bytes()[:4096]
    for enc in ENCODINGS:
        try:
            head.decode(enc)
        except UnicodeDecodeError:
            continue
        return path.read_text(encoding=enc), enc
    raise SgrFormatError(
        f"не удалось прочитать {path.name} ни в одной из кодировок: {', '.join(ENCODINGS)}")


def read_export(path: Path) -> ParseResult:
    """Выгрузка → строки реестра. Дубликаты по всем переносимым полям схлопнуты.

    Схлопывание здесь, а не в БД: у dietary_supplements ключ суррогатный, и
    ON CONFLICT ловить нечего, а в выгрузке один и тот же продукт встречается
    несколько раз (перерегистрации с тем же содержимым).
    """
    csv.field_size_limit(CSV_FIELD_LIMIT)
    text, encoding = _decode(path)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=";")
    try:
        header = next(reader)
    except StopIteration:
        raise SgrFormatError(f"{path.name} пуст")

    index: dict[str, int] = {}
    for pos, name in enumerate(header):
        key = _norm_header(name)
        if key in COLUMNS and key not in index:
            index[key] = pos
    missing = [c for c in REQUIRED if c not in index]
    if missing:
        raise SgrFormatError(
            f"в шапке {path.name} нет обязательных колонок: {', '.join(missing)}. "
            f"Прочитано {len(header)} колонок: {', '.join(_norm_header(h) for h in header[:8])}…"
        )

    rows: list[Row] = []
    seen: set[tuple] = set()
    duplicates = skipped = 0
    for raw in reader:
        if len(raw) <= index[REQUIRED[0]]:
            skipped += 1
            continue
        values: dict[str, object] = {}
        for key, field in COLUMNS.items():
            pos = index.get(key)
            cell = raw[pos].strip() if pos is not None and pos < len(raw) else ""
            values[field] = _parse_date(cell) if field == "registered_at" else (cell or None)
        if not values.get("product_name"):
            skipped += 1
            continue
        row = Row(**values)  # type: ignore[arg-type]
        key_tuple = tuple(getattr(row, f) for f in (
            "registration_number", "product_name", "status", "registered_at",
            "manufacturer_name", "country_of_manufacture", "scope_of_application",
            "label_info"))
        if key_tuple in seen:
            duplicates += 1
            continue
        seen.add(key_tuple)
        rows.append(row)

    if not rows:
        raise SgrFormatError(f"в {path.name} не нашлось ни одной записи с наименованием продукции")
    return ParseResult(rows=rows, duplicates_dropped=duplicates, skipped_no_name=skipped,
                       encoding=encoding, columns=len(header))
