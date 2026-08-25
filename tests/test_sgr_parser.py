"""Разбор выгрузки реестра СГР ЕАЭС.

Прежний загрузчик (scripts/knowledge/seed-reference-lists.sh) прибивал ширину выгрузки в
SQL — col01 … col39 — и кодировку к UTF-8. Выгрузка 2026-08-24 приехала с 43
колонками в CP1251 и загрузчик уронила. Эти тесты держат оба урока: колонки
берутся по имени, кодировка определяется.
"""
import gzip
import json
from datetime import date

import pytest

from sgr.dump import ENGINE_COLS, write_dump
from sgr.parser import SgrFormatError, read_export

HEADER = [
    "Номер свидетельства", "Статус", "Типографский номер бланка",
    "Дата оформления документа", "Наименование продукции",
    "Наименование изготовителя", "Наименование получателя",
    "Страна изготовителя продукции", "Код страны изготовителя продукции",
    "Юр. адрес изготовителя", "Юр. адрес получателя ", "Нормативная документация",
    "Область применения", "Протоколы исследований", "Условия хранения",
    "Информация, наносимая на этикетку ",
]


def _row(name="Биологически активная добавка к пище «Селен-актив»", number="RU.77.01.R.000001",
         status="подписан и действует", issued="24.08.2026", maker="Диод",
         country="Россия", scope="источник селена", label="состав: селен"):
    cells = [""] * len(HEADER)
    cells[0], cells[1], cells[3], cells[4] = number, status, issued, name
    cells[5], cells[7], cells[12], cells[15] = maker, country, scope, label
    return cells


def _write(tmp_path, rows, header=None, encoding="cp1251", name="export.csv"):
    lines = [";".join(f'"{c}"' for c in (header or HEADER))]
    lines += [";".join(f'"{c}"' for c in r) for r in rows]
    path = tmp_path / name
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode(encoding))
    return path


def test_columns_are_taken_by_name_not_by_position(tmp_path):
    """Регулятор добавил колонок — загрузчик обязан пережить это молча.

    Именно на ширине упал прежний скрипт: в выгрузке стало 43 поля вместо 39.
    """
    wide_header = HEADER + [f"Служебное поле {i}" for i in range(1, 28)]
    rows = [_row() + [""] * 27]
    result = read_export(_write(tmp_path, rows, header=wide_header))
    assert result.columns == len(wide_header)
    assert result.rows[0].product_name.endswith("«Селен-актив»")
    assert result.rows[0].registration_number == "RU.77.01.R.000001"


def test_a_reordered_header_still_maps(tmp_path):
    """Позиции переставлены — имена те же, разбор обязан сойтись."""
    order = [4, 0, 1, 3, 5, 7, 12, 15]
    header = [HEADER[i] for i in order]
    cells = _row()
    rows = [[cells[i] for i in order]]
    result = read_export(_write(tmp_path, rows, header=header))
    assert result.rows[0].manufacturer_name == "Диод"
    assert result.rows[0].scope_of_application == "источник селена"


@pytest.mark.parametrize("encoding", ["cp1251", "utf-8-sig"])
def test_both_encodings_are_read(tmp_path, encoding):
    """Регулятор отдаёт CP1251; пересохранённый руками файл приезжает в UTF-8."""
    result = read_export(_write(tmp_path, [_row()], encoding=encoding, name=f"{encoding}.csv"))
    assert result.encoding == encoding
    assert "Селен-актив" in result.rows[0].product_name


@pytest.mark.parametrize("issued,expected", [
    ("24.08.2026", date(2026, 8, 24)),
    ("2026-08-24", date(2026, 8, 24)),
    ("2026-08-24T00:00", date(2026, 8, 24)),
    ("", None),
    ("не дата", None),
])
def test_dates_come_in_two_shapes(tmp_path, issued, expected):
    """В одной выгрузке живут «24.08.2026» и «2026-08-24T00:00». Мусор — не повод
    ронять импорт: без даты запись годится, без наименования — нет."""
    result = read_export(_write(tmp_path, [_row(issued=issued)]))
    assert result.rows[0].registered_at == expected


def test_exact_duplicates_are_dropped(tmp_path):
    """Один продукт встречается в выгрузке несколько раз — перерегистрации с тем
    же содержимым. В таблице у них разошлись бы только суррогатные id."""
    result = read_export(_write(tmp_path, [_row(), _row(), _row(number="RU.77.01.R.000002")]))
    assert len(result.rows) == 2
    assert result.duplicates_dropped == 1


def test_a_row_without_a_product_name_is_skipped(tmp_path):
    """Искать по такой записи нечего и показать врачу нечего."""
    result = read_export(_write(tmp_path, [_row(), _row(name="")]))
    assert len(result.rows) == 1 and result.skipped_no_name == 1


def test_a_header_without_the_product_name_is_refused(tmp_path):
    """Отказ обязан назвать пропавшую колонку: по номеру позиции причину не видно."""
    header = [h for h in HEADER if h != "Наименование продукции"]
    rows = [[c for i, c in enumerate(_row()) if i != 4]]
    with pytest.raises(SgrFormatError) as e:
        read_export(_write(tmp_path, rows, header=header))
    assert "наименование продукции" in str(e.value).lower()


def test_an_export_without_rows_is_refused(tmp_path):
    """Пустая выгрузка — почти всегда сбой, а не отмена реестра. Дальше по
    цепочке стоит replace_all, который вычистил бы справочник."""
    with pytest.raises(SgrFormatError):
        read_export(_write(tmp_path, []))


def test_empty_cells_become_none_not_empty_strings(tmp_path):
    """Пустая строка в колонке и отсутствие значения — одно и то же состояние.
    Разные означали бы, что фильтр IS NULL врёт."""
    row = read_export(_write(tmp_path, [_row(scope="", label="")])).rows[0]
    assert row.scope_of_application is None and row.label_info is None


def test_dump_carries_exactly_the_keys_the_engine_reads(tmp_path):
    """Набор ключей — контракт с движком (references/supplements/sync/client.py).

    Лишний ключ движок молча проигнорирует, пропавший так же молча оставит поле
    пустым у врача в справке.
    """
    result = read_export(_write(tmp_path, [_row()]))
    path = tmp_path / "dump.jsonl.gz"
    assert write_dump(path, result.rows) == 1
    with gzip.open(path, "rt", encoding="utf-8") as f:
        record = json.loads(f.readline())
    assert set(record) == set(ENGINE_COLS)
    assert record["registered_at"] == "2026-08-24"
    assert record["product_name"].endswith("«Селен-актив»")
