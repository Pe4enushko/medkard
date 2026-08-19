# tests/test_grls_normalize.py
from datetime import date, datetime

from grls import normalize as n


def test_clean_cell_null_markers_and_xlsx_cr():
    assert n.clean_cell(None) is None
    assert n.clean_cell("") is None
    assert n.clean_cell("  ~ ") is None
    assert n.clean_cell("  Ампициллин ") == "Ампициллин"
    assert n.clean_cell("ЛП-000001-280722_x000D_\nИзм. №1") == "ЛП-000001-280722\nИзм. №1"


def test_parse_date_variants():
    assert n.parse_date("05.06.2000") == date(2000, 6, 5)
    assert n.parse_date("17.08.2026 05:00:00") == date(2026, 8, 17)
    assert n.parse_date(datetime(2024, 2, 14, 10, 0)) == date(2024, 2, 14)
    assert n.parse_date(date(2024, 2, 14)) == date(2024, 2, 14)
    assert n.parse_date("") is None
    assert n.parse_date("~") is None
    assert n.parse_date("2024-02-14") is None      # unexpected format → None + warning
    assert n.parse_date("31.02.2024") is None      # invalid calendar date


FORMS_RAW = ("таблетки, покрытые пленочной оболочкой, 5 мг, 10 шт. - блистеры (2 шт.)  - пачки картонные (20 шт.)  - Без рецепта; "
             "таблетки, покрытые пленочной оболочкой, 5 мг, 7 шт. - блистеры (4 шт.)  - пачки картонные (28 шт.)  - Без рецепта; "
             "мазь для местного и наружного применения, 0.2%, 5 кг - ведра - для стационаров; "
             " - Без рецепта; "
             "капсулы")


def test_split_forms_trims_and_drops_empty():
    forms = n.split_forms(FORMS_RAW)
    assert len(forms) == 5
    assert forms[3] == "- Без рецепта"
    assert n.split_forms(None) == []
    assert n.split_forms("") == []


def test_derive_dosage_forms_unique_in_order_skips_fragments():
    forms = n.split_forms(FORMS_RAW)
    assert n.derive_dosage_forms(forms) == [
        "таблетки", "мазь для местного и наружного применения", "капсулы"]


def test_derive_dispensing_unique_skips_elements_without_separator():
    forms = n.split_forms(FORMS_RAW)
    assert n.derive_dispensing(forms) == ["Без рецепта", "для стационаров"]


def test_is_substance_by_number_or_form():
    assert n.is_substance("ФС-000001", ["субстанция-порошок"]) is True
    assert n.is_substance("ЛП-000001", ["Субстанция-жидкость"]) is True
    assert n.is_substance("ФС-000002", []) is True
    assert n.is_substance("ЛП-000001", ["таблетки"]) is False


def test_parse_yes_no_and_narcotic():
    assert n.parse_yes_no("Да") is True
    assert n.parse_yes_no("нет") is False
    assert n.parse_yes_no("") is None
    assert n.parse_yes_no("~") is None
    assert n.parse_narcotic("~") is None
    assert n.parse_narcotic("Нет") is None
    assert n.parse_narcotic("ПIII") == "ПIII"


def _hash_kwargs(**over):
    base = dict(status="Действующий", reg_number="ЛП-000001", registered_at=date(2020, 1, 1),
                expires_at=None, annulled_at=None, holder="ООО Тест", holder_country="Россия",
                trade_name="Тестин", inn_name="тестамол", forms_raw="таблетки, 5 мг - По рецепту;",
                production_stages=None, normative_docs=None, pharm_group=None,
                is_vital=True, narcotic_list=None, is_orphan=None)
    base.update(over)
    return base


def test_row_hash_deterministic_and_sensitive():
    h1 = n.row_hash(**_hash_kwargs())
    assert h1 == n.row_hash(**_hash_kwargs())
    assert len(h1) == 64
    assert h1 != n.row_hash(**_hash_kwargs(status="Истёкший"))
    assert h1 != n.row_hash(**_hash_kwargs(is_vital=False))
    assert h1 != n.row_hash(**_hash_kwargs(is_vital=None))
    assert h1 != n.row_hash(**_hash_kwargs(registered_at=date(2020, 1, 2)))


def test_row_hash_is_stable_across_versions():
    # Pin the algorithm: engine recomputes this from the dump (spec §4.3/§7).
    expected = n.row_hash(**_hash_kwargs())
    import hashlib
    parts = ["Действующий", "ЛП-000001", "2020-01-01", "", "", "ООО Тест", "Россия",
             "Тестин", "тестамол", "таблетки, 5 мг - По рецепту;", "", "", "", "1", "", ""]
    assert expected == hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def test_normalize_query():
    assert n.normalize_query('  "ЭФКУРИЯ®"  ') == "эфкурия"
    assert n.normalize_query("«Кей Джи Пи»") == "кей джи пи"
    assert n.normalize_query("Ёлкин\tчай") == "елкин чай"
    assert n.normalize_query("Аспирин™ 500") == "аспирин 500"
    assert n.normalize_query("~") == ""
