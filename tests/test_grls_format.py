from datetime import date

from grls import status as st
from grls.format import (NOT_FOUND, MedicineLookup, format_medicine_lookup, format_record,
                         status_line)
from grls.parser import build_record
from tests.grls_fixtures import sample_row
from storage.models.dietary_supplement import DietarySupplement

VISIT = date(2025, 3, 10)


def _rec(status, **over):
    return build_record(status, sample_row(**over))


def _lookup(**over):
    base = dict(query="амоксиклав", on=None, registry_date=date(2026, 8, 17),
                inn_records=[], inn_counts={}, trade_records=[], supplements=[])
    base.update(over)
    return MedicineLookup(**base)


def test_status_line_active_termless_and_termed():
    assert status_line(_rec(st.STATUS_ACTIVE), None) == "Действующий (РУ ЛП-000001, бессрочно)"
    assert status_line(_rec(st.STATUS_EAEU, expires_at="01.03.2027"), None) == \
        "Выдано по правилам ЕАЭС (РУ ЛП-000001, действует до 2027-03-01)"


def test_status_line_notes():
    assert status_line(_rec(st.STATUS_SUSPENDED), None) == \
        "Действующий, приостановлено применение (предупреждение, не запрет назначения) (РУ ЛП-000001)"
    assert status_line(_rec(st.STATUS_CONFIRMING, expires_at="17.10.2021"), None) == \
        "Действующий, на подтверждении регистрации (РУ ЛП-000001, срок до 2021-10-17)"
    assert status_line(_rec(st.STATUS_FOREIGN_PACK), None) == \
        "Действующий, в иностранной упаковке (РУ ЛП-000001)"


def test_status_line_dead_and_softened():
    r = _rec(st.STATUS_EXPIRED, expires_at="31.12.2025")
    assert status_line(r, None) == "Истёкший (истекло 2025-12-31; РУ ЛП-000001)"
    assert status_line(r, VISIT) == "Истёкший (истекло 2025-12-31; на дату визита 2025-03-10 действовало; РУ ЛП-000001)"
    a = _rec(st.STATUS_ANNULLED, annulled_at="14.02.2024")
    assert status_line(a, VISIT) == "Исключённый (аннулировано 2024-02-14; РУ ЛП-000001)"
    assert status_line(_rec(st.STATUS_ANNULLED), VISIT) == "Исключённый (дата неизвестна; РУ ЛП-000001)"


def test_format_record_uses_derived_forms_and_caps():
    r = _rec(st.STATUS_ACTIVE, forms_raw="; ".join(
        f"форма{i}, 5 мг - блистеры - По рецепту" for i in range(7)) + "; мазь, 1% - тубы - для стационаров;")
    text = format_record(r, None)
    assert "Торговое наименование: Тестин®" in text
    assert "МНН: тестамол" in text
    assert "Лекарственные формы: форма0; форма1; форма2; форма3; форма4 (+ ещё 3)" in text
    assert "Отпуск: По рецепту; для стационаров" in text
    assert "ЖНВЛП: да" in text
    assert "Формы выпуска:" not in text


def test_inn_branch_counts_and_examples():
    recs = [_rec(st.STATUS_ACTIVE, trade_name="Амоксиклав"), _rec(st.STATUS_EXPIRED, trade_name="Аугментин")]
    text = format_medicine_lookup(_lookup(query="амоксициллин+клавулановая кислота", inn_records=recs,
                                          inn_counts={st.STATUS_ACTIVE: 12, st.STATUS_EXPIRED: 3}))
    assert text.startswith("В ГРЛС «амоксициллин+клавулановая кислота» — это МНН.")
    assert "Регистраций: 15, из них действующих: 12" in text
    assert "Амоксиклав" in text and "Аугментин" in text
    assert "реестр от 2026-08-17" in text
    assert "внимание" not in text.lower()


def test_inn_branch_warns_when_nothing_live():
    text = format_medicine_lookup(_lookup(inn_records=[_rec(st.STATUS_EXPIRED)],
                                          inn_counts={st.STATUS_EXPIRED: 2}))
    assert "Внимание: все РУ по этому МНН истекли или аннулированы." in text


def test_trade_branch_header_and_blocks():
    text = format_medicine_lookup(_lookup(trade_records=[_rec(st.STATUS_ACTIVE), _rec(st.STATUS_EXPIRED, expires_at="31.12.2025")]))
    assert text.startswith("Найдено в ГРЛС (2; реестр от 2026-08-17):")
    assert "--- 1 ---" in text and "--- 2 ---" in text
    assert "Статус РУ: Действующий (РУ ЛП-000001, бессрочно)" in text


def test_supplement_and_not_found():
    s = DietarySupplement(product_name="Бак-Сет", registration_number="RU.77.99.11.003.Е.000001",
                          status="действует")
    text = format_medicine_lookup(_lookup(supplements=[s]))
    assert "Найдено как БАД" in text and "Бак-Сет" in text
    assert format_medicine_lookup(_lookup()) == NOT_FOUND
