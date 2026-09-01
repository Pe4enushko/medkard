"""Классификатор вида приёма против наименований 804н.

Та же сверка, что делает ``scripts/checks/check-nmu-classifier.py``, но по
выгрузке из ``resources/`` — чтобы она шла в обычном прогоне, а не только
у человека с корпусом НПА под рукой.

Скрипт импортируется по пути: имя файла с дефисами обычным import не берётся,
а дублировать нормативные регулярки в тесте хуже — они разойдутся.

Проверка сторожит регрессию классификатора, а не устаревание номенклатуры:
выгрузка заморожена на редакции с изм. 24.09.2020 (экспорт ГАРАНТ 06.09.2023).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "check_nmu_classifier", ROOT / "scripts" / "checks" / "check-nmu-classifier.py"
)
check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check)


def _report():
    return check.compare(check.entries_from_csv())


def test_nomenclature_export_is_whole():
    """Пустая или обрезанная выгрузка сделала бы все проверки ниже зелёными."""
    entries = check.entries_from_csv()
    assert len(entries) > 700, len(entries)


def test_code_verdict_never_contradicts_the_service_name():
    """Таблица и приказ не должны называть разные виды приёма.

    Окончание .001/.002 приказом не расшифровано — это наблюдённая
    закономерность с 28 нарушителями, и держится она на списках _NOT_A_PAIR.
    Новая специальность-нарушитель проявится здесь.
    """
    r = _report()
    assert r.contradictions == []


def test_code_verdict_never_appears_where_the_order_sees_no_appointment():
    """Вердикт там, где приказ приёма не видит, — тоже ошибка таблицы."""
    r = _report()
    assert r.extra == []


def test_exception_lists_still_catch_the_lying_names():
    """Списки запретов не должны опустеть незаметно.

    Если ряд перестанет срабатывать, наименование снова начнёт выносить
    вердикт за него — молча и правдоподобно.
    """
    r = _report()
    barred_codes = {line.split(" ", 1)[0] for line in r.barred}
    assert {
        "B04.070.002", "B04.070.003", "B04.070.004", "B04.070.005",
        "B03.070.001", "B03.070.002",
    } <= barred_codes, sorted(barred_codes)
