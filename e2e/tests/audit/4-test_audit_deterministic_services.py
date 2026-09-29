#!/usr/bin/env python3
"""
Группа 4 — детерминированный каталог, правила строки услуги.

Четыре правила, все без модели:
  • УСЛУГА_БЕЗ_КОДА_НОМЕНКЛАТУРЫ        — кода 804н нет ни в одном поле строки;
  • ВИД_ПРИЁМА_НЕ_СООТВЕТСТВУЕТ_КОДУ_УСЛУГИ — врач записал «повторная»,
    а код услуги первичный;
  • NMU_CODE_CONTRADICTION              — наименование услуги называет не тот
    вид приёма, что её код;
  • НАИМЕНОВАНИЕ_УСЛУГИ_НЕ_ПО_НОМЕНКЛАТУРЕ — НаименованиеЕГИСЗ расходится
    с наименованием этого кода в 804н.

NMU_CODE_CONTRADICTION проверяется здесь ещё и как проводка: этот флаг раньше
ставил формальный валидатор, а теперь он правило каталога
(`service_name_matches_code`). Флаг обязан прийти ровно один раз и со слепком —
дубль означал бы, что старое срабатывание никуда не делось.

Карты собраны взрослыми первичными приёмами с полным минимумом 025/у: иначе к
целевому флагу добавился бы НЕ_ЗАПОЛНЕН_НОРМАТИВНЫЙ_МИНИМУМ_ЗАПИСИ, правило
которого проверяется соседним скриптом.

Кейсы стоят `only=False` и идут парами с отрицательными: правила услуги
пересекаются с формальным НЕСООТВЕТСТВИЕ_УСЛУГИ_И_ВИЗИТА, которое читает те же
поля и на противоречии внутри строки может высказаться законно.

Запуск (нужны БД и LLM):
    python e2e/tests/audit/4-test_audit_deterministic_services.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import base_visit, dx  # noqa: E402
from harness import Case, VisitType, run_cases  # noqa: E402

PRIMARY = {VisitType.PRIMARY}
PRIMARY_CODE = "B01.047.001"
# Дословно из 804н под этим кодом — с ним сверяется НаименованиеЕГИСЗ.
PRIMARY_NAME = "Прием (осмотр, консультация) врача-терапевта первичный"
REPEAT_NAME = "Прием (осмотр, консультация) врача-терапевта повторный"
SPECIALTY = "Терапевт"

NO_CODE = "УСЛУГА_БЕЗ_КОДА_НОМЕНКЛАТУРЫ"
TYPE_VS_CODE = "ВИД_ПРИЁМА_НЕ_СООТВЕТСТВУЕТ_КОДУ_УСЛУГИ"
NAME_VS_CODE = "NMU_CODE_CONTRADICTION"
EGISZ_NAME = "НАИМЕНОВАНИЕ_УСЛУГИ_НЕ_ПО_НОМЕНКЛАТУРЕ"

_ARI = dx("J06.9", "Острая инфекция верхних дыхательных путей неуточнённая", first_time=True)

# Минимум формы 025/у для первичного приёма плюс безупречное всё остальное.
_RECORD = [
    ("Жалобы", "Боль в горле и насморк в течение двух дней, повышение температуры до 37.8 °C."),
    (
        "Анамнез",
        "Заболел остро 01.08.2026 после переохлаждения. До обращения лекарственных препаратов "
        "не принимал. Аллергологический анамнез не отягощён. Хронических заболеваний нет.",
    ),
    (
        "Объективный осмотр",
        "Состояние удовлетворительное. Температура 37.6 °C. Зев гиперемирован, миндалины без "
        "налётов. В лёгких дыхание везикулярное, хрипов нет, ЧДД 17 в минуту. Тоны сердца ясные, "
        "ритмичные, ЧСС 78 в минуту, АД 120/78 мм рт. ст.",
    ),
    (
        "Обоснование диагноза",
        "Острое начало, катаральный синдром и субфебрильная лихорадка при отсутствии физикальных "
        "признаков поражения нижних дыхательных путей соответствуют острой инфекции верхних "
        "дыхательных путей.",
    ),
    (
        "Лечение",
        "Парацетамол 500 мг внутрь, однократно при температуре выше 38.0 °C, до 3 дней — "
        "обоснование назначения: жаропонижающая терапия. Обильное тёплое питьё, домашний режим.",
    ),
    ("Рекомендации", "Повторная явка 06.08.2026 либо ранее при ухудшении состояния."),
]


def _visit(guid: str, **over: object) -> dict:
    """Взрослый первичный приём терапевта; отличия кейса передаются поверх."""
    params: dict = dict(
        guid=guid,
        service_code=PRIMARY_CODE,
        service_name=PRIMARY_NAME,
        specialty=SPECIALTY,
        age=45,
        gender="Мужской",
        diagnoses=[_ARI],
        inspection=list(_RECORD),
    )
    params.update(over)
    return base_visit(**params)  # type: ignore[arg-type]


# ── Кода номенклатуры у услуги нет ни в одном поле ───────────────────────────
# Вид приёма тогда выводится из наименования — ровно тот запасной разбор,
# которым система живёт на картах МДС, где КодЕГИСЗ пуст во всех строках.
service_without_code = _visit("e2e-audit-det-service-no-code", service_code="")
service_with_code = _visit("e2e-audit-det-service-with-code")


# ── Врач записал «повторная», код услуги — первичный ──────────────────────────
declared_repeat = _visit(
    "e2e-audit-det-type-vs-code",
    inspection=[*_RECORD, ("Консультация перв/повтор", "повторная")],
)
declared_primary = _visit(
    "e2e-audit-det-type-matches-code",
    inspection=[*_RECORD, ("Консультация перв/повтор", "первичная")],
)


# ── Наименование услуги называет повторный приём, код — первичный ─────────────
name_says_repeat = _visit("e2e-audit-det-name-vs-code", service_name=REPEAT_NAME)


# ── НаименованиеЕГИСЗ расходится с наименованием кода в 804н ──────────────────
# Собственное «Наименование» оставлено верным: свободное поле клиники сверять
# не с чем, правило смотрит только пару КодЕГИСЗ + НаименованиеЕГИСЗ.
egisz_name_wrong = _visit(
    "e2e-audit-det-egisz-name-wrong",
    egisz_name="Прием (осмотр, консультация) врача-терапевта",
)
egisz_name_right = _visit(
    "e2e-audit-det-egisz-name-right",
    egisz_name=PRIMARY_NAME,
)


CASES = [
    Case(
        name="услуга без кода номенклатуры",
        visit=service_without_code,
        expect=NO_CODE,
        visit_types=PRIMARY,
        only=False,
    ),
    Case(
        name="код услуги на месте — правило молчит",
        visit=service_with_code,
        expect=NO_CODE,
        visit_types=PRIMARY,
        present=False,
        only=False,
    ),
    Case(
        name="в записи «повторная», код услуги первичный",
        visit=declared_repeat,
        expect=TYPE_VS_CODE,
        visit_types=PRIMARY,
        only=False,
    ),
    Case(
        name="запись и код согласны — правило молчит",
        visit=declared_primary,
        expect=TYPE_VS_CODE,
        visit_types=PRIMARY,
        present=False,
        only=False,
    ),
    Case(
        name="наименование услуги повторное, код первичный",
        visit=name_says_repeat,
        expect=NAME_VS_CODE,
        visit_types=PRIMARY,
        only=False,
    ),
    Case(
        name="наименование и код согласны — правило молчит",
        visit=service_with_code,
        expect=NAME_VS_CODE,
        visit_types=PRIMARY,
        present=False,
        only=False,
    ),
    Case(
        name="НаименованиеЕГИСЗ не по номенклатуре",
        visit=egisz_name_wrong,
        expect=EGISZ_NAME,
        visit_types=PRIMARY,
        only=False,
    ),
    Case(
        name="НаименованиеЕГИСЗ как в 804н — правило молчит",
        visit=egisz_name_right,
        expect=EGISZ_NAME,
        visit_types=PRIMARY,
        present=False,
        only=False,
    ),
]


if __name__ == "__main__":
    sys.exit(asyncio.run(run_cases("Детерминированный каталог — правила строки услуги", CASES)))
