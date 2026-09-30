#!/usr/bin/env python3
"""Full-pipeline regressions for the chief physician's false-positive reports.

Uses the deployed checkout's rules, normalisation, formal/ICD/diagnosis audit.
Requires that checkout's .env, PostgreSQL and LLM. No audit rows are persisted.
Unrelated findings are allowed: each scenario checks its target flag and wording.
The paired atomic suite reuses exactly the same contrast examples.
"""
from __future__ import annotations

import asyncio

from fixtures import base_visit, dx
from formal_false_positive_cases import CASES as ATOMIC_CASES, Case as Example
from harness import Case, VisitType, run_cases


FLAGS = {
    "placeholder_values_are_defect": "ОБНАРУЖЕНЫ_ЗАГЛУШКИ",
    "has_typos": "ОРФОГРАФИЧЕСКИЕ_ОШИБКИ",
    "plan_vs_result_separation": "СМЕШАНЫ_ПЛАН_И_РЕЗУЛЬТАТЫ",
    "diagnosis_should_be_supported": "ДИАГНОЗ_НЕ_ПОДТВЕРЖДЁН_ЗАПИСЬЮ",
    "treatment_dosage_clarity": "НЕПОЛНОЕ_НАЗНАЧЕНИЕ_ПРЕПАРАТА",
    "prescription_by_trade_name": "НАЗНАЧЕНИЕ_ПО_ТОРГОВОМУ_БЕЗ_МНН",
}

BASE_FIELDS = [
    ("Жалобы", "Нет."),
    ("Анамнез заболевания", "Ребёнок осматривается в плановом порядке, развитие соответствует возрасту."),
    ("Объективные данные", "Состояние удовлетворительное. Температура 36,6 °C. Зев чистый. В лёгких дыхание везикулярное, хрипов нет. ЧСС 120 в мин., ЧД 30 в мин."),
    ("Эпидемиологический анамнез", "Контакты с инфекционными больными отрицает. Не организован."),
    ("Аллергологический анамнез", "Не отягощён."),
    ("Рекомендации", "Питание по возрасту, прогулки, повторный осмотр через месяц."),
    ("На приеме пациент с", "Матерью."),
]


def card(example: Example) -> dict:
    replaced = {name for name, _ in example.fields}
    fields = [pair for pair in BASE_FIELDS if pair[0] not in replaced] + example.fields
    return base_visit(
        guid=f"e2e-full-false-positive-{example.name}",
        service_code="B01.031.001",
        service_name="Приём (осмотр, консультация) врача-педиатра первичный",
        specialty="Педиатр", age=1, visit_date="24.09.2026",
        diagnoses=[dx(*example.diagnosis)], inspection=fields,
    )


def full_case(example: Example) -> Case:
    return Case(
        name=example.name, visit=card(example), expect=FLAGS[example.rule],
        visit_types={VisitType.PRIMARY}, present=example.violated, only=False,
        issue_contains=example.required, issue_excludes=example.forbidden,
    )


CASES = [full_case(example) for example in ATOMIC_CASES]

# The earlier e16939f fixes must also survive the full pipeline.
CASES.extend([
    full_case(Example(
        "study_abbreviations_not_drugs", "treatment_dosage_clarity",
        [("Рекомендации", "Контроль ЭКГ, КАК, ОАМ")], False,
    )),
    full_case(Example(
        "study_and_real_incomplete_drug", "treatment_dosage_clarity",
        [("Рекомендации", "КАК, ОАМ. Парацетамол при температуре выше 38 °C, 3 дня")],
        True, required=("парацетамол",), forbidden=("'как'", "'оам'"),
    )),
])

chief = card(Example("chief_title_not_specialty_conflict", "placeholder_values_are_defect", [], False))
chief["Услуги"] = [{
    "КодЕГИСЗ": "B04.031.002",
    "Наименование": "Профилактический прием (осмотр, консультация) главного врача-педиатра",
}]
CASES.append(Case(
    name="chief_title_not_specialty_conflict", visit=chief,
    expect="НЕСООТВЕТСТВИЕ_УСЛУГИ_И_ВИЗИТА",
    visit_types={VisitType.PROPHYLACTIC}, present=False, only=False,
))

duplicate = card(Example(
    "plan_and_recommendations_are_allowed", "placeholder_values_are_defect",
    [("План лечения", "Режим, диета, симптоматическая терапия"),
     ("Рекомендации", "Режим домашний. Питание по возрасту. Промывание носа солевым раствором при насморке.")],
    False,
))
CASES.append(Case(
    name="plan_and_recommendations_are_allowed", visit=duplicate,
    expect="ДУБЛИРОВАНИЕ_СМЫСЛОВЫХ_БЛОКОВ",
    visit_types={VisitType.PRIMARY}, present=False, only=False,
))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_cases("Формальные правила — ложные замечания главврача", CASES)))
