"""
fixtures.py — building blocks for audit e2e fixture cards.

Every fixture card is a visit dict shaped exactly like 1C's payload
(docs/clinic-data-requirements.md §3) and carries **exactly one** defect.
Everything else in the card must be impeccable, because the audit tests
assert the complete set of formal findings, not just the presence of the
expected one — see e2e/tests/audit/README.md for why.

Practical consequence when writing a new fixture: the card has to satisfy
every rule that `FormalValidator.get_rules()` hands to the prompt for its
visit type and age group, not only the rule under test.  The rules that
apply to *all* visit types are the easy ones to trip by accident:

  ОТСУТСТВУЮТ_МЕТАДАННЫЕ_ВИЗИТА       — date, age, sex, service must be present
  ОБНАРУЖЕНЫ_ЗАГЛУШКИ                 — no "-", "уточнить", "б/д" stand-ins
  ДУБЛИРОВАНИЕ_СМЫСЛОВЫХ_БЛОКОВ       — do not repeat a block in two Параметр's
  НЕСООТВЕТСТВИЕ_УСЛУГИ_И_ВИЗИТА      — service, specialty and visit type must agree
  НЕСООТВЕТСТВИЕ_МКБ_И_ТЕКСТА_ДИАГНОЗА— diagnosis text must match its ICD code
  ОРФОГРАФИЧЕСКИЕ_ОШИБКИ              — proofread the Russian
  НЕПОЛНОЕ_НАЗНАЧЕНИЕ_ПРЕПАРАТА       — any drug needs name, dose, route,
                                        regimen, duration and justification
  ОТСУТСТВУЕТ_ИНФОРМАЦИЯ_О_СОПРОВОЖДАЮЩЕМ — children under 15 need a named
                                        legal representative
                                        (1C ships this field as «родственник
                                        лвн»; the audit renames it to «На
                                        приеме пациент с» on the way in —
                                        see parsers.inspection_labels)

Dates are fixed, never `today`: fixtures are pure data, nothing is written
to the database, and a stable date keeps runs reproducible and log lines
greppable.
"""

from __future__ import annotations

from typing import Any

VISIT_DATE = "2026-08-03"
DOCTOR_CODE = "00042"
DOCTOR_NAME = "Иванова Анна Сергеевна"


def dx(
    code: str,
    name: str,
    *,
    detail: str = "",
    first_time: bool = False,
) -> dict[str, Any]:
    """One entry for `Диагнозы`.

    `code` is what drives both DiagnosisValidator (guideline lookup) and,
    for Z11.1, the PROPHYLACTIC_TUBERCULIN branch of get_visit_types.
    """
    return {
        "КодМКБ": code,
        "НаименованиеМКБ": name,
        "Детализация": detail,
        "ВыявленВпервые": first_time,
    }


def base_visit(
    *,
    guid: str,
    service_code: str,
    service_name: str,
    specialty: str,
    age: int,
    inspection: list[tuple[str, str]],
    diagnoses: list[dict[str, Any]],
    gender: str = "Женский",
    visit_date: str = VISIT_DATE,
) -> dict[str, Any]:
    """Assemble a complete visit card.

    `inspection` is a list of (Параметр, Значение) pairs — kept as tuples at
    the call site so a fixture reads as a medical record rather than as JSON.
    `service_code` goes into `КодЕГИСЗ`; it is what get_visit_types classifies,
    so the visit type is *derived by the system*, never declared by the test.
    Pass an empty string to leave the service unclassified (used by the
    tuberculin fixtures, whose type comes from the Z11.1 diagnosis alone).

    `Пациент.AGE` is written as an int under the key the validator actually
    reads — `validator.py` looks at `AGE` only, with no fallback to «Возраст».
    """
    return {
        "Прием": {
            "GUID": guid,
            "NUM": guid.rsplit("-", 1)[-1],
            "DATE": visit_date,
            "Врач_код": DOCTOR_CODE,
            "Врач": DOCTOR_NAME,
        },
        "Врач": {"SPECIALIZATION": specialty},
        "Пациент": {"CODE": "P-000001", "GENDER": gender, "AGE": age},
        "Услуги": [{"КодЕГИСЗ": service_code, "Наименование": service_name}],
        "ДанныеОсмотра": [
            {"Параметр": param, "Значение": value} for param, value in inspection
        ],
        "Диагнозы": diagnoses,
    }
