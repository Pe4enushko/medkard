"""Synthetic contrast cases shared by atomic and full-audit regressions."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    name: str
    rule: str
    fields: list[tuple[str, str]]
    violated: bool
    diagnosis: tuple[str, str] = ("Z00.1", "Рутинное обследование состояния здоровья ребёнка")
    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()


CASES = [
    Case("units_without_numbers", "placeholder_values_are_defect", [("ЧСС", "в мин."), ("ЧД", "в мин.")], True),
    Case("numbers_with_units", "placeholder_values_are_defect", [("ЧСС", "120 в мин."), ("Рост", "75 см")], False),
    Case("recipient_instruction", "placeholder_values_are_defect", [("Кому выдана", "ФИО, дата рождения")], True),
    Case("unfinished_medical_exemption", "placeholder_values_are_defect", [("Прививочный анамнез", "не привит в связи с медотводом диагноз")], True),
    Case("meaningful_short_values", "placeholder_values_are_defect", [("Эпидемиологический анамнез", "Не организован(а)."), ("Ф20", "отриц."), ("План обследования", "КАК"), ("Аллергологический анамнез", "пищевая непереносимость?")], False),
    Case("statuses_and_specialty", "placeholder_values_are_defect", [("Квалификация хирурга", "Врач - онколог"), ("Консультация перв/повтор", "Первичная"), ("В выдаче листке нетрудоспособности", "не нуждается")], False),
    Case("epicrisis_is_content", "placeholder_values_are_defect", [("Жалобы", "Нет. Эпикриз 1 год: за прошедшие три месяца ОРВИ не болел, развитие соответствует возрасту.")], False),
    Case("template_label_not_value", "placeholder_values_are_defect", [("Первичный эпикриз на ребенка в ХХ дней", "Период новорожденности без осложнений.")], False),
    Case("pending_test_result", "placeholder_values_are_defect", [("Комментарий к вакцинации", "24.09.2026 поставлена проба Манту. Доза 2 ТЕ. Результат:"), ("Рекомендации", "Оценить результат пробы 27.09.2026")], False),
    Case("typo_is_not_placeholder", "placeholder_values_are_defect", [("Анамнез заболевания", "Высыпания воозвращаются после отмены крема.")], False),
    Case("template_label_typo", "has_typos", [("Рекомендованна следующая плановая консультация", "в 1 год")], False),
    Case("real_value_typo", "has_typos", [("Анамнез заболевания", "Высыпания воозвращаются после отмены крема."), ("Рекомендованна следующая плановая консультация", "в 1 год")], True, required=("воозвращаются", "возвращаются"), forbidden=("рекомендованна",)),
    Case("no_invented_typos", "has_typos", [("Анамнез заболевания", "Заболела после переохлаждения. Наследственность не отягощена."), ("Рекомендации", "Охранительный режим. КАК, ОАМ, ЭКГ.")], False),
    Case("result_in_anamnesis", "plan_vs_result_separation", [("Анамнез заболевания", "ОАК от 24.09.2026: Hb 120 г/л, лейкоциты 6 × 10⁹/л."), ("План обследования", "Повторить ОАК через месяц")], False),
    Case("notes_without_plan_field", "plan_vs_result_separation", [("Заметки", "ОАК от 24.09.2026: Hb 120 г/л. Рекомендовано повторить ОАК через месяц.")], False),
    Case("result_as_plan", "plan_vs_result_separation", [("План обследования", "ОАК от 24.09.2026: Hb 120 г/л, лейкоциты 6 × 10⁹/л.")], True),
    Case("past_result_explains_plan", "plan_vs_result_separation", [("План обследования", "Из-за сниженного Hb в прошлом анализе повторить ОАК через месяц")], False),
    Case("past_period_not_today", "diagnosis_should_be_supported", [("Эпикриз", "За прошедшие три месяца ОРВИ не болел."), ("Жалобы", "Сегодня появился насморк и першение в горле."), ("Объективные данные", "Зев гиперемирован, слизистое отделяемое из носа.")], False, diagnosis=("J06.9", "Острая инфекция верхних дыхательных путей неуточнённая")),
    Case("unsupported_current_diagnosis", "diagnosis_should_be_supported", [("Эпикриз", "За прошедшие три месяца ОРВИ не болел."), ("Жалобы", "Нет"), ("Объективные данные", "Состояние удовлетворительное, температура нормальная, зев чистый, дыхание везикулярное, хрипов нет.")], True, diagnosis=("J18.9", "Пневмония неуточнённая")),
    Case("d3_goal_in_other_field", "treatment_dosage_clarity", [("План лечения", "Профилактика дефицита витамина Д"), ("Рекомендации", "Колекальциферол 1000 МЕ внутрь ежедневно, 3 месяца")], False),
    Case("d3_goal_absent", "treatment_dosage_clarity", [("Рекомендации", "Колекальциферол 1000 МЕ внутрь ежедневно, 3 месяца")], True, required=("обоснован",)),
    Case("d3_goal_absent_healthy_context", "treatment_dosage_clarity", [("Жалобы", "Нет."), ("Анамнез заболевания", "Ребёнок осматривается в плановом порядке, развитие соответствует возрасту."), ("Рекомендации", "Колекальциферол 1000 МЕ внутрь ежедневно, 3 месяца")], True, required=("обоснован",)),
    Case("d3_goal_present_healthy_context", "treatment_dosage_clarity", [("Жалобы", "Нет."), ("Анамнез заболевания", "Ребёнок осматривается в плановом порядке, развитие соответствует возрасту."), ("План лечения", "Профилактика дефицита витамина Д"), ("Рекомендации", "Колекальциферол 1000 МЕ внутрь ежедневно, 3 месяца")], False),
    Case("d3_goal_does_not_replace_duration", "treatment_dosage_clarity", [("План лечения", "Профилактика дефицита витамина Д"), ("Рекомендации", "Колекальциферол 1000 МЕ внутрь ежедневно")], True, required=("продолжитель",), forbidden=("обоснован",)),
    Case("trade_name_without_inn", "prescription_by_trade_name", [("Рекомендации", "Нурофен 200 мг внутрь до 3 раз в сутки, 3 дня при температуре выше 38 °C, для снижения температуры")], True, required=("нурофен", "ибупрофен"), forbidden=("комисси",)),
    Case("inn_no_commission_needed", "prescription_by_trade_name", [("Рекомендации", "Ибупрофен 200 мг внутрь до 3 раз в сутки, 3 дня для снижения температуры")], False),
    Case("trade_with_inn", "prescription_by_trade_name", [("Рекомендации", "Ибупрофен (Нурофен) 200 мг внутрь до 3 раз в сутки, 3 дня для снижения температуры")], False),
    Case("five_inn_count_is_irrelevant", "prescription_by_trade_name", [("Рекомендации", "Амоксициллин, азитромицин, ацетилцистеин, ибупрофен, сальбутамол. Решение ВК не указано.")], False),
    Case("four_trade_names_issue_is_inn", "prescription_by_trade_name", [("Рекомендации", "Нурофен, Панадол, Нольпаза, Дона. Решение ВК не указано.")], True, forbidden=("комисси", "четыре",)),
    Case("trade_with_commission_exception", "prescription_by_trade_name", [("Рекомендации", "Нурофен 200 мг внутрь до 3 раз в сутки, 3 дня для снижения температуры"), ("Решение ВК", "Назначение Нурофена по торговому наименованию согласовано врачебной комиссией, протокол №1 от 24.09.2026")], False),
]
