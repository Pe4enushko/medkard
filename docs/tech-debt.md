# Технический долг

Список ведётся с ревью ветки `dev` перед релизом (2026-08-22, отчёт —
`~/projects/report-2026-08-22-medkard-dev-release-review.html`). Сюда попадает
то, что подтверждено чтением кода, но сознательно не чинится сейчас.

Правила ведения: одна запись — одна проверяемая вещь, с файлом и строкой;
запись убирается только вместе с кодом, к которому относится.

## 1. Мёртвый код

Опасен не сам по себе, а тем, что читается как работающий контур: следующий
разработчик будет чинить то, что не исполняется.

| Что | Где | Состояние |
|---|---|---|
| `_enrich_flags` + `_levenshtein` — подтягивание `source` и fuzzy-починка флага модели с расстоянием ≤ 3 | `src/audit/formal_structure/validator.py:62–88` | Не вызывается: после перехода на атомарные правила флаг проставляет Python (`validations.validate_rule`), а `source` — `validate()` напрямую из правила |
| Сторожа `_DROPPED` / `_FUZZY` в e2e-харнессе | `e2e/tests/audit/harness.py:117–119, 139–143` | Ловят лог-сообщения `_enrich_flags`, то есть источник недостижим. Списки `report.dropped` и `report.fuzzy` не наполнятся никогда — **снимать вместе с `_enrich_flags`**, иначе останется вид работающей страховки |
| Адаптеры `ChineseDetector`: `check_findings`, `check_hypothetical_queries`, `check_visit_label`, `check_file_id`, `check_raw_content` | `src/LLM/chinese_detector.py:54–98` | Ни одного вызова. Используется только `check_str`. `check_hypothetical_queries` адаптирует поля контура reverse-HyDE, которого в ветке уже нет |
| `VisitClassifier` (LLM-классификация типа визита) и таблица `_LLM_LABEL_TO_TYPE` | `src/LLM/visit_classifier.py`, импорт в `validator.py:27`, таблица в `validator.py:139` | Классификация давно детерминированная; модуль достижим только через неиспользуемый импорт |
| Tool `retrieve`, `hybrid_search`, докстринг про `create_rag_agent` | `src/LLM/rag_agent.py:12–14, 47–75`, `src/RAG/retrieval/vector_store.py:382–458` | Функции `create_rag_agent` не существует (есть `create_checker_agent`); tool не входит ни в один набор, `hybrid_search` жив только за счёт `tests/test_vector_store*.py` |
| `validate_visit` (проверка всех правил одним запросом) | `src/LLM/validations.py:96–128` | Заменён атомарным `validate_rule`; вызывается только из `tests/test_validations.py` |
| `total_tokens += 0` | `src/LLM/client.py:181` | Ничего не делает |
| `LLM_OBSERVABILITY_PATH` | `.env.example:9` | Модуль `src/LLM/observability.py` и `scripts/summarize-llm-observability.py` удалены в `4166fe6` |

Решение 2026-08-22: до релиза не трогаем — вычистка задевает тесты
(`test_chinese_detector.py`, `test_validations.py`, `test_vector_store*.py`) и
съест время, которого перед выкаткой нет. Убирать одним заходом после релиза,
вместе с их тестами.

## 2. Объявлено в данных, не реализовано в коде

| Что | Где | Следствие |
|---|---|---|
| `applies_to.specialties` объявлено у всех 42 правил (4 — `["pediatrics"]`, 2 — пустой список), но `get_rules` это поле не читает | `src/audit/formal_structure/rules.json`, `validator.py:get_rules` | Педиатрические правила (`ОРФОГРАФИЧЕСКИЕ_ОШИБКИ`, `НЕСООТВЕТСТВИЕ_МКБ_И_ТЕКСТА_ДИАГНОЗА`, `СЛИШКОМ_ОБЩИЙ_КОД_МКБ`, `ДОПУСТИМО_БЕЗ_ЛЕЧЕНИЯ_ПРОФИЛАКТИКА`) применяются к любому врачу. `Врач.SPECIALIZATION` из карты используется только для колонки в Excel |
| `CRITERIA_MAX_CHUNKS` (спека §4.5a, дефолт 8) | `src/LLM/graphs/diagnosis_nodes.py:508` (`limit=None`), `src/RAG/retrieval/searches.py:192–223` (SQL без `LIMIT`) | После реингеста criteria-пул одного КР доходит до 83 чанков (собственный эвал) и целиком уезжает судье |
| Два правила делят `flag_code` `ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО`, а `get_rules` дедуплицирует по флагу | `rules.json`, `validator.py:get_rules` | Пара разведена по `age_group`, поэтому сейчас безопасна; любая новая пара с общим флагом молча потеряет второе правило |

## 3. Поведение, которое стоит померить и починить после релиза

Подробности и способ проверки — в отчёте ревью, здесь только якоря.

| Что | Где |
|---|---|
| `last_exc` не сбрасывается: удачная, но обрезанная попытка после `APIError` поднимает исключение первой попытки вместо возврата контента | `src/LLM/client.py:120, 212–221` |
| `_is_context_overflow` разбирает текст чужого исключения по подстроке (парный `_is_length_limit_error` снят вместе с тихим провалом ICD) | `src/LLM/client.py:32–42` |
| ReAct-обвязка осиротела: после перевода ICD-чекера на двухэтапный конвейер у `LLMClient.call_agent`, `LLM/rag_agent.py` и `LLM/tools.py` не осталось вызывающих. `tests/test_llm_client_agent.py` продолжает закреплять поведение мёртвого кода. Снимать одним проходом вместе с остальным мёртвым кодом | `src/LLM/client.py:243–495`, `src/LLM/rag_agent.py`, `src/LLM/tools.py` |
| Замечание с невалидными `chunk_refs` всё равно попадает в результат — только `logger.warning` | `src/LLM/graphs/diagnosis_nodes.py:896–925` |
| Смешение шкал `rerank_score` и `rrf_score` в одном пуле при частичном отказе реранкера | `src/LLM/graphs/diagnosis_nodes.py:823–825` |
| BM25 считается только среди кандидатов, отобранных вектором | `src/RAG/retrieval/searches.py:87–101` |
| `sum(pg_column_size(done_cards.*))` детоастит JSONB: дорого, и это распакованный размер, а не занятый на диске | `src/storage/stats_storage.py:42–58` |
| `GRAPH_TRACE_PATH` по умолчанию относительный — трейс уезжает в CWD процесса | `src/audit/graph_trace.py:81` |
| `Result` не содержит `icd_check`, поэтому `audit.completed.output` в трейсе показывает только formal и diagnosis | `src/audit/pipeline.py:556–561` |
