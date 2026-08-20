# Diagnosis graph e2e on `eval_broken_cards`

Эти standalone-тесты прогоняют реальные обезличенные карты через
`DiagnosisValidator`: подбор КР, PostgreSQL/RAG, embeddings, опциональный
reranker, реальные LLM-вызовы и скомпилированный diagnosis-граф. Моков нет.

ICD-ReAct намеренно не запускается: он не менялся в ветке diagnosis-графа и его
историческая рекурсия не должна маскировать результат проверяемого компонента.

Выбранные карты и основание выбора лежат в
`e2e/fixtures/eval_broken_cards/cases.json`. Покрыты разные исторические
причины: recursion, context overflow, output length, invalid JSON, отсутствие
`parsed/refusal` и нарушение схемы `issues[0]`.

Запуск из корня репозитория тратит реальные LLM-токены и требует настроенных
PostgreSQL, embeddings и LLM:

Общий runner находит тесты по имени, печатает полный вывод каждого и запускает
следующий даже после ошибки. В конце он возвращает ненулевой exit code, если
упал хотя бы один тест:

```bash
e2e/run-diagnosis-graph-tests.sh
```

Посмотреть найденные тесты без LLM-вызовов:

```bash
e2e/run-diagnosis-graph-tests.sh --list
```

Другой Python-интерпретатор задаётся через
`E2E_PYTHON_BIN=/path/to/python`.

Таймаут на одну карту по умолчанию — 900 секунд. Его можно изменить через
`E2E_DIAG_GRAPH_TIMEOUT_SECONDS`.
