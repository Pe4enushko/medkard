# Graph audit traces

Полный структурированный трейс аудита пишется построчно в
`logs/graphtraces.jsonl`. Путь можно переопределить переменной
`GRAPH_TRACE_PATH`; пустое значение отключает запись.

Каждая строка — самостоятельный JSON-объект. Общие поля:

- `timestamp` — время события в UTC, ISO 8601;
- `event` — тип события;
- `correlation_id` — UUID одного запуска аудита одной карты;
- `card_guid` — GUID карты, если он доступен.

События вне контекста аудита карты в этот файл не записываются, поэтому у
каждой строки есть непустой `correlation_id`.

Событие `audit.started` содержит `card_data_priem` — исходный объект
`card_data -> 'Прием'`. Один `correlation_id` проходит через формальный,
ICD- и diagnosis-контуры до `audit.completed` либо `audit.failed`.

Основные семейства событий:

- `audit.*` — начало, завершение, падение и сохранение карты;
- `checker.*` — запуск и итог formal, ICD и каждого диагноза;
- `diagnosis.*` — выбор КР, начало и итог графа;
- `graph.node.*` — вход, выход и деградация каждого узла diagnosis-графа;
- `retrieval.*` — запросы и полные списки найденных чанков;
- `medicine.*` — извлечённые названия и результат поиска каждого препарата;
- `llm.call.*` — structured-вызовы, попытки, ответы и ошибки;
- `llm.agent.*` — ReAct-вызовы, сообщения, действия инструментов и итог.

## Как читать

Все события конкретной карты, в исходном порядке:

```bash
jq -c 'select(.correlation_id == "UUID")' logs/graphtraces.jsonl
```

Найти UUID по GUID карты:

```bash
jq -r 'select(.card_guid == "GUID") | .correlation_id' \
  logs/graphtraces.jsonl | sort -u
```

Посмотреть только retrieval и лекарства:

```bash
jq -c 'select(.event | startswith("retrieval.") or startswith("medicine."))' \
  logs/graphtraces.jsonl
```

Выводы всех чекеров по одному аудиту:

```bash
jq -c 'select(.correlation_id == "UUID" and .event == "checker.completed")' \
  logs/graphtraces.jsonl
```

Проверить незавершённые аудиты:

```bash
jq -s '
  group_by(.correlation_id)[]
  | select(.[0].correlation_id != null)
  | select(any(.event == "audit.started") and
           (any(.event == "audit.completed" or .event == "audit.failed") | not))
  | .[0].correlation_id
' logs/graphtraces.jsonl
```

## Безопасность и объём

Трейс содержит полные данные приёма, промпты, ответы модели, чанки КР и
лекарственные назначения — то есть медицинские и персональные данные. Новый
файл создаётся с правами `0600`, не добавляется в Git и должен храниться,
передаваться и удаляться по тем же правилам, что и исходные карты. Ротация
автоматически не выполняется: её следует настроить средствами окружения или
указывать новый `GRAPH_TRACE_PATH` для каждого периода запуска.

Запись fail-open: ошибка файловой системы не останавливает аудит. После запуска
следует отдельно проверить существование файла, его права и появление событий
`audit.started`/`audit.completed`.
