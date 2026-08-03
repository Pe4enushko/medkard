# LLM observability

Каждая попытка LLM и каждый вызов tool записываются в append-only JSONL-файл:

```text
logs/llm_observability.jsonl
```

Путь меняется через `LLM_OBSERVABILITY_PATH`; пустое значение отключает запись.

## Безопасность

Telemetry не сохраняет prompts, card data или tool results. Сохраняются GUID карты, checker label, длины, token usage, hash аргументов tools, исключение и режим retry. Это позволяет прислать файл для разбора без копирования медицинского текста.

## События

| Event | Назначение |
|---|---|
| `llm_start` / `llm_attempt_start` / `llm_attempt_end` | raw LLM call и finish reason каждой попытки |
| `llm_success` / `llm_error` / `llm_retry` | результат или повтор raw-вызова |
| `agent_start` | начало одного checker invocation и его `trace_id` |
| `agent_attempt_start` | лимит шагов и режим `normal`/`compact` |
| `agent_tool` | имя tool, hash аргументов, размеры результата, duplicate/budget flags |
| `agent_retry` | почему и в какой режим ушёл retry |
| `agent_error` | тип, короткий текст исключения и номер попытки |
| `retrieval_rerank` | candidate count, модель reranker и число возвращённых chunks |

## Как быстро подготовить данные для разбора

```bash
python scripts/summarize-llm-observability.py logs/llm_observability.jsonl
python scripts/summarize-llm-observability.py logs/llm_observability.jsonl --card-guid <GUID>
```

Для совместного разбора достаточно прислать JSON-вывод summary и, если нужно, строки JSONL с одним `trace_id`. Полный файл нужен только при анализе межкарточной конкуренции.
