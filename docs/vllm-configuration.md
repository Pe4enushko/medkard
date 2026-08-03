# vLLM configuration and endpoint checks

Проверено 2026-08-03 на `http://192.168.1.80:8010`.

## Что запущено

- vLLM: `0.20.2rc1.dev201+g215e2f799.d20260510`
- модель: `Qwen/Qwen3.6-35B-A3B-FP8`
- `max_model_len`: `100000`
- текущий endpoint — генеративный `/v1/chat/completions`; `/rerank` отвечает `404 Not Found`.

## Результаты тестов

| Запрос | Результат |
|---|---|
| `response_format.type=json_schema` + `chat_template_kwargs.enable_thinking=false` | Работает: валидный JSON, `content` заполнен, `finish_reason=stop`. |
| top-level `structured_outputs.json` + `enable_thinking=false` | Работает: валидный JSON, `finish_reason=stop`. |
| top-level `guided_json` + `enable_thinking=false` | Запрос принимается, но модель возвращает Markdown fences; не использовать как основной режим. |
| Любой structured request с `max_tokens=80` без отключения thinking | `content=null`, reasoning съедает лимит, `finish_reason=length`. |
| `repetition_penalty` + `repetition_detection` | Запрос принимается; в коротком тесте детектор не сработал, потому что модель не вошла в повтор. |

Основной рабочий вариант для текущей модели:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "Output",
      "schema": {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": false
      }
    }
  },
  "chat_template_kwargs": {"enable_thinking": false},
  "max_tokens": 2048
}
```

В OpenAI Python SDK `chat_template_kwargs`, `repetition_penalty` и `repetition_detection` передаются через `extra_body`; при прямом HTTP они должны находиться в корне JSON-запроса.

`repetition_detection` — это защита от повторяющихся n-грамм внутри одной генерации. Она не видит семантический цикл LangGraph, поэтому в проекте дополнительно нужен `ToolCallGuard`.

## Настройки проекта

Они собираются в `src/LLM/vllm_config.py` и автоматически включаются, если `OPENAI_BASE_URL` не указывает на `api.openai.com`. Для явного управления используйте `VLLM_PARAMS_ENABLED=true|false`.

В agent path теперь заданы:

- native structured output через `create_react_agent(response_format=...)`;
- `enable_thinking=false`;
- `max_completion_tokens=2048`;
- `repetition_penalty=1.05`;
- `repetition_detection={max_pattern_size: 20, min_pattern_size: 3, min_count: 4}`;
- `temperature=0.2`.

Значения являются стартовым профилем. После сбора telemetry их следует откалибровать на медицинских терминах и кодах МКБ.

## Отдельный rerank server

Текущий `8010` нельзя использовать для rerank: на нём загружена generative Qwen-модель, а `/rerank` отсутствует. vLLM поддерживает Cohere/Jina-compatible rerank API для score/pooling моделей; нужен отдельный процесс, например:

```bash
vllm serve BAAI/bge-reranker-v2-m3 \
  --runner pooling \
  --host 0.0.0.0 \
  --port 8011
```

После проверки конкретной модели:

```text
RERANK_BASE_URL=http://192.168.1.80:8011
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_CANDIDATE_LIMIT=20
RERANK_TIMEOUT_SECONDS=10
```

Приложение сначала получает ограниченный candidate set через HNSW + BM25 + RRF, затем отправляет до 20 chunks на `/rerank` и возвращает только `top_k`. При недоступности reranker автоматически остаётся RRF-порядок.

Документация vLLM: [scoring usages](https://docs.vllm.ai/en/latest/models/pooling_models/scoring/) и [v0.20 score examples](https://docs.vllm.ai/en/v0.20.2/examples/pooling/score/).
