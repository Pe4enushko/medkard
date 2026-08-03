# Отчёт по `broken_cards`

Дата анализа: 2026-08-03. Источник: `broken_cards.csv` (96 строк, 32 точных текста traceback, 18 нормализованных вариантов). Ниже разделены первопричины, а не только классы исключений.

## Итог

Главная проблема — не SQL и не размер исходной JSON-карты сами по себе. 81 из 96 сбоев приходится на ICD ReAct-агент: он либо не завершает цикл tool-call → model-call, либо раздувает историю до лимита. В коде нет защиты от повторного вызова одного и того же инструмента, лимита токенов/размера tool-result, `repetition_penalty` и vLLM `repetition_detection`.

| Класс | Количество | Контур | Вывод |
|---|---:|---|---|
| `GraphRecursionError`, limit 25 | 56 | ICD | Агент не пришёл к финальному ответу за 25 шагов. |
| `GraphRecursionError`, limit 50 | 11 | ICD | Увеличение лимита не устранило цикл, а позволило ему тратить больше токенов. |
| Input context `>= 70000` | 13 | ICD | История ReAct уже разрослась; сервер получил prompt минимум 70001 токен. |
| `LengthFinishReasonError` | 3 | 2 ICD, 1 diagnosis | Генерация упёрлась в общий лимит 70000/100000 токенов. |
| Invalid JSON | 7 | 5 diagnosis, 2 ICD | Модель вернула Markdown/`<tool_call>` вместо ожидаемого JSON. |
| Нет `parsed`/`refusal` | 5 | 4 diagnosis, 1 ICD | vLLM вернул `finish_reason=stop`, но structured-output объект не сформировал. |
| `issues.0` — строка вместо объекта | 1 | diagnosis | Нарушена схема `CheckerOutput`. |

Итого: `67 + 13 + 3 + 7 + 5 + 1 = 96`.

## Как устроен проблемный путь

1. `AuditPipeline` для визита сначала формирует формальную проверку, затем передаёт все возрастно-допустимые guidelines в ICD checker: [pipeline.py:211](src/audit/pipeline.py#L211), [pipeline.py:216](src/audit/pipeline.py#L216).
2. ICD checker вставляет весь manifest в пользовательский prompt. В выгрузке и старых логах встречается `manifest rows=558`: [icd_check/validator.py:54](src/audit/icd_check/validator.py#L54), [icd_check/validator.py:124](src/audit/icd_check/validator.py#L124).
3. ICD prompt требует сначала получить оглавление guideline, затем читать разделы `1.x` последовательно. Инструмент чтения возвращает **все chunks раздела**, без ограничения размера: [tools.py:266](src/LLM/tools.py#L266), [tools.py:287](src/LLM/tools.py#L287).
4. LangGraph продолжает цикл, пока модель не выдаст финальный structured response. Единственный встроенный предохранитель — `recursion_limit`: [client.py:163](src/LLM/client.py#L163), [client.py:166](src/LLM/client.py#L166).
5. Любая ошибка агента, включая `GraphRecursionError`, ловится как общий `Exception` и повторяется ещё до двух раз с тем же тяжёлым prompt: [client.py:200](src/LLM/client.py#L200), [client.py:208](src/LLM/client.py#L208).

Это объясняет последовательность: повторный tool-call раздувает историю → следующий model-call получает больше контекста → vLLM отвечает `input_tokens=70001` или продолжает генерировать до `length` → retry повторяет тот же сценарий.

## Уникальные корневые классы

### 1. Зацикливание ICD ReAct — 67 карт

Трейсы не содержат последовательности tool-call, поэтому по одному `stacktrace` нельзя доказать, повторялся ли конкретно `get_guideline_structure`, `read_guideline_section`, один и тот же раздел или переход между несколькими кандидатами. Но все 67 traceback проходят через `audit/icd_check/validator.py`, а код имеет все предпосылки для цикла:

- prompt даёт агенту таблицу из сотен кандидатов;
- агенту разрешено читать разделы последовательно «пока не получит достаточно оснований»;
- нет множества уже прочитанных `(file_id, section)`;
- нет запрета на повтор одинакового вызова;
- `read_guideline_section` возвращает полный раздел;
- `AGENT_MAX_STEPS` — это только аварийный потолок, не детектор цикла.

Распределение по времени показывает, что смена потолка с 25 на 50 шагов не решила проблему: 56 случаев с limit 25 приходятся на 2026-06-25—2026-07-01, 11 случаев с limit 50 — на 2026-07-03—2026-07-29.

Вероятность цикла особенно высока, если у карты много диагнозов, один код имеет несколько guideline-кандидатов, возраст пациента не определён или в manifest много строк с одним префиксом МКБ. Но сама карта может быть небольшой: это нужно проверять по `card_data`, а не предполагать по SQL-строке из view.

### 2. Переполнение входного контекста — 13 карт

Все 13 ошибок возникают в ICD checker, с сообщением модели: prompt содержит минимум 70001 токен, output budget уже равен нулю. Это не «модель слишком много ответила»: это уже слишком большая **входная история**.

GUID для обязательной проверки карты и связанных guidelines:

```text
f0be29f6-bb63-4038-a365-135deb8518b0
17c18a7c-056e-4d8a-aa4a-91328957c04c
6a75e9c6-a897-430c-a7dc-5154029601df
c54840ff-22ca-4d01-b767-83ca4a68a454
c8addb37-3501-4774-a8a2-0a88bfc508fa
21704926-22ae-4048-a42b-3a400f559ae0
a5ea1036-f231-4792-9194-3443c842a78c
edfbbe28-5b87-4812-b6e3-a1b122e22a68
c50df8ef-5b92-44b6-90fa-5553f33196f1
6bc711e3-70ca-48c9-b702-9dd82178873e
48a45e23-8585-418d-81e4-58968473ffe4
5e66809f-d5b2-4142-8e0c-cd03ab58892f
fa20c798-fefc-46bd-9f16-19d58940c065
```

Проверять нужно не только размеры карты, но и:

- число диагнозов и кодов МКБ;
- возраст пациента и число строк manifest после age-фильтра;
- число guideline-кандидатов для каждого кода/префикса;
- количество chunks и суммарный размер секций `1.x` в `docs`;
- повторяющиеся вызовы и размеры ответов tools в runtime-логах.

Локальный контрольный пример: GUID `21704926-22ae-4048-a42b-3a400f559ae0` найден в `data_snapshots/one_c_MDS_15-07-2026_to_15-07-2026.json`. Карта занимает примерно 4.6 KB, содержит один диагноз и 10 элементов `ДанныеОсмотра`, но в логах для неё был manifest из 558 строк. Значит, «тяжёлая карта» в этой системе может означать тяжёлый производный prompt/agent history, а не большой исходный JSON.

### 3. Выходной лимит — 3 карты

| GUID | Контур | Наблюдение |
|---|---|---|
| `0f79e555-535e-47d1-9895-2d3d8fcde3c5` | diagnosis | `completion_tokens=95813`, `prompt_tokens=4187`, total 100000. |
| `dcde43c4-3d7f-4458-a91d-9328222e0d29` | ICD | `completion_tokens=58753`, `prompt_tokens=41247`, total 100000. |
| `b7f2eff4-53c7-463f-aff1-2550a20c2085` | ICD | `completion_tokens=15254`, `prompt_tokens=54746`, total 70000. |

Для diagnosis это очень сильный признак генерационного зацикливания или бесконтрольного reasoning/tool-call, а не просто длинного нормального JSON: итоговый объект `issues` обычно мал. Для ICD это может быть комбинация большой manifest-таблицы, длинной истории чтения guideline и отсутствия остановки.

### 4. Structured output несовместим с фактическим ответом модели — 13 карт

Сюда входят 7 `Invalid JSON`, 5 случаев отсутствующего `parsed/refusal` и 1 неправильная форма `issues.0`.

Наблюдаемые варианты ответа:

- JSON в Markdown fences: ` ```json ... ``` `;
- `<tool_call>...</tool_call>` вместо финального объекта;
- обычный русский текст;
- `issues` содержит строку, а не объект `DiagnosisIssue`;
- `finish_reason=stop`, но `content=''`, `parsed=None`, `refusal=None`.

Это уже не обязательно дефект карты. В текущем коде agent path использует отдельный `ChatOpenAI` с `temperature=0.7` и передаёт только `response_format` в `create_react_agent`: [rag_agent.py:98](src/LLM/rag_agent.py#L98), [rag_agent.py:104](src/LLM/rag_agent.py#L104). В отличие от raw path, agent path не получает настроек sampling, `max_tokens` или vLLM `extra_body`.

Часть ошибок может быть связана с моделью `Qwen/Qwen3.6-35B-A3B-FP8` и vLLM `0.20.2rc1.dev...`, которые указаны непосредственно в traceback. Нужно отдельно подтвердить поддерживаемую комбинацию `create_react_agent(response_format=...)` + chat template + structured outputs на текущем build.

## Что обязательно смотреть в самой карте

SQL/view достаточно, чтобы сгруппировать ошибки по GUID, организации, дате и финальному traceback. Для следующих случаев SQL без `card_data` недостаточен:

| Случай | Что смотреть в карте обязательно | Что дополнительно смотреть вне карты |
|---|---|---|
| Все 67 recursion | `Диагнозы`, возраст/пол, `ДанныеОсмотра`, `Услуги`, любые нестандартные верхнеуровневые поля | полный tool-call trace; повтор `(tool, arguments)`; manifest и chunks выбранных guidelines |
| Все 13 input overflow | длину каждого клинического поля и число диагнозов | размер manifest, размер каждой tool-выдачи, длину всей LangGraph history |
| 3 length | для diagnosis — `ДанныеОсмотра`, `Назначения`, `Рекомендации`, большие текстовые поля; для ICD — diagnosis + manifest | `prompt_tokens`, `completion_tokens`, finish reason, reasoning tokens |
| 13 schema/JSON | только если в карте есть необычно длинный/инъекционный текст | raw request/response, chat template, structured-output режим vLLM |

Особенно важно, что `_format_visit_context` включает не только заранее перечисленные поля, но и все прочие поля, кроме `Пациент`, `Диагнозы`, `Прием`, `Врач`: [diagnosis/validator.py:151](src/audit/diagnosis/validator.py#L151), [diagnosis/validator.py:165](src/audit/diagnosis/validator.py#L165). Поэтому размер надо считать после сериализации prompt, а не только по размеру JSON.

## Не выполненные профилактические меры на момент исходной выгрузки

### На уровне генерации

- `repetition_penalty` отсутствует во всех вызовах агента. В коде есть только `temperature=0.7`: [rag_agent.py:98](src/LLM/rag_agent.py#L98).
- `frequency_penalty` и `presence_penalty` не заданы.
- `top_p`, `top_k`, `min_p` не заданы; фактические defaults могут прийти из `generation_config.json` модели.
- `max_tokens`/резерв output budget для agent path не задан. Поэтому при полном prompt сервер сообщает `requested 0 output tokens`.
- vLLM `repetition_detection` не задан.
- нет `stop`/`stop_token_ids` для известных завершающих маркеров JSON/tool protocol. Это дополнительная мера, не замена graph guard.
- нет отдельного низкого `temperature` для механического structured output: checker agents работают на 0.7.

### На уровне LangGraph и приложения

- `recursion_limit` только ограничивает число шагов; он не обнаруживает повтор одного tool-call.
- нет лимита числа вызовов конкретного tool, числа уникальных sections или суммарных tool-result tokens.
- нет cache для `get_guideline_structure(file_id)` и `read_guideline_section(file_id, section)`.
- нет защиты от повторной пары `(tool_name, normalized_args)`.
- нет preflight token count и мягкого отказа до отправки запроса.
- нет ограничения размера `read_guideline_section`; возвращается весь раздел без budget.
- `GraphRecursionError` и context `BadRequestError` попадают под общий retry. Повторять заведомо неисправимый prompt три раза нельзя.
- нет отдельной ветки fallback: «сократить карту/manifest → повторить», «перейти к non-agent structured call» или «сохранить диагностический отказ».
- логируется traceback, но не сохраняется компактный audit trail tool calls, поэтому из `broken_cards` нельзя восстановить конкретную петлю.

## Рекомендованные исправления

### P0 — остановить расход контекста и токенов

1. Для ICD ввести hard budget: например, 8–12 graph steps, максимум 1 `get_guideline_structure` на `file_id`, максимум 1 чтение каждой секции и максимум 2–3 секции на кандидат. При нарушении — контролируемый `agent_loop_detected`, без retry.
2. Добавить нормализованный duplicate-call guard. Для одинакового `(tool, file_id, section)` возвращать cached result или сообщение «этот раздел уже прочитан; переходи к выводу», а не делать новый вызов.
3. Не передавать все 558 manifest rows. Сначала SQL-фильтр по точному коду, префиксу, возрасту и названию; в агент отдавать только небольшой shortlist. В коде уже есть TODO на эту оптимизацию: [pipeline.py:213](src/audit/pipeline.py#L213).
4. Ограничить размер tool output и итогового prompt. Для секции возвращать top-N chunks или свёрнутый extract с жёстким char/token budget.
5. Для `GraphRecursionError`, context overflow и `LengthFinishReasonError` выключить обычный retry либо повторять только после уменьшения входа. Сейчас retry повторяет ту же причину: [client.py:200](src/LLM/client.py#L200).

### P1 — подключить защиту vLLM 0.20

В OpenAI SDK параметры vLLM передаются через `extra_body`; официальная документация 0.20.2 прямо указывает этот способ для `top_k`, а в API перечисляет `repetition_penalty` и `repetition_detection`: [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/v0.20.2/serving/openai_compatible_server/). `RepetitionDetectionParams` присутствует уже в vLLM 0.17 и завершает генерацию при повторяющихся n-граммах: [vLLM 0.17 sampling params](https://docs.vllm.ai/en/v0.17.0/api/vllm/sampling_params/).

Стартовый профиль для проверки, не догма:

```python
extra_body = {
    "repetition_penalty": 1.05,
    "frequency_penalty": 0.10,
    "presence_penalty": 0.00,
    "top_p": 0.90,
    "top_k": 50,
    "repetition_detection": {
        "max_pattern_size": 20,
        "min_pattern_size": 3,
        "min_count": 4,
    },
}
```

Для structured JSON сначала протестировать без penalty и затем включать по одному: penalty способен испортить редкие медицинские термины/коды. `repetition_detection` — именно аварийный детектор, а не способ исправить плохой prompt. Его нужно обрабатывать как отдельный finish reason и не считать валидным structured result.

Также явно задать `max_tokens` для каждой операции и запускать vLLM с контролируемым `--generation-config vllm`, если defaults модели неожиданны. Документация предупреждает, что `generation_config.json` модели может переопределять sampling defaults: [vLLM serving](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/).

### P1 — упростить structured output

- Для ICD и трёх diagnosis checker-ов разделить «поиск» и «финальный JSON»: ограниченный retrieval-фан-аут, затем один обычный structured call без ReAct history.
- Либо явно настроить `ChatOpenAI` agent path на тот же native JSON-schema режим, который уже формируется в raw `LLMClient.call`: [client.py:64](src/LLM/client.py#L64).
- Валидировать `finish_reason`, `content`, `parsed` и `structured_response` до возврата. `stop` с пустым content не должен превращаться в обычный ответ.
- На один repair retry отправлять только исходную клиническую задачу и короткий schema error, не историю из неудавшегося агента.
- Уменьшить temperature checker agents до детерминированного диапазона `0.0–0.2` после проверки совместимости Qwen/vLLM.

### P2 — наблюдаемость и регрессии

Сохранять для каждой LLM-попытки: `card_guid`, checker label, model, prompt tokens, completion tokens, finish reason, graph step, tool name, normalized args hash, tool-result chars/tokens и reason retry. Ввести метрики:

- `agent_duplicate_tool_call_total`;
- `agent_steps_used`;
- `agent_tool_result_tokens`;
- `llm_context_overflow_total`;
- `llm_repetition_detected_total`;
- `structured_output_invalid_total`.

Добавить тестовые карты: маленькая карта, карта с 558 manifest-кандидатами, карта с длинным `ДанныеОсмотра`, карта с повторяющимся tool-call и карта с модельным `<tool_call>` вместо JSON. Тест должен проверять не только итоговый exception, но и что не было третьего одинакового запроса.

## Приоритет просмотра карт

1. Сначала 13 GUID с input overflow — это наиболее быстрый способ найти реальные тяжёлые карты/manifest-кандидаты.
2. Затем 11 карт с limit 50: это наиболее вероятные повторные циклы после уже неудачного увеличения `AGENT_MAX_STEPS`.
3. Затем 56 карт с limit 25 — проверить, есть ли общий диагноз, guideline или пустой/неопределённый возрастной контекст.
4. 13 schema-error карт смотреть после проверки raw model response и vLLM build; карта здесь вторична.

## Что уже изменено после анализа

В текущей рабочей копии после анализа добавлены: native vLLM `json_schema`, отключение thinking для Qwen, bounded `ToolCallGuard`, один compact retry для `GraphRecursionError`, fail-fast для context overflow, optional vLLM rerank и JSONL telemetry. Исторические количества в начале отчёта относятся к исходной выгрузке и не являются результатом нового прогона.

## Вывод

Первое исправление должно быть в ICD agent: уменьшить входной shortlist, ограничить sections/tool-results, детектировать duplicate calls и не retry-ить context overflow. Для цикла теперь предусмотрен один compact retry с меньшими лимитами. `repetition_penalty` и `repetition_detection` в vLLM 0.20 включены как дополнительный предохранитель, но они не остановят LangGraph-цикл сами по себе: vLLM видит только текущую генерацию токенов, а не семантическую историю вызовов tools. Для этого нужен отдельный graph-level guard.
