# E2E-набор: остальные роуты, аудит по типам карт, методика

**Дата:** 2026-08-20 · **Статус:** утверждён · **Ветка:** `push-log-e2e-tests`
**Связано:** `e2e/tests/test_push_log_smoke.py` (единственный существующий e2e-тест на этой ветке),
`e2e/tests/helpers/{organizations,api_keys,cards}.py`, `docs/superpowers/specs/2026-08-20-e2e-push-log-smoke-design.md`
(спека первого теста — этот документ продолжает её паттерн, не меняет его),
`src/api/routes/visits.py`, `src/api/routes/stats.py`, `src/audit/pipeline.py`,
`src/audit/formal_structure/{validator.py,rules.json}`, `docs/formal_validator.md`, `docs/clinic-data-requirements.md`

## Зачем

На ветке `push-log-e2e-tests` есть один e2e-тест (`push_log`/`push_metrics_by_date`) и переиспользуемые
фикстуры под него. Нужно:

1. Покрыть e2e-тестами остальные роуты pull-API (`/visits/check`, `/visits/pull`, `/visits/export`,
   `/visits/check_updates`, `/visits/doctors`, `/stats/storage`) — сейчас у них есть только
   `scripts/smoke-cards-push.sh` (только `/visits/push`) и старый `scripts/smoke-push-check-updates.py`.
2. Покрыть e2e-тестами прогон аудита по типам карт — один тест на один `visit_type` × `age_group`
   (10 тестов), с реальными LLM-вызовами (`FormalValidator` + `DiagnosisValidator`) и ожидаемыми
   находками.
3. Задокументировать общую методику e2e — как писать, запускать и расширять эти тесты.

## Решение

### 1. Route-тесты — по одному standalone-скрипту на роут

Паттерн `test_push_log_smoke.py` без изменений: `argparse` с позиционным `url` (default `"local"` →
`http://localhost:{API_PORT}`), `--keep`, `check(label, condition, detail)`-аккумулятор, teardown в
`finally` даже при падении/Ctrl-C, exit-код по числу неудачных проверок. Каждый скрипт создаёт свою
org+key+card через `e2e/tests/helpers/*` и убирает их за собой — никакого общего состояния между
скриптами (решение пользователя: раздельные скрипты, не один общий).

Новые файлы:

```
e2e/tests/test_visits_check_smoke.py
e2e/tests/test_visits_pull_smoke.py
e2e/tests/test_visits_export_smoke.py
e2e/tests/test_visits_check_updates_smoke.py
e2e/tests/test_visits_doctors_smoke.py
e2e/tests/test_stats_storage_smoke.py
```

Каждому нужна карта в статусе `done` с управляемой датой/врачом — существующий
`CardFixtures.stage_audited` не подходит один в один (он не проставляет `Прием.DATE`/`Врач_код`,
только флаги аудита). Добавляем в `e2e/tests/helpers/cards.py`:

```python
async def stage_done_with_meta(
    self, card_guid: str, *, visit_date: str, doctor_code: str | None = None, doctor_name: str | None = None,
) -> None:
    """UPDATE done_cards SET status='done', formal_result=<fake finding>, ignored=FALSE, broken=FALSE,
    card_data = card_data || jsonb с обновлённым Прием.DATE (и Прием.Врач_код/Врач, если заданы).
    Расширяет stage_audited полями, которые нужны check/pull/export/doctors для фильтрации по дате/врачу."""
```

Не трогаем существующий `stage_audited` (использует `push_log_smoke`) — новый метод рядом, с
собственным docstring, объясняющим отличие.

#### `test_visits_check_smoke.py`

1. Push карты → `stage_done_with_meta(visit_date="<сегодня DD.MM.YYYY>")`.
2. `GET /visits/check?date=<сегодня>` → 200, `count == 1` (создаём для свежей org — база пуста для
   неё, абсолютное сравнение, не дельта).
3. `GET /visits/check?date=<вчера>` → 200, `count == 0`.
4. `GET /visits/check?date=...&org=unknown-org-xxxx` (с валидным ключом другой org) → 404.
5. Без `Authorization` → 401. С валидным чужим ключом (создать второй, не скоуп для этой org) → 403.

#### `test_visits_pull_smoke.py`

1. Та же схема, что check, но:
2. `count > 0` день → `GET /visits/pull?date=` → 200, `Content-Type` = xlsx media type,
   `Content-Disposition` содержит `report_<org>_<date>.xlsx`.
3. День без карт, без `doctor_code` → 404.
4. День без карт, с `doctor_code=любой` → 200 + непустое xlsx-тело (объект "нет приёмов врача ..." —
   `build_empty_report_bytes`), не 404 — это разный путь от п.3 (`ApiFormatter.pull`,
   `src/api/routes/visits.py:59-69`).

#### `test_visits_export_smoke.py`

1. Push + `stage_done_with_meta` для двух разных карт (разные guid) в одной org.
2. `GET /visits/export?org=` без `since`/`limit` → обе карты присутствуют, `status="done"`.
3. `GET /visits/export?since=<будущая метка>` → пусто (граница по `updated_at`, будущее исключает всё).
4. `GET /visits/export?limit=1&cursor=0` затем `cursor=1` → пагинация не дублирует и не теряет карты
   (объединение двух страниц == полный набор без `limit`).
5. Одну из карт помечаем как `ignored` (прямой UPDATE через новый/существующий helper) —
   `include_ignored=false` (по умолчанию) её не возвращает, `include_ignored=true` возвращает.

#### `test_visits_check_updates_smoke.py`

1. Push карты (не аудируем — специально оставляем `pending`, в отличие от export, который
   audited-only).
2. `GET /visits/check_updates?org=` без `since` → карта присутствует (в отличие от `export`, отдаёт и
   pending; `since=None` берёт последнюю неделю — свежий push внутри окна).
3. `since=<время до push>` → карта присутствует (граница включающая).
4. `since=<время после push>` → карты нет.

#### `test_visits_doctors_smoke.py`

1. Push двух карт с разными `Прием.Врач_код`/`Врач` в одной org, `stage_done_with_meta` с этими полями.
2. `GET /visits/doctors?org=` → оба врача есть, отсортированы по имени (`ORDER BY name, code` в
   `fetch_doctors`).
3. Третья карта с тем же `Врач_код`, но другим `Врач` (переименование) → `doctors` отдаёт последнее
   имя по `updated_at DESC` (не дублирует код).

#### `test_stats_storage_smoke.py`

1. Снимаем baseline `GET /stats/storage?org=` для свежесозданной org (`done_cards_kb`/`push_log_kb`
   == 0 — org только что создана, дельта не нужна, как в `push_log`-тесте).
2. Push + `stage_done_with_meta` одной карты.
3. `GET /stats/storage` снова → `done_cards_kb > 0`, `total_kb == done_cards_kb + push_log_kb`
   (`push_log_kb` может остаться 0 — `push_log` пишется отдельным триггером на UPDATE, стадирование
   через прямой SQL его не запускает; тест это не форсирует, просто проверяет равенство).

### 2. Аудит-тесты — 10 файлов, по одному на `visit_type` × `age_group`

Каждый файл — отдельный standalone-скрипт (без CLI-аргументов, без сети/API — не нужен HTTP или
организация): напрямую собирает fixture-визит и гонит его через `AuditPipeline._audit_visit()`,
как `tests/run_single_visit.py`. Решение пользователя: реальные LLM-вызовы, включая
`DiagnosisValidator` (значит каждой фикстуре нужен диагноз с реальным МКБ-кодом, для которого в
`guidelines`/`docs` есть склад клинрекомендаций).

```
e2e/tests/audit/
  __init__.py
  fixtures.py                      # общий helper: собрать visit dict, минимальные общие поля
  test_audit_primary_adult.py
  test_audit_primary_child.py
  test_audit_repeat_adult.py
  test_audit_repeat_child.py
  test_audit_prophylactic_adult.py
  test_audit_prophylactic_child.py
  test_audit_lab_research_intervention_adult.py
  test_audit_lab_research_intervention_child.py
  test_audit_prophylactic_tuberculin_adult.py
  test_audit_prophylactic_tuberculin_child.py
```

#### Фикстуры — деконструированы под `get_visit_types`, не назначены вручную

Ключевое ограничение (уточнение пользователя): тип визита должен быть **угадан системой** через
`FormalValidator.get_visit_types()` — детерминированный разбор NMU-кода/ключевых слов в `Услуги`
(`src/audit/formal_structure/validator.py:116-195`), а не передан в обход. Каждый тест поэтому:

1. Собирает `visit` dict с `Услуги`, несущими код, который по таблице из `docs/formal_validator.md`
   классифицируется в нужный тип:

   | Тест | NMU-код в `Услуги` | Ожидаемый `VisitType` |
   |---|---|---|
   | primary | `B01.070.001` | `PRIMARY` |
   | repeat | `B01.070.011` | `REPEAT` |
   | prophylactic | `B04.031.002` (педиатр, child) / `B04.047.002` (терапевт, adult) | `PROPHYLACTIC` |
   | lab_research_intervention | `A05.10.006` (ЭКГ) | `LAB_RESEARCH_INTERVENTION` |
   | prophylactic_tuberculin | `Диагноз.Код = "Z11.1"` (не NMU-код в `Услуги` — отдельная ветка в
     `get_visit_types`) | `PROPHYLACTIC_TUBERCULIN` |

   Только префикс/суффикс кода перед последней группой цифр определяет классификацию
   (`get_visit_types` матчит по `prefix`/`middle`/`last` частям, см. §validator.py:160-173) — точный
   5-значный код специальности после `B04.` (`031` педиатр vs `047` терапевт) не влияет на
   `VisitType`, это только для правдоподобия фикстуры; при реализации годится любой `B04.*.002`.

2. **Первым шагом теста** (до вызова pipeline) вызывает `FormalValidator().get_visit_types(visit)` и
   ассертит, что результат — ровно ожидаемый `{VisitType.X}` (или содержит его, если код допускает
   объединение с `PROPHYLACTIC_TUBERCULIN` через `Диагноз.Код`). Если фикстура классифицируется не
   так, как задумано, тест падает здесь — до дорогого LLM-вызова, с понятной причиной ("фикстура не
   долетела до системы правильно", а не "LLM что-то не нашла").
3. `Пациент.AGE` — `8` для `child`, `45` для `adult` (порог `_ADULT_THRESHOLD = 15` в
   `clinic_recs.py`, а формальные правила используют `< 18` — оба теста используют значения, чётко
   попадающие в свою группу по обеим границам: `8 < 15` и `8 < 18`, `45 > 15` и `45 >= 18`).
4. Один диагноз — `КодМКБ: "J06.9"` (острый тонзиллофарингит, guideline `306_3`, `age_category` =
   `{Взрослые,дети}` — единственный найденный в текущей БД код с гайдлайном на обе возрастные
   группы, что позволяет переиспользовать один и тот же ICD-код во всех 10 тестах вместо подбора
   разных кодов под каждую возрастную группу). Диагноз присутствует всегда (условие пользователя —
   DiagnosisValidator должен реально отработать), кроме случая, когда сама цель теста — правило типа
   `diagnosis_required` (см. ниже, primary/child).
5. Каждая фикстура **намеренно** нарушает ровно одно применимое правило из `rules.json`, чтобы
   ожидаемая находка была конкретной, а не "что угодно, что найдёт LLM":

   | Тест | Что убрано/испорчено в фикстуре | Ожидаемый `flag_code` |
   |---|---|---|
   | primary_adult | Нет `objective_exam` в `ДанныеОсмотра` | `ПЕРВИЧНЫЙ_ОТСУТСТВУЮТ_ОСНОВНЫЕ_РАЗДЕЛЫ` или `ОТСУТСТВУЕТ_ОБЪЕКТИВНЫЙ_ОСМОТР` |
   | primary_child | Нет `Диагнозы` вовсе (пустой массив) — единственная фикстура без диагноза, т.к. цель — проверить именно это правило; `DiagnosisValidator` в этом одном тесте не вызывается (pipeline.py:199, ожидаемое поведение, не баг) | `ОТСУТСТВУЕТ_ДИАГНОЗ` (`age_group=child`-специфичное правило) |
   | repeat_adult | Нет `dynamics`/динамики состояния в `ДанныеОсмотра` | `ПОВТОРНЫЙ_ОТСУТСТВУЕТ_ДИНАМИКА` |
   | repeat_child | `Услуги.Наименование` содержит слово "первичный" при NMU-суффиксе `.011` (repeat) | `NMU_CODE_CONTRADICTION` (детерминированная проверка, не LLM — `_check_nmu_keyword_contradiction`, надёжнее для теста) |
   | prophylactic_adult | В `ДанныеОсмотра` план обследования содержит уже готовый результат (термин "результат:" внутри поля плана) | `СМЕШАНЫ_ПЛАН_И_РЕЗУЛЬТАТЫ` |
   | prophylactic_child | Заглушка вида `"уточнить"`/`"-"` в одном из значимых полей | `ОБНАРУЖЕНЫ_ЗАГЛУШКИ` |
   | lab_research_intervention_adult | Для ЭКГ отсутствует заключение (только протокол, без итоговой интерпретации) | `ФУНКЦИОНАЛЬНОЕ_ИССЛЕДОВАНИЕ_НЕПОЛНОЕ` (`ecg_functional_description_and_conclusion`) |
   | lab_research_intervention_child | Дублирующиеся смысловые блоки (одна и та же жалоба продублирована дважды разными словами) | `ДУБЛИРОВАНИЕ_СМЫСЛОВЫХ_БЛОКОВ` (`duplicate_semantic_blocks_are_defect`) |
   | prophylactic_tuberculin_adult | Нет объективных данных туберкулинодиагностики | `ТУБЕРКУЛИН_ОТСУТСТВУЮТ_ОБЪЕКТИВНЫЕ_ДАННЫЕ` (`tuberculin_objective_data_required`) |
   | prophylactic_tuberculin_child | Нет заключения по пробе | `ТУБЕРКУЛИН_ОТСУТСТВУЕТ_ЗАКЛЮЧЕНИЕ` (`tuberculin_conclusion_required`) |

6. Тест запускает `AuditPipeline()._audit_visit(visit)` напрямую (без HTTP, без org — pipeline
   принимает `org_id: str | None = None`), получает `Result`, проверяет:
   - `result.formal.findings` содержит finding с ожидаемым `flag_code` (проверка **присутствия**,
     не точного списка — LLM недетерминирован, могут быть дополнительные находки).
   - `result.diagnosis` непустой (кроме primary_child) и хотя бы один `DiagnosisResult.guideline_file_id
     == "306_3"` — подтверждает, что `ClinicRecs` нашла гайдлайн, а не откатилась в "нет рекомендаций".
7. Teardown: `AuditPipeline._audit_visit` не пишет в БД сама по себе — запись делает
   `run_batched`/`_upsert_done_card`, вызываемый только внутри `run_batched`. Прямой вызов
   `_audit_visit()` (как в `tests/run_single_visit.py`) ничего не персистит — значит **отдельного
   teardown для аудит-тестов не требуется**, они не трогают БД вообще, только `GuidelinesStorage`
   (read) и `LLM`-клиент. Это единственная причина, по которой можно позволить себе вызывать
   `_audit_visit` напрямую, а не через `run_batched`+`DoneCardsStorage`.

`e2e/tests/audit/fixtures.py` — общий helper, чтобы 10 файлов не копировали один и тот же
boilerplate:

```python
def base_visit(*, guid: str, nmu_code: str | None, service_name: str, age: int,
                diagnosis_icd: str | None = "J06.9", extra_diagnoses: list[dict] | None = None,
                tuberculin: bool = False) -> dict:
    """Собирает валидный по docs/clinic-data-requirements.md visit dict с одной услугой,
    несущей nmu_code, и (опционально) одним диагнозом. Каждый test_audit_*.py достраивает
    ДанныеОсмотра под свой сценарий поверх этого скелета — не через параметры (иначе
    fixtures.py разрастётся под 10 разных наборов kwargs), а прямым dict-мёржем в самом тесте."""
```

### 3. Методика e2e — `docs/e2e-testing.md`

Новый файл, ссылается на оба спека (`push_log_smoke` + этот). Разделы:

- **Что такое e2e в этом репозитории и что не e2e.** e2e = реальный HTTP API + реальный Postgres (для
  route-тестов) или реальный `AuditPipeline` + реальные LLM-вызовы (для аудит-тестов) — никаких
  моков. Отличие от `tests/` (pytest, `pythonpath=src`, обычно с моками LLM) и от `scripts/smoke-*`
  (более старые скрипты с похожим паттерном, которые e2e не заменяет и не дублирует).
- **Когда писать новый e2e-скрипт.** Новый route → новый `test_visits_<route>_smoke.py`. Новое
  поведение внутри существующего route (как `push_log` для `/visits/push`) → новый файл рядом с
  существующим для этого route, не расширение старого (пример: `test_push_log_smoke.py` не трогает
  `/visits/push`'ные проверки из `smoke-cards-push.sh`, специально их не дублирует).
- **Контракт хелперов** (`e2e/tests/helpers/organizations.py`, `api_keys.py`, `cards.py`) — что
  каждый даёт, когда добавлять новый метод в существующий helper vs. создавать новый.
- **Паттерн скрипта**: `argparse` (`url` positional default `"local"`, `--keep`), `TAG = uuid4().hex[:8]`
  для изоляции от параллельных прогонов на общей БД, `check()`-аккумулятор, `finally`-teardown,
  ненулевой exit code при провале. Копия шаблона из `test_push_log_smoke.py` целиком как
  "скопируй и переименуй".
- **Как запускать.** Индивидуально: `python e2e/tests/test_X_smoke.py [url] [--keep]`. Все
  route-тесты разом: `for f in e2e/tests/test_*_smoke.py; do python "$f" || exit 1; done` (простой
  bash-цикл, не отдельный раннер — скриптов немного, отдельная обвязка избыточна). Аудит-тесты
  отдельно, т.к. дороги: `for f in e2e/tests/audit/test_*.py; do python "$f" || exit 1; done`,
  с явным предупреждением в доке, что это тратит реальные LLM-токены.
  API-тесты требуют поднятого API (`uvicorn`/докер) — аудит-тесты нет (не ходят по HTTP).
- **Недетерминизм LLM.** Правило: ассертить *наличие* конкретного `flag_code`/находки, никогда не
  ассертить точный список находок или точный текст `issue`. Формулируется явно, чтобы будущие
  аудит-тесты не превращались в хрупкие snapshot-проверки.
- **Изоляция и cleanup.** Почему `TAG`, зачем `finally`, почему `--keep` существует (ручная отладка
  на реальной БД), что аудит-тесты исключение — им cleanup не нужен (см. §2 пункт 7).

## Не в скоупе

- Оборачивание e2e-скриптов в pytest — решение пользователя: standalone-скрипты, как есть.
- Мок LLM для аудит-тестов — решение пользователя: реальные вызовы.
- CI-интеграция/автозапуск e2e на pipeline (GitHub Actions и т.п.) — не запрошено, e2e остаётся
  ручным инструментом, как и `push_log_smoke`.
- Изменение существующих `scripts/smoke-cards-push.sh` / `smoke-push-check-updates.py` на новые
  хелперы — не в скоупе ни у первой спеки, ни у этой.
