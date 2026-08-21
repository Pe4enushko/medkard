# E2E-набор: остальные роуты, аудит по типам карт, методика

**Дата:** 2026-08-20 · **Статус:** утверждён · **Ветка:** `push-log-e2e-tests`
**Связано:** `e2e/tests/test_push_log_smoke.py` (единственный существующий e2e-тест на этой ветке),
`e2e/tests/helpers/{organizations,api_keys,cards}.py`, `docs/superpowers/specs/2026-08-20-e2e-push-log-smoke-design.md`
(спека первого теста — этот документ продолжает её паттерн, не меняет его),
`src/api/routes/visits.py`, `src/api/routes/stats.py`, `src/audit/pipeline.py`,
`src/audit/formal_structure/{validator.py,rules.json}`, `docs/formal_validator.md`, `docs/clinic-data-requirements.md`.
Аудит-часть (§2) заимствует архитектуру харнесса (не фикстуры — там другой `rules.json`) с ветки
`formal-rules-npa-revision` (`e2e/tests/audit/{harness.py,fixtures.py,README.md}`), которая
ответвилась от той же точки, что и эта спека, и независимо решала ту же задачу для своего набора
фикстур.

## Зачем

На ветке `push-log-e2e-tests` есть один e2e-тест (`push_log`/`push_metrics_by_date`) и переиспользуемые
фикстуры под него. Нужно:

1. Покрыть e2e-тестами остальные роуты pull-API (`/visits/check`, `/visits/pull`, `/visits/export`,
   `/visits/check_updates`, `/visits/doctors`, `/stats/storage`) — сейчас у них есть только
   `scripts/smoke-cards-push.sh` (только `/visits/push`) и старый `scripts/smoke-push-check-updates.py`.
2. Покрыть e2e-тестами прогон аудита по типам карт — одна фикстура на один `visit_type` × `age_group`
   (10 фикстур, сгруппированные в 5 файлов по `visit_type`), с реальными LLM-вызовами
   (`FormalValidator` + `DiagnosisValidator`) и ожидаемыми находками.
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

### 2. Аудит-тесты — 5 файлов по `visit_type`, 10 фикстур по `visit_type` × `age_group`

Каждый файл — отдельный standalone-скрипт (без CLI-аргументов, без сети/API — не нужен HTTP или
организация): собирает fixture-визит(ы) и передаёт их общему раннеру, который гоняет их через
`AuditPipeline._audit_visit()` напрямую (без `async with` — см. пункт 7). Решение пользователя:
реальные LLM-вызовы, включая `DiagnosisValidator` (значит каждой фикстуре нужен диагноз с реальным
МКБ-кодом, для которого в `guidelines`/`docs` есть склад клинрекомендаций).

**Источник паттерна.** Ветка `formal-rules-npa-revision` (ответвилась от той же точки, что и эта —
коммит спеки — и независимо построила свой набор аудит-фикстур поверх переписанного `rules.json`)
уже прошла через эту задачу и пришла к общему `harness.py` вместо копирования boilerplate в каждый
файл. Ниже — тот же архитектурный паттерн (`Case`/`run_cases`, двухэтапный прогон, сверка *полного*
набора флагов, страховка от неразобранного ответа LLM), адаптированный под `rules.json` и
`get_rules(visit_types, patient_age)` этой ветки (там сигнатура уже другая — добавлен параметр
`icd_prefixes` — портировать нельзя, только сам подход).

```
e2e/tests/audit/
  __init__.py
  fixtures.py                      # dx(), base_visit() — сборка visit dict
  harness.py                       # Case, run_cases() — двухэтапный раннер, общий для всех файлов
  test_audit_primary.py            # 2 Case: adult + child
  test_audit_repeat.py             # 2 Case: adult + child
  test_audit_prophylactic.py       # 2 Case: adult + child
  test_audit_lab_research_intervention.py   # 2 Case: adult + child
  test_audit_prophylactic_tuberculin.py     # 2 Case: adult + child
```

Файлов пять, не десять: harness группирует `Case`-фикстуры по `visit_type` (как файлы на ветке
`formal-rules-npa-revision` группируют по НПА), а `age_group` — измерение внутри списка `CASES`
одного файла, а не отдельный файл. Итоговое число фикстур (10 = 5 × 2) от спеки не меняется, меняется
только то, что несёт единицу изоляции: раньше — файл, теперь — `Case` в списке `CASES` файла на
`visit_type`. Красной линии в 10 отдельных файлов пользователь не требовал — требовалось 10
тестов/фикстур; harness-паттерн даёт то же покрытие с меньшим дублированием, ровно как на
`formal-rules-npa-revision`.

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

2. **Этап 1 харнесса** (до вызова pipeline, без LLM и без БД) вызывает
   `FormalValidator().get_visit_types(visit)` для каждого `Case` и ассертит, что результат — ровно
   `case.visit_types` (или содержит `PROPHYLACTIC_TUBERCULIN`, если фикстура строится через
   `Диагноз.Код = "Z11.1"`). Тем же шагом харнесс вызывает `get_rules(got_types, patient_age)` и
   ассертит, что `case.expect` попал в список выбранных правил (`[r["flag_code"] for r in rules]`) —
   если правило не дошло до промпта, дальше проверять нечего. Если хоть один `Case` не прошёл этап 1,
   этап 2 (дорогой, с LLM) не запускается вовсе — сразу ошибка со списком проваленных кейсов.
3. `Пациент.AGE` — `8` для `child`, `45` для `adult` (порог `_ADULT_THRESHOLD = 15` в
   `clinic_recs.py`, а формальные правила используют `< 18` — оба теста используют значения, чётко
   попадающие в свою группу по обеим границам: `8 < 15` и `8 < 18`, `45 > 15` и `45 >= 18`).
4. Один диагноз — `КодМКБ: "J06.9"` (острый тонзиллофарингит, guideline `306_3`, `age_category` =
   `{Взрослые,дети}` — единственный найденный в текущей БД код с гайдлайном на обе возрастные
   группы, что позволяет переиспользовать один и тот же ICD-код во всех 10 тестах вместо подбора
   разных кодов под каждую возрастную группу). Диагноз присутствует всегда (условие пользователя —
   DiagnosisValidator должен реально отработать), кроме случая, когда сама цель теста — правило типа
   `diagnosis_required` (см. ниже, primary/child).
5. Каждая фикстура **намеренно** нарушает ровно одно применимое правило из `rules.json` — и, что
   отличает этот подход от исходного черновика спеки, **ничего больше**: остальные поля карты должны
   быть безупречны относительно каждого правила, применимого к её `visit_type`/`age_group`, включая
   правила с `"visit_types": ["all"]` (`ОТСУТСТВУЮТ_МЕТАДАННЫЕ_ВИЗИТА`, `ОБНАРУЖЕНЫ_ЗАГЛУШКИ`,
   `ДУБЛИРОВАНИЕ_СМЫСЛОВЫХ_БЛОКОВ`, `has_typos`). Так тест проверяет ожидаемую находку как *конкретную*:

   | Тест (`Case` внутри файла) | Что убрано/испорчено в фикстуре | Ожидаемый `flag_code` |
   |---|---|---|
   | primary_adult | Убрана **только** «Жалобы» (`объективный осмотр` остаётся) — изолирует `primary_core_sections_required` от пересекающегося по цели `objective_exam_required` (оба применимы к `primary`, но только первый таргетит `complaints`) | `ПЕРВИЧНЫЙ_ОТСУТСТВУЮТ_ОСНОВНЫЕ_РАЗДЕЛЫ` |
   | primary_child | Нет `Диагнозы` вовсе (пустой массив) — единственная фикстура без диагноза, т.к. цель — проверить именно это правило; `DiagnosisValidator` в этом одном тесте не вызывается (pipeline.py:199, ожидаемое поведение, не баг) | `ОТСУТСТВУЕТ_ДИАГНОЗ` (`age_group=child`-специфичное правило) |
   | repeat_adult | Нет `dynamics`/динамики состояния в `ДанныеОсмотра` (объективный осмотр и диагноз остаются — иначе задело бы `repeat_core_sections_required`) | `ПОВТОРНЫЙ_ОТСУТСТВУЕТ_ДИНАМИКА` |
   | repeat_child | `Услуги.Наименование` содержит слово «первичный» при NMU-суффиксе `.011` (repeat), остальная карта полная (динамика, осмотр, диагноз — все на месте) | `NMU_CODE_CONTRADICTION` (детерминированная проверка, не LLM — `_check_nmu_keyword_contradiction`, надёжнее для теста) |
   | prophylactic_adult | В `ДанныеОсмотра` план обследования содержит уже готовый результат (термин «результат:» внутри поля плана) | `СМЕШАНЫ_ПЛАН_И_РЕЗУЛЬТАТЫ` |
   | prophylactic_child | Заглушка вида `"уточнить"` в одном из второстепенных полей (не в тех, что несут диагноз/осмотр — иначе задело бы соседнее правило) | `ОБНАРУЖЕНЫ_ЗАГЛУШКИ` |
   | lab_research_intervention_adult | Для ЭКГ отсутствует заключение (только протокол, без итоговой интерпретации) | `ФУНКЦИОНАЛЬНОЕ_ИССЛЕДОВАНИЕ_НЕПОЛНОЕ` (`ecg_functional_description_and_conclusion`) |
   | lab_research_intervention_child | Дублирующиеся смысловые блоки (одна и та же жалоба продублирована дважды разными словами под разными `Параметр`), заключение по ЭКГ присутствует (иначе задело бы соседнее правило `lab_research_intervention_adult`) | `ДУБЛИРОВАНИЕ_СМЫСЛОВЫХ_БЛОКОВ` (`duplicate_semantic_blocks_are_defect`) |
   | prophylactic_tuberculin_adult | Нет объективных данных туберкулинодиагностики (заключение присутствует — иначе задело бы соседнее правило) | `ТУБЕРКУЛИН_ОТСУТСТВУЮТ_ОБЪЕКТИВНЫЕ_ДАННЫЕ` (`tuberculin_objective_data_required`) |
   | prophylactic_tuberculin_child | Нет заключения по пробе (объективные данные — размер папулы, гиперемия — присутствуют, как в аналоге на `formal-rules-npa-revision`) | `ТУБЕРКУЛИН_ОТСУТСТВУЕТ_ЗАКЛЮЧЕНИЕ` (`tuberculin_conclusion_required`) |

   Дополнительно к столбцу выше — обязательные условия для **всех** 10 фикстур, иначе сработает
   правило `"visit_types": ["all"]` вторым, незапланированным флагом: без орфографических ошибок
   (`has_typos`), без заглушек вне целевой (`placeholder_values_are_defect`), без дублей вне целевой
   (`duplicate_semantic_blocks_are_defect`), с полными метаданными визита (`visit_meta_required`), а
   для всех child-фикстур — с явно названным законным представителем в осмотре
   (`legal_representative_info`, `age_group=child`), например «Осмотр проведён в сопровождении
   матери» — иначе `ОТСУТСТВУЕТ_ИНФОРМАЦИЯ_О_СОПРОВОЖДАЮЩЕМ` добавится вторым флагом к любой
   child-фикстуре.

6. **Этап 2 харнесса** запускает `AuditPipeline()._audit_visit(case.visit)` напрямую для каждого
   `Case` (без HTTP, без org — конструктор `AuditPipeline()` принимает `org_id: str | None = None`,
   и здесь используется без `async with`, см. пункт 7), получает `Result`, проверяет:
   - **Полный набор** формальных флагов равен `{case.expect}` — `{f.flag for f in
     result.formal.findings} == {case.expect}`, не проверка присутствия. Так тест ловит не только
     пропуск целевого правила, но и правило, которое срабатывает беспричинно (испорченная фикстурой
     карта дала бы ожидаемый флаг в обоих случаях; лишний флаг проявится на *других* по видимости
     нетронутых полях карты только при точной сверке).
   - Ответ формального валидатора **разобран**, а не просто пуст: `LLM.validations` при ошибке
     разбора JSON возвращает `[]` и пишет только `logger.error(...)` — пустой список неотличим от
     «нарушений нет» без отдельной проверки лога. Харнесс слушает логгеры `LLM.validations` и
     `audit.formal_structure.validator` через `logging.Handler` на маркер `"failed to parse JSON
     response"` и явно проваливает кейс, если он встретился — иначе неразобранный ответ молча читался
     бы как «карта чистая».
   - `result.diagnosis` непустой (кроме primary/child — единственной фикстуры без диагноза, см. ниже)
     и хотя бы один `DiagnosisResult.guideline_file_id == "306_3"` — подтверждает, что `ClinicRecs`
     нашла гайдлайн, а не откатилась в «нет рекомендаций».
7. Прогон **без DB-teardown**: `AuditPipeline._audit_visit` вызывает `_upsert_done_card`, но тот
   выходит немедленно, пока `self._done_cards is None` — а это поле заполняется только в
   `AuditPipeline.__aenter__`. Значит харнесс инстанцирует `AuditPipeline()` и вызывает
   `_audit_visit()` напрямую, **не** оборачивая в `async with` — тогда `_upsert_done_card` гарантированно
   не пишет в БД, и убирать за собой действительно нечего. БД всё равно нужна (её читает
   `GuidelinesStorage`/ICD-чекер), но ничего не создаётся и не удаляется.

`e2e/tests/audit/fixtures.py` — общий helper для сборки карт:

```python
def dx(code: str, name: str, *, detail: str = "", first_time: bool = False) -> dict:
    """Один элемент Диагнозы. code управляет и DiagnosisValidator (поиск гайдлайна),
    и, для Z11.1, веткой PROPHYLACTIC_TUBERCULIN в get_visit_types."""


def base_visit(*, guid: str, service_code: str, service_name: str, specialty: str, age: int,
                inspection: list[tuple[str, str]], diagnoses: list[dict],
                gender: str = "Женский", visit_date: str = "20.08.2026") -> dict:
    """Собирает полную карту по docs/clinic-data-requirements.md §3. inspection — список
    (Параметр, Значение) пар, а не готовый список dict — на месте вызова фикстура читается
    как медкарта, а не как JSON. service_code идёт в КодЕГИСЗ и определяет visit_type —
    тип выводится системой, никогда не задаётся тестом напрямую. Пустая строка в service_code
    оставляет услугу неклассифицированной (нужно фикстурам туберкулинодиагностики, чей тип
    приходит только из Диагноз Z11.1). Дата визита фиксирована, не datetime.now() — фикстуры
    ничего не пишут в БД, стабильная дата облегчает воспроизводимость и чтение логов."""
```

`e2e/tests/audit/harness.py` — общий раннер:

```python
@dataclass(frozen=True)
class Case:
    """Одна фикстура-карта и единственный флаг, который должен вызвать её единственный дефект."""
    name: str
    visit: dict
    expect: str
    visit_types: set[VisitType]


async def run_cases(title: str, cases: list[Case]) -> int:
    """Прогоняет все case через оба этапа, возвращает exit code (0 — все проверки пройдены)."""
```

Реализация `run_cases` — как в `harness.py` на `formal-rules-npa-revision` (см. текст файла,
воспроизведённый в плане реализации): `_Report.check()`-аккумулятор, этап 1 (`get_visit_types` +
`get_rules`, без LLM), при провале этапа 1 — немедленный возврат без траты токенов; этап 2
(`_audit_visit` + сверка полного набора флагов + `_FormalCallWatch` для перехвата неразобранного
ответа). Единственное отличие от исходника — вызов `get_rules(got_types, age)` без третьего
аргумента (`icd_prefixes` нет в `rules.json`/`validator.py` этой ветки).

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
  отдельно, т.к. дороги и не принимают аргументов: `for f in e2e/tests/audit/test_audit_*.py; do
  python "$f" || exit 1; done`, с явным предупреждением в доке, что это тратит реальные LLM-токены.
  API-тесты требуют поднятого API (`uvicorn`/докер) — аудит-тесты нет (не ходят по HTTP).
- **Недетерминизм LLM — два разных правила для двух разных видов тестов.** Route-тесты (реальный
  HTTP, без LLM) — обычные точные ассерты, недетерминизма там нет. Аудит-тесты (реальный LLM) сверяют
  **полный** набор формальных флагов с ожидаемым единственным (`{f.flag for f in
  result.formal.findings} == {case.expect}`), а не просто присутствие — решение, принятое по образцу
  `harness.py` на ветке `formal-rules-npa-revision`: проверка на присутствие не отличает работающее
  правило от правила, которое срабатывает всегда, а точная сверка ловит и то, и другое, если каждая
  фикстура несёт ровно один дефект и безупречна во всём остальном (см. §2 пункт 5). Плата — фикстуры
  дороже в написании (нужно явно закрыть все прочие применимые правила), но это цена самой чёткой из
  доступных проверок, не technical debt.
- **Изоляция и cleanup.** Route-тесты: почему `TAG`, зачем `finally`, почему `--keep` существует
  (ручная отладка на реальной БД). Аудит-тесты — исключение из этого контракта: cleanup не нужен, у
  них нет `--keep` и нет аргументов командной строки вовсе, а харнесс намеренно детерминирован в
  своём поведении раз от раза (см. §2 пункт 7 про `AuditPipeline()` без `async with`).

## Не в скоупе

- Оборачивание e2e-скриптов в pytest — решение пользователя: standalone-скрипты, как есть.
- Мок LLM для аудит-тестов — решение пользователя: реальные вызовы.
- CI-интеграция/автозапуск e2e на pipeline (GitHub Actions и т.п.) — не запрошено, e2e остаётся
  ручным инструментом, как и `push_log_smoke`.
- Изменение существующих `scripts/smoke-cards-push.sh` / `smoke-push-check-updates.py` на новые
  хелперы — не в скоупе ни у первой спеки, ни у этой.
