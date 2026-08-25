from __future__ import annotations

import sys
import json
import asyncio
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from LLM.validations import _Findings, validate_rule, validate_visit


class _FakeClient:
    def __init__(self, raw_content: str) -> None:
        self.raw_content = raw_content
        self.response_model: type[Any] | None = None
        self.messages: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] | None = None

    async def call(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        response_model: type[Any] | None = None,
        reasoning_effort: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, int]:
        self.response_model = response_model
        self.messages = messages
        self.metadata = metadata
        return self.raw_content, 42


def test_findings_schema_is_bare_array() -> None:
    schema = _Findings.model_json_schema()

    assert schema["type"] == "array"
    assert schema["items"]["$ref"] == "#/$defs/_Finding"


def test_validate_visit_parses_empty_array_without_fallback_warning(caplog) -> None:
    client = _FakeClient("[]")

    findings, tokens = asyncio.run(
        validate_visit("prompt", {"visit": "data"}, client=client)  # type: ignore[arg-type]
    )

    assert findings == []
    assert tokens == 42
    assert client.response_model is _Findings
    assert "failed to parse as _Findings" not in caplog.text


def test_validate_visit_parses_bare_findings_array() -> None:
    client = _FakeClient(
        '[{"flag":"MISSING_DIAGNOSIS","issue":"Диагноз не указан","comment":"Пустой список"}]'
    )

    findings, _ = asyncio.run(
        validate_visit("prompt", {"visit": "data"}, client=client)  # type: ignore[arg-type]
    )

    assert findings == [
        {
            "flag": "MISSING_DIAGNOSIS",
            "issue": "Диагноз не указан",
            "comment": "Пустой список",
        }
    ]


def test_validate_visit_keeps_legacy_root_wrapper_fallback() -> None:
    client = _FakeClient(
        '{"root":[{"flag":"MISSING_DIAGNOSIS","issue":"Диагноз не указан"}]}'
    )

    findings, _ = asyncio.run(
        validate_visit("prompt", {"visit": "data"}, client=client)  # type: ignore[arg-type]
    )

    assert findings == [
        {
            "flag": "MISSING_DIAGNOSIS",
            "issue": "Диагноз не указан",
            "comment": "",
        }
    ]


def test_validate_visit_parses_fenced_json_array_without_dropping_findings(caplog) -> None:
    client = _FakeClient(
        """```json
[
  {
    "flag": "НЕСООТВЕТСТВИЕ_МКБ_И_ТЕКСТА_ДИАГНОЗА",
    "issue": "Код МКБ не соответствует клинической картине.",
    "comment": "В записи описана острая инфекция, но указан I10."
  },
  {
    "flag": "НЕПОЛНОЕ_НАЗНАЧЕНИЕ_ПРЕПАРАТА",
    "issue": "Назначение препарата не содержит кратности приема.",
    "comment": "Парацетамол указан без интервала и длительности."
  }
]
```"""
    )

    findings, _ = asyncio.run(
        validate_visit("prompt", {"visit": "data"}, client=client)  # type: ignore[arg-type]
    )

    assert findings == [
        {
            "flag": "НЕСООТВЕТСТВИЕ_МКБ_И_ТЕКСТА_ДИАГНОЗА",
            "issue": "Код МКБ не соответствует клинической картине.",
            "comment": "В записи описана острая инфекция, но указан I10.",
        },
        {
            "flag": "НЕПОЛНОЕ_НАЗНАЧЕНИЕ_ПРЕПАРАТА",
            "issue": "Назначение препарата не содержит кратности приема.",
            "comment": "Парацетамол указан без интервала и длительности.",
        },
    ]
    assert "failed to parse JSON response" not in caplog.text


def _verdict(*, violated=True, issue="Нет диагноза", comment="Диагнозы: []"):
    return json.dumps({"comment": comment, "violated": violated, "issue": issue},
                      ensure_ascii=False)


def _run_rule(client, *, flag_code="TRUSTED_FLAG"):
    return asyncio.run(
        validate_rule(
            "prompt",
            {"Прием": {"GUID": "visit-1"}},
            "rule",
            flag_code=flag_code,
            rule_id="rule-id",
            client=client,  # type: ignore[arg-type]
        )
    )


def test_validate_rule_uses_cacheable_prompt_visit_rule_order() -> None:
    client = _FakeClient(_verdict())
    visit = {"Прием": {"GUID": "visit-1"}, "Диагнозы": []}

    findings, tokens = asyncio.run(
        validate_rule(
            "common prompt",
            visit,
            "(MISSING_DIAGNOSIS) Должен быть диагноз.",
            flag_code="MISSING_DIAGNOSIS",
            rule_id="diagnosis_required",
            client=client,  # type: ignore[arg-type]
        )
    )

    roles = [m["role"] for m in client.messages]
    assert roles == ["system", "user", "user"]
    assert client.messages[0]["content"] == "common prompt"
    assert "Диагнозы" in client.messages[1]["content"]
    assert client.messages[2]["content"].endswith("(MISSING_DIAGNOSIS) Должен быть диагноз.")
    assert findings == [{"flag": "MISSING_DIAGNOSIS", "issue": "Нет диагноза",
                         "comment": "Диагнозы: []"}]
    assert tokens == 42


def test_flag_comes_from_code_not_from_the_model() -> None:
    """Вердикт не содержит флага вовсе: его нечем подменить."""
    findings, _ = _run_rule(_FakeClient(_verdict(issue="Нет обязательного поля",
                                                 comment="Факты")))
    assert findings == [{"flag": "TRUSTED_FLAG", "issue": "Нет обязательного поля",
                         "comment": "Факты"}]


def test_one_rule_yields_at_most_one_flag() -> None:
    """Схема не даёт вернуть несколько замечаний на одно правило.

    Массив findings это позволял, и все замечания получали один и тот же флаг —
    так один дефект попадал в отчёт несколько раз разными формулировками.
    """
    from LLM.validations import _RuleVerdict

    schema = _RuleVerdict.model_json_schema()
    assert schema["properties"]["issue"]["type"] == "string"
    assert "array" not in json.dumps(schema)


def test_applicability_is_not_asked_of_the_model() -> None:
    """Какие правила относятся к приёму, решает get_rules по коду ЕГИСЗ и возрасту.

    Отдельное поле «применимо ли» дублировало этот отбор и давало модели тихий
    выключатель: выбранное кодом правило отменялось ответом, неотличимым в
    отчёте от «нарушения нет».
    """
    from LLM.validations import _RuleVerdict

    assert "condition_met" not in _RuleVerdict.model_json_schema()["properties"]
    prompt = (ROOT / "src" / "LLM" / "prompts" / "formal_structure_validator.txt").read_text(
        encoding="utf-8"
    )
    assert "condition_met" not in prompt


def test_clean_rule_yields_no_flag() -> None:
    findings, _ = _run_rule(_FakeClient(_verdict(violated=False, issue="")))
    assert findings == []


def test_violation_without_wording_is_not_a_finding() -> None:
    """Флаг без текста замечания врачу бесполезен и в отчёт не идёт."""
    findings, _ = _run_rule(_FakeClient(_verdict(issue="   ")))
    assert findings == []


def test_verdict_survives_a_fenced_answer(caplog) -> None:
    """vLLM оборачивает структурный ответ в ``` — разбор это переживает."""
    with caplog.at_level("ERROR"):
        findings, _ = _run_rule(_FakeClient("```json\n" + _verdict() + "\n```"))
    assert findings == [{"flag": "TRUSTED_FLAG", "issue": "Нет диагноза",
                         "comment": "Диагнозы: []"}]
    assert "failed to parse" not in caplog.text


def test_unparsable_verdict_yields_no_flag(caplog) -> None:
    """Пропустить дефект дешевле, чем выдумать его врачу."""
    with caplog.at_level("ERROR"):
        findings, tokens = _run_rule(_FakeClient("не json вовсе"))
    assert findings == []
    assert tokens == 42
    assert "failed to parse rule verdict" in caplog.text


def test_evidence_is_generated_before_the_decisions() -> None:
    """Порядок полей в схеме — часть контракта, а не оформление.

    В json_schema он задаёт порядок генерации: модель сперва выписывает факты
    из карты и только потом принимает решения.
    """
    from LLM.validations import _RuleVerdict

    assert list(_RuleVerdict.model_json_schema()["properties"]) == [
        "comment", "violated", "issue",
    ]


def test_validate_rule_anchors_today_on_the_visit_date() -> None:
    client = _FakeClient('{"comment":"","violated":false,"issue":""}')
    visit = {"Прием": {"DATE": "20.08.2026", "GUID": "g"}, "Диагнозы": []}

    asyncio.run(
        validate_rule(
            "prompt", visit, "rule", flag_code="F", rule_id="r", client=client  # type: ignore[arg-type]
        )
    )

    visit_message = client.messages[1]["content"]
    assert visit_message.startswith("## Сегодняшний день")
    assert "20.08.2026" in visit_message
    # сама запись передаётся следом, без изменений
    assert json.dumps(visit, ensure_ascii=False, indent=2) in visit_message


def test_validate_rule_without_a_visit_date_sends_the_bare_record() -> None:
    client = _FakeClient('{"comment":"","violated":false,"issue":""}')
    visit = {"Прием": {"GUID": "g"}}

    asyncio.run(
        validate_rule(
            "prompt", visit, "rule", flag_code="F", rule_id="r", client=client  # type: ignore[arg-type]
        )
    )

    assert client.messages[1]["content"] == json.dumps(visit, ensure_ascii=False, indent=2)
