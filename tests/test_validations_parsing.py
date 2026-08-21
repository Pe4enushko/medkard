from __future__ import annotations

import sys
import asyncio
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from LLM.validations import _Findings, _RuleVerdict, validate_rule, validate_visit


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


def test_validate_rule_uses_cacheable_prompt_visit_rule_order() -> None:
    client = _FakeClient(
        '{"condition_met":true,"violated":true,"issue":"Нет диагноза","comment":"Диагнозы: []"}'
    )
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

    assert [message["content"] for message in client.messages] == [
        "common prompt",
        '{\n  "Прием": {\n    "GUID": "visit-1"\n  },\n  "Диагнозы": []\n}',
        "## Единственное проверяемое правило\n\n(MISSING_DIAGNOSIS) Должен быть диагноз.",
    ]
    assert client.response_model is _RuleVerdict
    assert client.metadata == {
        "card_guid": "visit-1",
        "checker": "formal",
        "rule_id": "diagnosis_required",
        "flag_code": "MISSING_DIAGNOSIS",
    }
    assert findings == [
        {
            "flag": "MISSING_DIAGNOSIS",
            "issue": "Нет диагноза",
            "comment": "Диагнозы: []",
        }
    ]
    assert tokens == 42


def test_validate_rule_attaches_trusted_flag_and_drops_clean_verdict() -> None:
    client = _FakeClient(
        '{"condition_met":true,"violated":false,"issue":"","comment":""}'
    )

    findings, _ = asyncio.run(
        validate_rule(
            "prompt",
            {"Прием": {}},
            "rule",
            flag_code="TRUSTED_FLAG",
            rule_id="rule-id",
            client=client,  # type: ignore[arg-type]
        )
    )

    assert findings == []


def test_validate_rule_drops_violation_when_rule_condition_is_not_met() -> None:
    client = _FakeClient(
        '{"condition_met":false,"violated":true,'
        '"issue":"Модель описала другой дефект","comment":""}'
    )

    findings, _ = asyncio.run(
        validate_rule(
            "prompt",
            {"Прием": {}},
            "rule",
            flag_code="TRUSTED_FLAG",
            rule_id="rule-id",
            client=client,  # type: ignore[arg-type]
        )
    )

    assert findings == []


def test_validate_rule_drops_self_contradictory_violation() -> None:
    client = _FakeClient(
        '{"comment":"Все препараты указаны по МНН.","condition_met":true,'
        '"violated":true,"issue":"Нарушение данного конкретного правила отсутствует."}'
    )

    findings, _ = asyncio.run(
        validate_rule(
            "prompt",
            {"Прием": {}},
            "rule",
            flag_code="TRUSTED_FLAG",
            rule_id="rule-id",
            client=client,  # type: ignore[arg-type]
        )
    )

    assert findings == []


def test_validate_rule_keeps_real_missing_field_wording() -> None:
    client = _FakeClient(
        '{"comment":"Длительность не указана.","condition_met":true,'
        '"violated":true,"issue":"Нарушение правила: отсутствует длительность лечения."}'
    )

    findings, _ = asyncio.run(
        validate_rule(
            "prompt",
            {"Прием": {}},
            "rule",
            flag_code="TRUSTED_FLAG",
            rule_id="rule-id",
            client=client,  # type: ignore[arg-type]
        )
    )

    assert [finding["flag"] for finding in findings] == ["TRUSTED_FLAG"]
