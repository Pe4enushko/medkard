from __future__ import annotations

import sys
import asyncio
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from LLM.validations import _Findings, validate_visit


class _FakeClient:
    def __init__(self, raw_content: str) -> None:
        self.raw_content = raw_content
        self.response_model: type[Any] | None = None

    async def call(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        response_model: type[Any] | None = None,
    ) -> tuple[str, int]:
        self.response_model = response_model
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
