"""
DiagnosisValidator — diagnosis-check audit for a single ambulatory visit.

Workflow::
    validator = DiagnosisValidator(visit)

    diagnoses  = validator.get_diagnoses()      # list of diagnosis dicts
    findings   = await validator.validate_diagnosis(diagnosis)
    # [{flag, issue}, ...] per diagnosis

``validate_diagnosis`` converts ДанныеОсмотра into readable text, then
invokes the LangChain RAG agent from LLM.rag_agent to reason over the
clinical context and return structured findings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from LLM.rag_agent import create_rag_agent

# ── Prompt ────────────────────────────────────────────────────────────────────
_PROMPT_PATH = Path(__file__).parent.parent.parent / "LLM" / "prompts" / "diagnosis_validator.txt"
_SYSTEM_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")
# ─────────────────────────────────────────────────────────────────────────────


def _parse_inspection_data(raw_visit: dict[str, Any]) -> str:
    """Convert ДанныеОсмотра list into human-readable ``key: value`` text.

    Each element ``{"Параметр": P, "Значение": V}`` becomes a line ``P: V``.
    Empty or missing values are omitted.
    """
    items: list[dict] = raw_visit.get("ДанныеОсмотра", [])
    lines: list[str] = []
    for item in items:
        key: str = str(item.get("Параметр", "")).strip()
        value: str = str(item.get("Значение", "")).strip()
        if key and value:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _format_diagnosis(diagnosis: dict[str, Any]) -> str:
    """Render a single diagnosis record as structured text."""
    code: str = diagnosis.get("КодМКБ", "—")
    name: str = diagnosis.get("НаименованиеМКБ", "—")
    detail: str = diagnosis.get("Детализация", "")
    first: bool | None = diagnosis.get("ВыявленВпервые")

    lines = [
        f"Код МКБ: {code}",
        f"Наименование МКБ: {name}",
    ]
    if detail:
        lines.append(f"Детализация: {detail}")
    if first is not None:
        lines.append(f"Выявлен впервые: {'да' if first else 'нет'}")
    return "\n".join(lines)


class DiagnosisValidator:
    """Validates each diagnosis in a visit against clinical context via RAG agent.

    Args:
        visit: Raw visit dict (as parsed from the source JSON).
    """

    def __init__(self, visit: dict[str, Any]) -> None:
        self._visit = visit

    def get_diagnoses(self) -> list[dict[str, Any]]:
        """Return the list of diagnosis records from the visit (key «Диагнозы»).

        Returns:
            List of diagnosis dicts. Empty list if the key is absent.
        """
        return self._visit.get("Диагнозы", [])

    async def validate_diagnosis(
        self,
        diagnosis: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Validate a single diagnosis against the clinical context of the visit.

        Steps:
            1. Parses ``ДанныеОсмотра`` into readable ``key: value`` text.
            2. Combines the diagnosis record and the inspection text into a
               single prompt message.
            3. Invokes the RAG agent (which can call ``retrieve`` as needed).
            4. Parses the final JSON output and returns the findings list.

        Args:
            diagnosis: A single element from ``get_diagnoses()``.

        Returns:
            List of finding dicts: ``[{"flag": ..., "issue": ...}, ...]``.
            Empty list means no defects were detected for this diagnosis.

        Raises:
            ValueError: If the agent output cannot be parsed as a JSON array.
        """
        inspection_text = _parse_inspection_data(self._visit)
        diagnosis_text = _format_diagnosis(diagnosis)

        human_message = (
            "## Диагноз\n"
            f"{diagnosis_text}\n\n"
            "## Клинический контекст (данные осмотра)\n"
            f"{inspection_text}"
        )

        agent = await create_rag_agent(_SYSTEM_PROMPT)
        result = await agent.ainvoke({"input": human_message})
        output: str = result.get("output", "[]").strip()

        # Strip possible markdown code fences
        if output.startswith("```"):
            output = output.split("```")[1]
            if output.startswith("json"):
                output = output[4:]
            output = output.strip()

        try:
            findings = json.loads(output)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"DiagnosisValidator: agent returned non-JSON output: {output!r}"
            ) from exc

        if not isinstance(findings, list):
            raise ValueError(
                f"DiagnosisValidator: expected a JSON array, got: {type(findings)}"
            )
        return [{"flag": f["flag"], "issue": f["issue"]} for f in findings]
#TODO: develop normal flag list for diagnosis or move from flags on issues array.
