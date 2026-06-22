"""
Integration tests for LLM.validations.validate_visit — makes real API calls
via the configured OPENAI_BASE_URL endpoint.

Each test supplies a minimal but realistic visit dict and a rendered system
prompt, then asserts structural properties of the response (shape, token count,
flag values) rather than exact LLM text, keeping tests stable across model
updates.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from LLM.client import LLMClient
from LLM.validations import validate_visit


# ── Shared client fixture ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def api_client() -> LLMClient:
    """Build an LLMClient from the loaded env — avoids singleton ordering issues."""
    return LLMClient()


# ── Prompts ───────────────────────────────────────────────────────────────────

# No rules — LLM must always return [].
_PROMPT_NO_RULES = (
    "Ты — медицинский эксперт по качеству амбулаторных записей. "
    "Тебе поступает запись о визите пациента и набор правил оформления.\n\n"
    "## Правила\n\n(правил нет)\n\n"
    "## Инструкции\n"
    "Отвечай строго в формате JSON: массив объектов "
    '{"flag": "<flag_code>", "issue": "<объяснение>"}. '
    "Если дефектов не обнаружено — верни пустой массив []."
)

# One deterministic rule.
_PROMPT_ONE_RULE = (
    "Ты — медицинский эксперт по качеству амбулаторных записей. "
    "Тебе поступает запись о визите пациента и набор правил оформления.\n\n"
    "## Правила\n\n"
    "(MISSING_DIAGNOSIS) В записи должен быть указан диагноз пациента.\n\n"
    "## Инструкции\n"
    "1. Оцени каждое правило.\n"
    "2. Если нарушено — включи флаг с кратким объяснением.\n"
    "3. Если выполнено — не включай.\n"
    "4. Отвечай строго в формате JSON: массив объектов "
    '{"flag": "<flag_code>", "issue": "<объяснение>"}.\n'
    "5. Если дефектов нет — верни []."
)


# ── Visit fixtures ────────────────────────────────────────────────────────────

# Complete valid visit — all sections present, diagnosis included.
_COMPLETE_VISIT = {
    "Прием": {"GUID": "test-guid-001", "DATE": "2024-01-15"},
    "Пациент": {"ФИО": "Иванов Иван Иванович", "AGE": 45, "SEX": "М"},
    "Диагнозы": [{"КодМКБ": "J06.9", "НаименованиеМКБ": "Острая инфекция верхних дыхательных путей"}],
    "Жалобы": "Кашель, насморк, боль в горле в течение 3 дней.",
    "Анамнез": "Заболел 3 дня назад. Ранее ничем серьёзным не болел.",
    "ОбъективныйОсмотр": "Общее состояние удовлетворительное. Зев гиперемирован.",
    "Рекомендации": "Постельный режим, обильное питьё, парацетамол при температуре.",
    "Услуги": [{"Наименование": "Прием врача-терапевта первичный", "Код": "B01.058.001"}],
}

# Visit with no diagnosis — should trigger MISSING_DIAGNOSIS.
_VISIT_NO_DIAGNOSIS = {
    "Прием": {"GUID": "test-guid-002", "DATE": "2024-01-16"},
    "Пациент": {"ФИО": "Петров Пётр Петрович", "AGE": 30, "SEX": "М"},
    "Диагнозы": [],
    "Жалобы": "Головная боль.",
    "Анамнез": "Болит голова уже неделю.",
    "ОбъективныйОсмотр": "АД 120/80. Пульс 72.",
    "Услуги": [{"Наименование": "Прием врача-терапевта повторный", "Код": "B01.058.002"}],
}

# Visit that is structurally present but explicitly has no diagnosis field.
_VISIT_DIAGNOSIS_FIELD_MISSING = {
    "Прием": {"GUID": "test-guid-003", "DATE": "2024-01-17"},
    "Пациент": {"ФИО": "Сидоров Сидор Сидорович", "AGE": 55, "SEX": "М"},
    "Жалобы": "Боль в груди.",
    "Анамнез": "Боль появилась вчера.",
    "ОбъективныйОсмотр": "АД 140/90.",
    "Услуги": [{"Наименование": "Прием врача-кардиолога первичный", "Код": "B01.014.001"}],
}


# ── Return shape ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_returns_tuple_of_list_and_int(api_client):
    findings, tokens = await validate_visit(_PROMPT_NO_RULES, _COMPLETE_VISIT, client=api_client)
    assert isinstance(findings, list)
    assert isinstance(tokens, int)


@pytest.mark.asyncio
async def test_tokens_are_positive(api_client):
    _, tokens = await validate_visit(_PROMPT_NO_RULES, _COMPLETE_VISIT, client=api_client)
    assert tokens > 0


@pytest.mark.asyncio
async def test_each_finding_has_flag_and_issue(api_client):
    findings, _ = await validate_visit(_PROMPT_ONE_RULE, _VISIT_NO_DIAGNOSIS, client=api_client)
    for f in findings:
        assert "flag" in f, f"finding missing 'flag': {f}"
        assert "issue" in f, f"finding missing 'issue': {f}"
        assert isinstance(f["flag"], str) and f["flag"], f"empty flag in: {f}"
        assert isinstance(f["issue"], str) and f["issue"], f"empty issue in: {f}"


# ── No-rules / clean-visit ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_rules_complete_visit_returns_empty(api_client):
    """When no rules are provided the LLM must return []."""
    findings, _ = await validate_visit(_PROMPT_NO_RULES, _COMPLETE_VISIT, client=api_client)
    assert findings == [], f"Expected no findings with no rules, got: {findings}"


# ── Violation detection ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_missing_diagnosis_triggers_flag(api_client):
    """A visit without a diagnosis must trigger MISSING_DIAGNOSIS."""
    findings, _ = await validate_visit(_PROMPT_ONE_RULE, _VISIT_NO_DIAGNOSIS, client=api_client)
    flags = [f["flag"] for f in findings]
    assert "MISSING_DIAGNOSIS" in flags, f"Expected MISSING_DIAGNOSIS in flags, got: {flags}"


@pytest.mark.asyncio
async def test_complete_visit_no_false_positive(api_client):
    """A complete visit must not trigger MISSING_DIAGNOSIS."""
    findings, _ = await validate_visit(_PROMPT_ONE_RULE, _COMPLETE_VISIT, client=api_client)
    flags = [f["flag"] for f in findings]
    assert "MISSING_DIAGNOSIS" not in flags, f"False positive for complete visit: {flags}"


@pytest.mark.asyncio
async def test_visit_without_diagnosis_field_triggers_flag(api_client):
    """A visit with no Диагнозы key at all must trigger MISSING_DIAGNOSIS."""
    findings, _ = await validate_visit(_PROMPT_ONE_RULE, _VISIT_DIAGNOSIS_FIELD_MISSING, client=api_client)
    flags = [f["flag"] for f in findings]
    assert "MISSING_DIAGNOSIS" in flags, f"Expected MISSING_DIAGNOSIS, got: {flags}"


# ── Flag content quality ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_flag_values_non_empty(api_client):
    findings, _ = await validate_visit(_PROMPT_ONE_RULE, _VISIT_NO_DIAGNOSIS, client=api_client)
    for f in findings:
        assert f["flag"].strip(), f"Blank flag in: {f}"


@pytest.mark.asyncio
async def test_issue_text_contains_cyrillic(api_client):
    """Issue explanations must be in Russian (contain Cyrillic)."""
    findings, _ = await validate_visit(_PROMPT_ONE_RULE, _VISIT_NO_DIAGNOSIS, client=api_client)
    for f in findings:
        has_cyrillic = any("Ѐ" <= ch <= "ӿ" for ch in f["issue"])
        assert has_cyrillic, f"Issue text appears non-Russian: {f['issue']!r}"


@pytest.mark.asyncio
async def test_no_invented_flags(api_client):
    """The LLM must not return flag codes not listed in the prompt."""
    findings, _ = await validate_visit(_PROMPT_ONE_RULE, _VISIT_DIAGNOSIS_FIELD_MISSING, client=api_client)
    allowed = {"MISSING_DIAGNOSIS"}
    for f in findings:
        assert f["flag"] in allowed, (
            f"Invented flag {f['flag']!r} not in allowed set {allowed}"
        )
