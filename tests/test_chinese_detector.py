"""
Integration tests for ChineseDetector — makes real LLM calls via the
configured OPENAI_BASE_URL endpoint.
"""

import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from LLM.chinese_detector import ChineseDetector


@pytest.fixture
def detector():
    return ChineseDetector()


# ── check_str ─────────────────────────────────────────────────────────────────

def test_check_str_clean_russian(detector):
    assert not detector.check_str("Нарушений не выявлено")

def test_check_str_clean_latin(detector):
    assert not detector.check_str("No issues found")

def test_check_str_empty(detector):
    assert not detector.check_str("")

def test_check_str_detects_cjk_mixed(detector):
    assert detector.check_str("Нет 违规 нарушений")

def test_check_str_detects_cjk_only(detector):
    assert detector.check_str("中文")


# ── repair_issue — real API calls ─────────────────────────────────────────────

async def test_repair_issue_removes_chinese(detector):
    bad_issue = "Отсутствует 病史 в карте пациента"
    repaired, tokens = await detector.repair_issue(bad_issue)

    assert not detector.check_str(repaired), f"CJK still present after repair: {repaired!r}"
    assert len(repaired) > 0
    assert tokens > 0


async def test_repair_issue_preserves_russian_context(detector):
    bad_issue = "Не заполнен раздел 诊断: диагноз отсутствует"
    repaired, tokens = await detector.repair_issue(bad_issue)

    assert not detector.check_str(repaired)
    assert "диагноз" in repaired.lower() or "раздел" in repaired.lower()
    assert tokens > 0


async def test_repair_issue_all_chinese(detector):
    bad_issue = "中文文本здесь"
    repaired, tokens = await detector.repair_issue(bad_issue)

    assert not detector.check_str(repaired)
    assert tokens > 0
