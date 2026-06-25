from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def _load_validator_module(monkeypatch):
    fake_chinese = types.ModuleType("LLM.chinese_detector")

    class ChineseDetector:
        def check_str(self, text: str) -> bool:
            return False

    fake_chinese.ChineseDetector = ChineseDetector
    monkeypatch.setitem(sys.modules, "LLM.chinese_detector", fake_chinese)

    fake_tools = types.ModuleType("LLM.tools")
    fake_tools.get_anamnesis_tools_for = lambda file_id: []
    fake_tools.get_inspection_tools_for = lambda file_id: []
    fake_tools.get_treatment_tools_for = lambda file_id: []
    monkeypatch.setitem(sys.modules, "LLM.tools", fake_tools)

    fake_diagnosis_pkg = types.ModuleType("audit.diagnosis")
    fake_diagnosis_pkg.__path__ = [str(SRC / "audit" / "diagnosis")]
    monkeypatch.setitem(sys.modules, "audit.diagnosis", fake_diagnosis_pkg)

    fake_clinic_recs = types.ModuleType("audit.diagnosis.clinic_recs")

    class ClinicRecs:
        pass

    fake_clinic_recs.ClinicRecs = ClinicRecs
    monkeypatch.setitem(sys.modules, "audit.diagnosis.clinic_recs", fake_clinic_recs)

    fake_storage_pkg = types.ModuleType("storage")
    fake_storage_pkg.__path__ = [str(SRC / "storage")]
    monkeypatch.setitem(sys.modules, "storage", fake_storage_pkg)

    fake_storage_models_pkg = types.ModuleType("storage.models")
    fake_storage_models_pkg.__path__ = [str(SRC / "storage" / "models")]
    monkeypatch.setitem(sys.modules, "storage.models", fake_storage_models_pkg)

    module_path = SRC / "audit" / "diagnosis" / "validator.py"
    spec = importlib.util.spec_from_file_location("audit.diagnosis.validator", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "audit.diagnosis.validator", module)
    spec.loader.exec_module(module)
    return module


def test_format_visit_context_includes_card_sections(monkeypatch) -> None:
    validator = _load_validator_module(monkeypatch)

    context = validator._format_visit_context(
        {
            "Пациент": {"ФИО": "Пациент"},
            "Диагнозы": [{"КодМКБ": "J06.9"}],
            "Секции": {
                "Жалобы и анамнез": "Кашель и насморк 3 дня.",
                "Осмотр": "Зев гиперемирован.",
                "Назначения": "Парацетамол 500 мг при температуре.",
            },
            "Рекомендации": "Обильное питье.",
            "ДанныеОсмотра": [{"Параметр": "АД", "Значение": "120/80"}],
        }
    )

    assert "## Секции" in context
    assert "Назначения:" in context
    assert "Парацетамол 500 мг" in context
    assert "## Рекомендации" in context
    assert "Обильное питье" in context
    assert "## ДанныеОсмотра" in context
    assert "АД: 120/80" in context
    assert "Пациент" not in context
    assert "Диагнозы" not in context


def test_format_visit_context_marks_missing_context(monkeypatch) -> None:
    validator = _load_validator_module(monkeypatch)

    context = validator._format_visit_context(
        {"Пациент": {"ФИО": "Пациент"}, "Диагнозы": [{"КодМКБ": "J06.9"}]}
    )

    assert context == "—"


def test_parse_issues_accepts_prose_with_fenced_checker_output(monkeypatch) -> None:
    validator = _load_validator_module(monkeypatch)

    issues = validator._parse_issues(
        """
        По результатам поиска найдено несоответствие.

        ```json
        {
          "issues": [
            {
              "issue": "Не указана длительность приема препарата.",
              "sources": [
                {
                  "doc_title": "Клинические рекомендации",
                  "section": "Лечение",
                  "cite": "рекомендуется указать режим дозирования"
                }
              ]
            }
          ]
        }
        ```
        """
    )

    assert len(issues) == 1
    assert issues[0].issue == "Не указана длительность приема препарата."
    assert issues[0].sources[0].section == "Лечение"


def test_parse_issues_accepts_legacy_bare_array(monkeypatch) -> None:
    validator = _load_validator_module(monkeypatch)

    issues = validator._parse_issues(
        '[{"issue":"Нет жалоб в записи.","sources":[{"doc_title":"КР"}]}]'
    )

    assert len(issues) == 1
    assert issues[0].issue == "Нет жалоб в записи."
    assert issues[0].sources[0].doc_title == "КР"
