from __future__ import annotations

import importlib.util
import sys
import types
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def _load_validator_module(monkeypatch):
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
    spec = importlib.util.spec_from_file_location(
        "audit.diagnosis.validator", module_path
    )
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


def test_visit_date_accepts_one_c_and_iso_shapes(monkeypatch) -> None:
    validator = _load_validator_module(monkeypatch)

    assert validator._visit_date("25.06.2026").isoformat() == "2026-06-25"
    assert validator._visit_date("2026-06-25T13:10:00").isoformat() == "2026-06-25"
    assert validator._visit_date("not-a-date") is None


async def test_validate_diagnosis_maps_graph_contract(monkeypatch) -> None:
    validator = _load_validator_module(monkeypatch)

    class ClinicRecs:
        async def pick_recs(self, patient, diagnosis):
            return "file-1", 3

    class GuidelinesStorage:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, file_id):
            return types.SimpleNamespace(name="КР по синуситу")

    guidelines_module = types.ModuleType("storage.guidelines_storage")
    guidelines_module.GuidelinesStorage = GuidelinesStorage
    monkeypatch.setitem(sys.modules, "storage.guidelines_storage", guidelines_module)

    searches_module = types.ModuleType("RAG.retrieval.searches")

    async def get_sections_for_file(file_id):
        return ["2 Диагностика"]

    searches_module.get_sections_for_file = get_sections_for_file
    monkeypatch.setitem(sys.modules, "RAG.retrieval.searches", searches_module)

    class Graph:
        async def ainvoke(self, state):
            assert state["visit_date"].isoformat() == "2026-06-25"
            assert state["doc_title"] == "КР по синуситу"
            uuid.UUID(state["correlation_id"])
            return {
                "issues": {
                    "inspection": [
                        {
                            "aspect": "inspection",
                            "issue": "Не выполнено исследование",
                            "sources": [
                                {
                                    "doc_title": "КР по синуситу",
                                    "section": "2 Диагностика",
                                    "cite": "Показано исследование",
                                    "chunk_id": "chunk-1",
                                    "chunk_index": 4,
                                }
                            ],
                        }
                    ]
                },
                "sources": [
                    {
                        "file_id": "file-1",
                        "doc_title": "КР по синуситу",
                        "sections": [
                            {
                                "section": "2 Диагностика",
                                "chunk_indices": [4],
                                "cited": True,
                            }
                        ],
                    }
                ],
                "errors": ["judge_treatment: timeout"],
                "tokens": 17,
            }

    monkeypatch.setattr(validator, "_get_graph", lambda: Graph())
    instance = validator.DiagnosisValidator(
        {
            "Пациент": {"Возраст": 10},
            "Прием": {"DATE": "25.06.2026", "GUID": "card-1"},
        }
    )
    instance._clinic_recs = ClinicRecs()

    result, tokens = await instance.validate_diagnosis(
        {"КодМКБ": "J01", "НаименованиеМКБ": "Синусит"}
    )

    assert tokens == 20
    assert result.inspection_issues[0].aspect == "inspection"
    assert result.inspection_issues[0].sources[0].chunk_id == "chunk-1"
    assert result.guideline_sources[0].sections[0].cited is True
    assert result.errors == ["judge_treatment: timeout"]
