"""Валидатор диагноза снимает редакцию клинрека вместе с результатом проверки.

Сам file_id живёт до следующей редакции: развернуть его в название задним
числом получается не всегда, поэтому снимок делается там, где Guideline уже
в руках, — в момент аудита.
"""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@dataclass
class _FakeGuideline:
    file_id: str
    name: str | None = None
    published_at: str | None = None
    age_category: list[str] = field(default_factory=list)


class _FakeGuidelinesStorage:
    """Возвращает ту же редакцию, что вернул бы справочник в момент аудита."""

    guideline: _FakeGuideline | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, file_id):
        return self.guideline


class _FakeClinicRecs:
    async def pick_recs(self, patient, diagnosis):
        return "file-1", 0


class _FakeGraph:
    async def ainvoke(self, state):
        return {"issues": {}, "sources": [], "errors": [], "tokens": 0}


def _validator_with_fakes(monkeypatch, guideline):
    import audit.diagnosis.validator as validator_module

    _FakeGuidelinesStorage.guideline = guideline

    storage_module = types.ModuleType("storage.guidelines_storage")
    storage_module.GuidelinesStorage = _FakeGuidelinesStorage
    monkeypatch.setitem(sys.modules, "storage.guidelines_storage", storage_module)

    rag_pkg = types.ModuleType("RAG")
    rag_retrieval = types.ModuleType("RAG.retrieval")
    searches = types.ModuleType("RAG.retrieval.searches")

    async def get_sections_for_file(file_id):
        return []

    searches.get_sections_for_file = get_sections_for_file
    monkeypatch.setitem(sys.modules, "RAG", rag_pkg)
    monkeypatch.setitem(sys.modules, "RAG.retrieval", rag_retrieval)
    monkeypatch.setitem(sys.modules, "RAG.retrieval.searches", searches)
    monkeypatch.setattr(validator_module, "_get_graph", lambda: _FakeGraph())

    validator = validator_module.DiagnosisValidator.__new__(
        validator_module.DiagnosisValidator
    )
    validator._visit = {"Прием": {"GUID": "card-1", "DATE": "22.07.2026"}, "Пациент": {}}
    validator._clinic_recs = _FakeClinicRecs()
    validator._card_guid = "card-1"
    validator._correlation_id = "corr-1"
    return validator


def test_validate_diagnosis_snapshots_the_guideline_edition(monkeypatch) -> None:
    validator = _validator_with_fakes(
        monkeypatch,
        _FakeGuideline(
            file_id="file-1",
            name="Острый синусит",
            published_at="2024",
            age_category=["Взрослые", "Дети"],
        ),
    )

    result, _ = asyncio.run(validator.validate_diagnosis({"КодМКБ": "J01"}))

    assert result.guideline_meta == {
        "name": "Острый синусит",
        "date": "2024",
        "age_group": "Взрослые, Дети",
    }


def test_validate_diagnosis_without_a_guideline_row_keeps_the_snapshot_empty(
    monkeypatch,
) -> None:
    # Справочник о выбранном file_id не знает: выдумывать название нечем, и
    # пустой снимок честнее пустых строк, которые в отчёте выглядят как данные.
    validator = _validator_with_fakes(monkeypatch, None)

    result, _ = asyncio.run(validator.validate_diagnosis({"КодМКБ": "J01"}))

    assert result.guideline_file_id == "file-1"
    assert result.guideline_meta is None
