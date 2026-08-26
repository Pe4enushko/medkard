from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _load_modules(monkeypatch):
    storage = types.ModuleType("storage")
    storage.__path__ = [str(SRC / "storage")]
    monkeypatch.setitem(sys.modules, "storage", storage)

    storage_models = types.ModuleType("storage.models")
    storage_models.__path__ = [str(SRC / "storage" / "models")]
    monkeypatch.setitem(sys.modules, "storage.models", storage_models)

    result_path = SRC / "storage" / "models" / "result.py"
    result_spec = importlib.util.spec_from_file_location(
        "storage.models.result", result_path
    )
    assert result_spec and result_spec.loader
    result = importlib.util.module_from_spec(result_spec)
    monkeypatch.setitem(sys.modules, "storage.models.result", result)
    result_spec.loader.exec_module(result)

    base = types.ModuleType("storage.base")
    base.BaseStorage = type("BaseStorage", (), {})

    # done_cards_storage импортирует его для ретрая апсерта; здесь до пула дело
    # не доходит, но без имени модуль не импортируется вовсе.
    async def _reopen_shared_pool():
        raise AssertionError("тест не должен ходить в БД")

    base.reopen_shared_pool = _reopen_shared_pool
    monkeypatch.setitem(sys.modules, "storage.base", base)

    done_path = SRC / "storage" / "done_cards_storage.py"
    done_spec = importlib.util.spec_from_file_location(
        "storage.done_cards_storage", done_path
    )
    assert done_spec and done_spec.loader
    done = importlib.util.module_from_spec(done_spec)
    monkeypatch.setitem(sys.modules, "storage.done_cards_storage", done)
    done_spec.loader.exec_module(done)

    parser_path = SRC / "reporting" / "result_parser.py"
    parser_spec = importlib.util.spec_from_file_location(
        "diagnosis_graph_result_parser", parser_path
    )
    assert parser_spec and parser_spec.loader
    parser = importlib.util.module_from_spec(parser_spec)
    parser_spec.loader.exec_module(parser)
    return result, done, parser


def test_diag_json_separates_issue_and_guideline_sources(monkeypatch) -> None:
    model, done, _ = _load_modules(monkeypatch)
    diagnosis = model.DiagnosisResult(
        icd_code="J01",
        guideline_file_id="file-1",
        issues=[
            model.DiagnosisIssue(
                issue="Замечание",
                aspect="inspection",
                sources=[
                    model.IssueSource(
                        doc_title="КР",
                        section="2 Диагностика",
                        cite="Фрагмент",
                        chunk_id="chunk-1",
                        chunk_index=10,
                    )
                ],
            )
        ],
        guideline_sources=[
            model.GuidelineSource(
                file_id="file-1",
                doc_title="КР",
                sections=[
                    model.GuidelineSourceSection(
                        section="2 Диагностика",
                        chunk_indices=[10, 11],
                        cited=True,
                    )
                ],
            )
        ],
        errors=["judge_treatment: timeout"],
    )

    payload = json.loads(done._diag_json([diagnosis]))[0]

    assert payload["issues"][0]["aspect"] == "inspection"
    assert payload["issues"][0]["sources"][0]["chunk_id"] == "chunk-1"
    assert payload["guideline_sources"][0]["sections"][0]["chunk_indices"] == [10, 11]
    assert payload["errors"] == ["judge_treatment: timeout"]
    assert "sources" not in payload


def test_parse_diagnosis_accepts_legacy_and_new_contracts(monkeypatch) -> None:
    _, _, parser = _load_modules(monkeypatch)
    legacy = parser.parse_diagnosis(
        [
            {
                "icd_code": "J01",
                "guideline_file_id": "old",
                "issues": [{"issue": "old", "sources": []}],
            }
        ]
    )[0]
    current = parser.parse_diagnosis(
        [
            {
                "icd_code": "J02",
                "guideline_file_id": "new",
                "issues": [
                    {
                        "issue": "new",
                        "aspect": "criteria",
                        "sources": [
                            {"doc_title": "КР", "chunk_id": "c1", "chunk_index": 4}
                        ],
                    }
                ],
                "guideline_sources": [
                    {
                        "file_id": "new",
                        "doc_title": "КР",
                        "sections": [
                            {"section": "Критерии", "chunk_indices": [4], "cited": True}
                        ],
                    }
                ],
                "errors": ["partial"],
            }
        ]
    )[0]

    assert legacy.guideline_sources == []
    assert legacy.errors == []
    assert legacy.issues[0].aspect is None
    assert current.issues[0].aspect == "criteria"
    assert current.issues[0].sources[0].chunk_id == "c1"
    assert current.guideline_sources[0].sections[0].cited is True
    assert current.errors == ["partial"]


def test_diag_json_omits_aspect_for_diagnosis_filter_marker(monkeypatch) -> None:
    model, done, _ = _load_modules(monkeypatch)
    diagnosis = model.DiagnosisResult(
        icd_code="Z00",
        issues=[model.DiagnosisIssue(issue="Диагноз пропущен фильтром МКБ")],
    )

    issue = json.loads(done._diag_json([diagnosis]))[0]["issues"][0]

    assert issue["issue"] == "Диагноз пропущен фильтром МКБ"
    assert "aspect" not in issue


def test_diag_json_stores_the_guideline_snapshot(monkeypatch) -> None:
    # Редакции клинреков меняются, и file_id пропадает из манифеста: развернуть
    # ссылку задним числом можно не всегда, поэтому снимок пишется вместе с картой.
    model, done, _ = _load_modules(monkeypatch)
    diagnosis = model.DiagnosisResult(
        icd_code="J01",
        guideline_file_id="file-1",
        guideline_meta={"name": "Острый синусит", "date": "2024", "age_group": "Взрослые"},
    )

    payload = json.loads(done._diag_json([diagnosis]))[0]

    assert payload["guideline_meta"] == {
        "name": "Острый синусит",
        "date": "2024",
        "age_group": "Взрослые",
    }


def test_diag_json_omits_the_snapshot_when_there_is_no_guideline(monkeypatch) -> None:
    # Ключ со значением null у карт без клинрека — лишнее расхождение со старым
    # JSON: у таких карт разворачивать нечего.
    model, done, _ = _load_modules(monkeypatch)
    diagnosis = model.DiagnosisResult(icd_code="Z00")

    payload = json.loads(done._diag_json([diagnosis]))[0]

    assert "guideline_meta" not in payload


def test_parse_diagnosis_prefers_the_stored_snapshot(monkeypatch) -> None:
    # Карту проверяли против той редакции, что записана в строке; свежий манифест
    # рассказывает про другую.
    _, _, parser = _load_modules(monkeypatch)
    entry = {
        "icd_code": "J01",
        "guideline_file_id": "file-1",
        "issues": [],
        "guideline_meta": {"name": "Редакция 2024", "date": "2024", "age_group": "Взрослые"},
    }

    parsed = parser.parse_diagnosis(
        [entry], {"file-1": {"name": "Редакция 2026", "date": "2026", "age_group": "Дети"}}
    )[0]

    assert parsed.guideline_meta["name"] == "Редакция 2024"


def test_parse_diagnosis_falls_back_to_the_manifest_without_a_snapshot(monkeypatch) -> None:
    # Карты до бэкфилла снимка не имеют, и манифест остаётся единственным источником.
    _, _, parser = _load_modules(monkeypatch)
    entry = {"icd_code": "J01", "guideline_file_id": "file-1", "issues": []}

    parsed = parser.parse_diagnosis(
        [entry], {"file-1": {"name": "Редакция 2026", "date": "2026", "age_group": "Дети"}}
    )[0]

    assert parsed.guideline_meta["name"] == "Редакция 2026"
