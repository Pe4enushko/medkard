from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _load(monkeypatch):
    """Модель и рендерер отчёта, загруженные без реальных openpyxl и storage."""
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
    model = importlib.util.module_from_spec(result_spec)
    monkeypatch.setitem(sys.modules, "storage.models.result", model)
    result_spec.loader.exec_module(model)

    audit_models = types.ModuleType("audit.models")
    audit_models.FormalStructureResult = model.FormalStructureResult
    monkeypatch.setitem(sys.modules, "audit.models", audit_models)

    openpyxl = types.ModuleType("openpyxl")
    openpyxl.Workbook = type("Workbook", (), {})
    openpyxl.load_workbook = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "openpyxl", openpyxl)
    styles = types.ModuleType("openpyxl.styles")
    styles.Alignment = type("Alignment", (), {})
    monkeypatch.setitem(sys.modules, "openpyxl.styles", styles)
    worksheet_package = types.ModuleType("openpyxl.worksheet")
    worksheet_package.__path__ = []
    monkeypatch.setitem(sys.modules, "openpyxl.worksheet", worksheet_package)
    worksheet = types.ModuleType("openpyxl.worksheet.worksheet")
    worksheet.Worksheet = type("Worksheet", (), {})
    monkeypatch.setitem(sys.modules, "openpyxl.worksheet.worksheet", worksheet)

    excel_path = SRC / "parsers" / "excel.py"
    excel_spec = importlib.util.spec_from_file_location(
        "diagnosis_graph_excel", excel_path
    )
    assert excel_spec and excel_spec.loader
    excel = importlib.util.module_from_spec(excel_spec)
    excel_spec.loader.exec_module(excel)
    return model, excel


def test_excel_adds_guideline_sources_before_icd_check(monkeypatch) -> None:
    model, excel = _load(monkeypatch)

    diagnosis = [
        model.DiagnosisResult(
            icd_code="J01",
            guideline_file_id="file-1",
            guideline_sources=[
                model.GuidelineSource(
                    file_id="file-1",
                    doc_title="КР по синуситу",
                    sections=[
                        model.GuidelineSourceSection(
                            section="2 Диагностика",
                            chunk_indices=[10],
                            cited=True,
                        )
                    ],
                )
            ],
        )
    ]

    row = excel._build_row({}, model.FormalStructureResult(), diagnosis, icd_check=[])

    assert excel._HEADERS[8:10] == ["Источники КР", "Проверка кодирования МКБ"]
    assert row[8] == "[J01] КР по синуситу: 2 Диагностика (цит.)"
    assert len(row) == len(excel._HEADERS) == 10


def test_excel_keeps_our_degradation_out_of_the_report(monkeypatch) -> None:
    """Упавший судья — наша внутренняя авария, а не факт о карте.

    Врач читает отчёт как заключение по приёму: строка про json_invalid ему
    ничего не говорит и выглядит как дефект карты. В строке done_cards errors
    остаются — по ним мы и разбираемся, что не отработало.
    """
    model, excel = _load(monkeypatch)
    diagnosis = [
        model.DiagnosisResult(
            icd_code="J01",
            guideline_file_id="file-1",
            errors=["judge_criteria: 1 validation error for JudgeOutput"],
        )
    ]

    row = excel._build_row({}, model.FormalStructureResult(), diagnosis, icd_check=[])

    assert "ДЕГРАДАЦИЯ" not in row[7]
    assert "JudgeOutput" not in row[7]
    assert diagnosis[0].errors == ["judge_criteria: 1 validation error for JudgeOutput"]
