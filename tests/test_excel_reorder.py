import openpyxl

from audit.models import FormalStructureResult
from parsers.excel import AuditExcelWriter


def _visit():
    return {
        "Врач": {"SPECIALIZATION": "педиатр"},
        "Прием": {"DATE": "25.06.2026"},
        "ДанныеОсмотра": [
            {"Параметр": "Диагноз", "Значение": "ОРВИ"},
            {"Параметр": "Жалобы на момент осмотра", "Значение": "кашель"},
            {"Параметр": "Анамнез заболевания", "Значение": "3 дня"},
        ],
    }


def _read_inspection_cell(path):
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    # column D (index 4) = "Данные осмотра"
    val = ws.cell(row=2, column=4).value
    wb.close()
    return val


def test_writer_without_order_tokens_keeps_source_order(tmp_path):
    path = tmp_path / "r.xlsx"
    writer = AuditExcelWriter(path)
    writer.append(_visit(), FormalStructureResult(findings=[]), diagnosis=[], icd_check=[])
    text = _read_inspection_cell(path)
    assert text.index("Диагноз") < text.index("Жалобы на момент осмотра")


def test_writer_with_order_tokens_reorders(tmp_path):
    path = tmp_path / "r.xlsx"
    tokens = ["жалобы на момент осмотра", "анамнез заболевания", "диагноз"]
    writer = AuditExcelWriter(path, order_tokens=tokens)
    writer.append(_visit(), FormalStructureResult(findings=[]), diagnosis=[], icd_check=[])
    text = _read_inspection_cell(path)
    assert text.index("Жалобы на момент осмотра") < text.index("Анамнез заболевания") < text.index("Диагноз")
