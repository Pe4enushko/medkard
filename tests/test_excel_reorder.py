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


from audit.excel_formatter import ExcelFormatter


def test_formatter_forwards_order_tokens_to_writer(tmp_path):
    tokens = ["диагноз"]
    fmt = ExcelFormatter(tmp_path / "r.xlsx", order_tokens=tokens)
    assert fmt._excel._order_tokens == tokens


# ── дозаполнение пустых полей шаблона ────────────────────────────────────────

from parsers.inspection_fill import PLACEHOLDER
from parsers.inspection_order import load_inspection_formats


def _alenka_card():
    """Базовый осмотр Алёнки без «Анамнеза заболевания» — так его и присылает
    1С, когда врач поле не заполнил: ключа в записи нет вовсе."""
    labels = (
        "Температура", "ЧСС", "ЧД", "Состояние", "Сознание", "Ф20", "Кожные покровы",
        "Видимые слизистые", "Слизистые ротоглотки", "Миндалины", "Неврологический статус",
        "Опорно-двигательная система", "Сердечно-сосудистая система",
        "Органы брюшной полости", "Стул", "Мочеиспускание",
    )
    return {
        "Врач": {"SPECIALIZATION": "педиатр"},
        "Прием": {"DATE": "25.06.2026"},
        "ДанныеОсмотра": [{"Значение": "норма", "Параметр": label} for label in labels],
    }


def test_writer_draws_missing_template_fields(tmp_path):
    path = tmp_path / "r.xlsx"
    writer = AuditExcelWriter(path, formats=load_inspection_formats("Alenka"))
    writer.append(_alenka_card(), FormalStructureResult(findings=[]), diagnosis=[], icd_check=[])
    text = _read_inspection_cell(path)

    assert f"Значение: {PLACEHOLDER}\n  Параметр: Анамнез заболевания" in text
    # дорисованные поля стоят на своих местах шаблона, а не свалены в хвост
    assert text.index("Анамнез заболевания") < text.index("Температура")


def test_writer_leaves_a_foreign_template_alone(tmp_path):
    """Туберкулинодиагностика — 16 карт из одного поля, чужой шаблон.
    Дорисовывать его базовым набором значило бы выдать два десятка пустых строк."""
    path = tmp_path / "r.xlsx"
    visit = {
        "Врач": {"SPECIALIZATION": "педиатр"},
        "Прием": {"DATE": "25.06.2026"},
        "ДанныеОсмотра": [{"Значение": "отрицательная", "Параметр": "Комментарий к вакцинации"}],
    }
    writer = AuditExcelWriter(path, formats=load_inspection_formats("Alenka"))
    writer.append(visit, FormalStructureResult(findings=[]), diagnosis=[], icd_check=[])
    text = _read_inspection_cell(path)

    assert PLACEHOLDER not in text
    assert "Температура" not in text


def test_order_tokens_still_apply_to_an_unrecognised_record(tmp_path):
    """Опознать шаблон нечем — остаётся прежнее поведение: упорядочить тем
    порядком, который задал оператор, и ничего не дорисовывать."""
    path = tmp_path / "r.xlsx"
    writer = AuditExcelWriter(
        path,
        order_tokens=["жалобы на момент осмотра", "анамнез заболевания", "диагноз"],
        formats=load_inspection_formats("Alenka"),
    )
    writer.append(_visit(), FormalStructureResult(findings=[]), diagnosis=[], icd_check=[])
    text = _read_inspection_cell(path)

    assert PLACEHOLDER not in text
    assert text.index("Жалобы на момент осмотра") < text.index("Анамнез заболевания") < text.index("Диагноз")


def test_formatter_forwards_formats_to_writer(tmp_path):
    formats = load_inspection_formats("Alenka")
    fmt = ExcelFormatter(tmp_path / "r.xlsx", formats=formats)
    assert fmt._excel._formats == formats
