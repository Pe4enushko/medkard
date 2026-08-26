"""Колонка diag_errors не покидает медкард.

Это журнал наших аварий (упавший судья, пустой ответ модели). Он нужен нам для
разбора, но в реплику движка не едет: агент медчека читает строку как есть и
понесёт нашу внутреннюю ошибку врачу как факт о карте.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def test_the_pull_api_never_selects_the_degradation_column():
    assert "diag_errors" not in (SRC / "reporting" / "api_formatter.py").read_text(
        encoding="utf-8"
    )


def test_the_report_renderer_never_touches_it():
    for path in ("parsers/excel.py", "reporting/result_parser.py"):
        assert "diag_errors" not in (SRC / path).read_text(encoding="utf-8"), path
