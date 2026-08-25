from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from LLM.prompt_context import today_block
from parsers.json_parser import visit_date


def test_visit_date_accepts_one_c_and_iso_shapes() -> None:
    assert visit_date({"DATE": "25.06.2026"}) == date(2026, 6, 25)
    assert visit_date({"DATE": "2026-06-25T13:10:00"}) == date(2026, 6, 25)
    assert visit_date({"DATE": "not-a-date"}) is None
    assert visit_date({}) is None


def test_today_block_names_the_visit_day_in_one_c_format() -> None:
    block = today_block(date(2026, 8, 20))

    assert block.startswith("## Сегодняшний день")
    assert "20.08.2026" in block
    # без этого чекеры считают год приёма опечаткой — на выгрузке за 20.08.2026
    # это дало ложные замечания в шести картах
    assert "опечатк" in block


def test_today_block_is_empty_without_a_date() -> None:
    assert today_block(None) == ""
