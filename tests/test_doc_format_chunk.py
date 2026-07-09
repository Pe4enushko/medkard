"""Регрессия Doc._format_chunk — шапка чанка при чтении полей из guidelines."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from storage.models.doc import Doc


def test_header_from_guideline_fields():
    doc = Doc(
        chunk="тело фрагмента",
        file_id="581_2",
        name="Острый бронхит",
        mkb=["J20.0", "J20.1"],
        age_category=["Взрослые", "дети"],
        metadata={"section": "Диагностика", "content_type": "text", "chunk_index": 3},
    )
    out = doc._format_chunk()
    assert "Острый бронхит | МКБ-10: J20.0, J20.1 | Взрослые, дети" in out
    assert "Диагностика | фрагмент 3" in out
    assert out.endswith("тело фрагмента")


def test_header_omits_absent_fields():
    doc = Doc(chunk="тело", file_id="x", metadata={"section": "S", "content_type": "text"})
    out = doc._format_chunk()
    assert out.endswith("тело")
    assert "МКБ-10" not in out
