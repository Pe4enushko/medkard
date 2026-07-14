"""Юнит-тест reporting.result_parser.build_manifest_meta (из Guideline)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reporting.result_parser import build_manifest_meta
from storage.models.guideline import Guideline


def test_build_manifest_meta_shape():
    guidelines = [
        Guideline(file_id="581_2", name="Острый бронхит",
                  age_category=["Взрослые", "Дети"], published_at="01.01.2020"),
    ]
    meta = build_manifest_meta(guidelines)
    assert meta["581_2"] == {
        "name": "Острый бронхит",
        "date": "01.01.2020",
        "age_group": "Взрослые, Дети",
    }
