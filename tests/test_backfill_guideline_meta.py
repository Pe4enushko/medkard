"""Unit tests for scripts/hacks/backfill-guideline-meta.py — the plan, not the DB.

Снимок редакции у карт, проаудированных до того, как его начали писать
(src/audit/diagnosis/validator.py). Что снимок побеждает манифест на чтении,
проверяет tests/test_diagnosis_graph_result_contract.py.
"""

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "backfill_guideline_meta",
    Path(__file__).resolve().parent.parent / "scripts" / "hacks" / "backfill-guideline-meta.py")
backfill = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = backfill
_spec.loader.exec_module(backfill)

MANIFEST = {
    "file-1": {"name": "Острый синусит", "date": "2024", "age_group": "Взрослые"},
}


def test_known_file_id_gets_its_snapshot():
    diag = [{"icd_code": "J01", "guideline_file_id": "file-1", "issues": []}]

    expanded, counts = backfill.expand(diag, MANIFEST)

    assert expanded[0]["guideline_meta"] == MANIFEST["file-1"]
    assert counts["expanded"] == 1


def test_the_rest_of_the_entry_survives():
    # Скрипт переписывает строку целиком, и всё, что аудит уже записал, обязано
    # доехать без изменений.
    diag = [{
        "icd_code": "J01",
        "guideline_file_id": "file-1",
        "issues": [{"issue": "Замечание", "aspect": "criteria", "sources": []}],
        "guideline_sources": [{"file_id": "file-1", "doc_title": "КР", "sections": []}],
        "errors": ["partial"],
    }]

    expanded, _ = backfill.expand(diag, MANIFEST)

    assert expanded[0]["issues"] == diag[0]["issues"]
    assert expanded[0]["guideline_sources"] == diag[0]["guideline_sources"]
    assert expanded[0]["errors"] == ["partial"]


def test_erased_file_id_is_skipped_not_invented():
    # Редакция сменилась, и file_id из манифеста пропал: назвать клинрек нечем,
    # а пустые строки в отчёте выглядят как данные.
    diag = [{"icd_code": "J01", "guideline_file_id": "gone", "issues": []}]

    expanded, counts = backfill.expand(diag, MANIFEST)

    assert expanded is None
    assert counts["missing"] == 1


def test_card_with_a_snapshot_is_left_alone():
    diag = [{
        "icd_code": "J01",
        "guideline_file_id": "file-1",
        "guideline_meta": {"name": "Редакция 2020", "date": "2020", "age_group": "Дети"},
    }]

    expanded, counts = backfill.expand(diag, MANIFEST)

    assert expanded is None
    assert counts["already"] == 1


def test_diagnosis_without_a_guideline_is_not_a_gap():
    # Клинрека для кода не нашлось ещё при аудите — разворачивать нечего.
    diag = [{"icd_code": "Z00", "guideline_file_id": None, "issues": []}]

    expanded, counts = backfill.expand(diag, MANIFEST)

    assert expanded is None
    assert counts["no_guideline"] == 1


def test_one_expanded_entry_carries_the_erased_one_along():
    # В карте два диагноза, и развернуть удалось только один: строку всё равно
    # пишем, второй остаётся без снимка.
    diag = [
        {"icd_code": "J01", "guideline_file_id": "file-1", "issues": []},
        {"icd_code": "J02", "guideline_file_id": "gone", "issues": []},
    ]

    expanded, counts = backfill.expand(diag, MANIFEST)

    assert expanded[0]["guideline_meta"] == MANIFEST["file-1"]
    assert "guideline_meta" not in expanded[1]
    assert counts["expanded"] == 1
    assert counts["missing"] == 1


def test_empty_diag_result_changes_nothing():
    assert backfill.expand([], MANIFEST)[0] is None
