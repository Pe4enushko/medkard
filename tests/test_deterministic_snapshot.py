"""Снимок детерминированного слоя: профиль карты и сравнение двух снимков."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "deterministic_snapshot", ROOT / "scripts" / "deterministic-snapshot.py"
)
snap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(snap)


def _profile(**over):
    base = {"visit_types": ["PRIMARY"], "age": 44, "icd": ["I10"],
            "rules": ["A", "B"], "nmu_contradiction": False}
    base.update(over)
    return base


def test_diff_reports_gained_and_lost_rules(capsys):
    before = {"card-1": _profile(rules=["A", "B"])}
    after = {"card-1": _profile(rules=["B", "C"])}

    assert snap._diff(before, after) == 1
    out = capsys.readouterr().out
    assert "правил добавилось: ['C']" in out
    assert "правил ушло: ['A']" in out
    assert "2 → 2" in out


def test_identical_snapshots_report_nothing_changed(capsys):
    before = {"card-1": _profile()}
    assert snap._diff(before, dict(before)) == 0
    assert "карт с изменениями: 0 из 1" in capsys.readouterr().out


def test_diff_notices_a_changed_visit_type(capsys):
    before = {"card-1": _profile(visit_types=["OTHER"], rules=[])}
    after = {"card-1": _profile(visit_types=["REPEAT"], rules=["A"])}

    snap._diff(before, after)
    assert "visit_types: ['OTHER'] → ['REPEAT']" in capsys.readouterr().out


def test_diff_reports_cards_present_on_one_side_only(capsys):
    assert snap._diff({"gone": _profile()}, {"new": _profile()}) == 1
    out = capsys.readouterr().out
    assert "пропали из снимка (1): gone" in out
    assert "появились в снимке (1): new" in out


async def test_profile_is_pure_and_needs_no_llm_or_db():
    """Профиль считается чистыми функциями — в этом весь смысл снимка.

    Всё, что здесь есть, решает код: тип визита по коду ЕГИСЗ, возраст, отбор
    правил, сверка кода с наименованием. Сработал ли чекер на выбранном
    правиле — вопрос к модели, и в снимок он не входит.
    """
    from audit.formal_structure.validator import FormalValidator

    visit = {
        "Прием": {"GUID": "abc"},
        "Пациент": {"AGE": 3},
        "Диагнозы": [{"КодМКБ": "J06.9"}],
        "Услуги": [{"Наименование": "Приём педиатра повторный", "Код": "B01.031.002"}],
    }
    profile = await snap._profile(visit, FormalValidator())

    assert profile["visit_types"] == ["REPEAT"]
    assert profile["age"] == 3
    assert profile["icd"] == ["J06.9"]
    assert profile["rules"] == sorted(profile["rules"])
    assert len(profile["rules"]) > 8, "детская карта должна набирать правила, а не 8 как раньше"


def test_cards_are_read_from_a_cases_wrapper(tmp_path):
    """Фикстуры e2e лежат обёрткой {cases: [{visit}]} — снимок это понимает."""
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [{"visit": {"Прием": {"GUID": "g1"}}}]}), encoding="utf-8")
    assert [v["Прием"]["GUID"] for v in snap._cards_from_file(path)] == ["g1"]


def test_a_bare_list_of_visits_is_also_accepted(tmp_path):
    path = tmp_path / "visits.json"
    path.write_text(json.dumps([{"Прием": {"GUID": "g1"}}, {"Прием": {"GUID": "g2"}}]), encoding="utf-8")
    assert len(list(snap._cards_from_file(path))) == 2


def test_unrecognised_shape_is_refused(tmp_path):
    path = tmp_path / "junk.json"
    path.write_text('{"что-то": "другое"}', encoding="utf-8")
    with pytest.raises(SystemExit):
        snap._cards_from_file(path)


def test_snapshot_records_what_took_it() -> None:
    """Снимок обязан свидетельствовать о себе сам.

    Иначе сравнение двух файлов доказывает не то, что кажется: два снимка
    разошлись по возрасту на 65 картах, и вопрос «на какой ревизии снят „до“»
    оказалось нечем закрыть, кроме догадок.
    """
    meta = asyncio.run(snap._snapshot([]))["_meta"]

    assert meta["age_reader"].endswith("patient_age")
    assert meta["get_rules_arity"] >= 2
    assert meta["revision"]
    assert meta["src"].endswith("src")


def test_diff_shouts_when_both_sides_are_the_same_revision(capsys) -> None:
    """Одна ревизия по обе стороны — сравнивать нечего, и молчать об этом нельзя."""
    meta = {"revision": "abc1234", "age_reader": "x.y", "get_rules_arity": 2, "src": "/s"}
    same = {"_meta": meta, "card": {"visit_types": ["PRIMARY"], "age": 40,
                                    "icd": [], "rules": [], "nmu_contradiction": False}}

    snap._diff(dict(same), dict(same))

    assert "сравнивать нечего" in capsys.readouterr().out


def test_diff_says_so_when_provenance_is_missing(capsys) -> None:
    snap._diff({}, {})

    assert capsys.readouterr().out.count("происхождение не записано") == 2
