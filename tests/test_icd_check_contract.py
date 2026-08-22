"""ICD-чекер: два этапа с раздельным контекстом, без ReAct-цикла."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.icd_check import validator as icd
from storage.models.guideline import Guideline


def _guideline(file_id: str, name: str, codes: list[str]) -> Guideline:
    return Guideline(file_id=file_id, name=name, mkb=codes, age_category=["Взрослые"])


MIGRAINE = _guideline("mig_1", "Мигрень", ["G43.0", "G43.1"])


def _manifest(n: int = 200) -> list[Guideline]:
    """Перечень больше снятого лимита 120 — на нём и врал прежний отбор."""
    rows = [_guideline(f"{i}_1", f"Рекомендация {i}", [f"I{i % 90:02d}.0"]) for i in range(n)]
    rows.append(MIGRAINE)
    return rows


class _ScriptedClient:
    """Отдаёт заготовленные ответы по порядку и запоминает, что видел."""

    def __init__(self, *answers: object) -> None:
        self.answers = list(answers)
        self.seen: list[str] = []
        self.systems: list[str] = []

    async def call(self, messages, *, temperature, response_model=None, metadata=None, **kw):
        self.systems.append(messages[0]["content"])
        self.seen.append(messages[-1]["content"])
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return (answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)), 100


async def _no_read(file_id: str, **kw):
    return f"### 1.1 Определение\nтекст {file_id}", ["1.1 Определение"]


@pytest.fixture
def patient() -> dict:
    return {"AGE": 44, "SEX": "Ж"}


@pytest.fixture(autouse=True)
def _stub_reading(monkeypatch):
    monkeypatch.setattr(icd, "_read_guideline", _no_read)


async def test_manifest_goes_to_the_picker_once_and_not_to_the_judge(monkeypatch, patient):
    """Перечень клинреков — стоимость первого этапа, а не каждого шага.

    В ReAct он лежал в человеческом сообщении и переотправлялся с историей на
    каждом вызове: 558 строк ≈ 18 тыс. токенов, помноженные на число шагов.
    """
    client = _ScriptedClient(
        {"candidates": [{"dx_index": 0, "file_id": "mig_1", "reason": "картина мигрени"}]},
        {"better": True, "confidence": 9, "suggested_code": "G43.0",
         "comment": "картина соответствует мигрени", "section": "1.1", "cite": "приступы"},
    )
    monkeypatch.setattr(icd, "_client", client)

    issues, tokens = await icd.check_icd_codes(
        patient=patient,
        diagnoses=[{"КодМКБ": "R51", "НаименованиеМКБ": "Головная боль"}],
        manifest_rows=_manifest(),
        card_guid="card-1",
    )

    assert [i.suggested_code for i in issues] == ["G43.0"]
    assert tokens == 200
    assert "mig_1 | Мигрень" in client.seen[0]
    assert "Рекомендация 199" in client.seen[0]
    assert "mig_1 | Мигрень" not in client.seen[1]
    assert "Рекомендация 199" not in client.seen[1]


async def test_candidates_are_not_gated_by_the_doctors_own_code(monkeypatch, patient):
    """Гипотеза из другой буквы МКБ достижима.

    Прежний отбор оставлял только клинреки, начало кода которых совпадало с
    кодом врача: на `R51` мигрень `G43.0` исчезала до промпта.
    """
    client = _ScriptedClient(
        {"candidates": [{"dx_index": 0, "file_id": "mig_1", "reason": "—"}]},
        {"better": True, "confidence": 8, "suggested_code": "G43.1",
         "comment": "картина ближе к мигрени с аурой", "section": "1.5", "cite": "аура"},
    )
    monkeypatch.setattr(icd, "_client", client)

    issues, _ = await icd.check_icd_codes(
        patient=patient,
        diagnoses=[{"КодМКБ": "R51", "НаименованиеМКБ": "Головная боль"}],
        manifest_rows=_manifest(),
    )
    assert issues[0].initial_code == "R51"
    assert issues[0].suggested_code == "G43.1"
    assert issues[0].sources[0].cite == "аура"


async def test_suggested_code_outside_the_guideline_is_dropped(monkeypatch, patient):
    """Код не из той рекомендации, по тексту которой судили, — не рекомендация.

    Проверить его нашим же контуром нечем: в прогоне 21.08 так появился
    `I78.1 → D22.9`, и на него ушёл второй полный проход графа диагнозов.
    """
    client = _ScriptedClient(
        {"candidates": [{"dx_index": 0, "file_id": "mig_1", "reason": "—"}]},
        {"better": True, "confidence": 10, "suggested_code": "D22.9",
         "comment": "выдуманный код", "section": "1.1", "cite": "—"},
    )
    monkeypatch.setattr(icd, "_client", client)

    issues, _ = await icd.check_icd_codes(
        patient=patient,
        diagnoses=[{"КодМКБ": "I78.1"}],
        manifest_rows=_manifest(),
    )
    # Ни одна гипотеза не дала пригодного суждения — мнения нет.
    assert issues is None


async def test_low_confidence_stays_silent(monkeypatch, patient):
    client = _ScriptedClient(
        {"candidates": [{"dx_index": 0, "file_id": "mig_1", "reason": "—"}]},
        {"better": True, "confidence": 7, "suggested_code": "G43.0",
         "comment": "сомнительно", "section": "1.1", "cite": "—"},
    )
    monkeypatch.setattr(icd, "_client", client)

    issues, _ = await icd.check_icd_codes(
        patient=patient, diagnoses=[{"КодМКБ": "R51"}], manifest_rows=_manifest()
    )
    assert issues == []


async def test_no_hypotheses_is_an_opinion_not_a_failure(monkeypatch, patient):
    client = _ScriptedClient({"candidates": []})
    monkeypatch.setattr(icd, "_client", client)

    issues, tokens = await icd.check_icd_codes(
        patient=patient, diagnoses=[{"КодМКБ": "J06.9"}], manifest_rows=_manifest()
    )
    assert issues == []
    assert tokens == 100
    assert len(client.seen) == 1


async def test_picker_off_contract_means_no_opinion(monkeypatch, patient):
    client = _ScriptedClient("свободный текст вместо JSON")
    monkeypatch.setattr(icd, "_client", client)

    issues, tokens = await icd.check_icd_codes(
        patient=patient, diagnoses=[{"КодМКБ": "J06.9"}], manifest_rows=_manifest()
    )
    assert issues is None
    assert tokens == 100


async def test_invented_file_id_is_dropped(monkeypatch, patient):
    client = _ScriptedClient({"candidates": [{"dx_index": 0, "file_id": "нет_такого", "reason": "—"}]})
    monkeypatch.setattr(icd, "_client", client)

    issues, _ = await icd.check_icd_codes(
        patient=patient, diagnoses=[{"КодМКБ": "J06.9"}], manifest_rows=_manifest()
    )
    assert issues == []


async def test_hypotheses_are_capped_and_judged_independently(monkeypatch, patient):
    """Число вызовов известно до начала работы — зацикливаться нечему."""
    picked = [{"dx_index": 0, "file_id": f"{i}_1", "reason": "—"} for i in range(6)]
    silent = {"better": False, "confidence": 0, "suggested_code": "",
              "comment": "", "section": "", "cite": ""}
    client = _ScriptedClient({"candidates": picked}, silent, silent, silent)
    monkeypatch.setattr(icd, "_client", client)

    issues, tokens = await icd.check_icd_codes(
        patient=patient, diagnoses=[{"КодМКБ": "J06.9"}], manifest_rows=_manifest()
    )
    assert issues == []
    assert len(client.seen) == 1 + icd._MAX_HYPOTHESES
    assert tokens == 100 * (1 + icd._MAX_HYPOTHESES)
    for judged in client.seen[1:]:
        assert "Диагнозы врача (все диагнозы визита)" not in judged


async def test_one_failed_judgement_does_not_sink_the_others(monkeypatch, patient):
    client = _ScriptedClient(
        {"candidates": [{"dx_index": 0, "file_id": "mig_1", "reason": "—"},
                        {"dx_index": 0, "file_id": "5_1", "reason": "—"}]},
        RuntimeError("вызов не удался"),
        {"better": False, "confidence": 0, "suggested_code": "", "comment": "",
         "section": "", "cite": ""},
    )
    monkeypatch.setattr(icd, "_client", client)

    issues, _ = await icd.check_icd_codes(
        patient=patient, diagnoses=[{"КодМКБ": "R51"}], manifest_rows=_manifest()
    )
    assert issues == []


async def test_no_diagnoses_costs_nothing(monkeypatch, patient):
    client = _ScriptedClient()
    monkeypatch.setattr(icd, "_client", client)
    assert await icd.check_icd_codes(patient=patient, diagnoses=[], manifest_rows=_manifest()) == ([], 0)


def test_sections_for_judging_are_chosen_without_the_model():
    """Разделы отбирает код: нумерация 1.x по шаблону МЗ плюс ключевые слова."""
    toc = [
        "1.1 Определение заболевания",
        "1.4 Особенности кодирования по МКБ-10",
        "2 Диагностика",
        "3.1.1 Наружная терапия",
        "Критерии установления диагноза",
        "Критерии оценки качества медицинской помощи",
        "10.1 Приложение",
    ]
    picked = icd._pick_sections(toc)
    assert "1.1 Определение заболевания" in picked
    assert "1.4 Особенности кодирования по МКБ-10" in picked
    assert "Критерии установления диагноза" in picked
    assert "3.1.1 Наружная терапия" not in picked
    assert "10.1 Приложение" not in picked
    # «Критерии оценки качества» — раздел для аудита лечения, нозологии он не различает.
    assert "Критерии оценки качества медицинской помощи" not in picked


def test_persistence_distinguishes_no_opinion_from_no_findings():
    """NULL и [] в icd_check_result — разные вещи, различимые одним запросом."""
    from storage.done_cards_storage import _icd_check_json

    assert _icd_check_json(None) is None
    assert _icd_check_json([]) == "[]"
