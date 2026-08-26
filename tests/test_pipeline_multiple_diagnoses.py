import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import audit.pipeline as pipeline_module
from audit.models import DiagnosisAuditResult
from storage.models.result import (
    DiagnosisIssue,
    GuidelineSource,
    GuidelineSourceSection,
)


class _FakeFormalValidator:
    async def validate(self, visit):
        return [], 0


class _FakeDiagnosisValidator:
    def __init__(self, visit):
        self.visit = visit

    async def validate_diagnosis(self, diagnosis):
        return DiagnosisAuditResult(
            guideline_file_id=f"guideline-{diagnosis['КодМКБ']}",
            guideline_meta={"name": "КР", "date": "2024", "age_group": "Взрослые"},
            icd_code=diagnosis["КодМКБ"],
            criteria_issues=[
                DiagnosisIssue(issue="Критерий не выполнен", aspect="criteria")
            ],
            guideline_sources=[
                GuidelineSource(
                    file_id=f"guideline-{diagnosis['КодМКБ']}",
                    doc_title="КР",
                    sections=[GuidelineSourceSection(section="Критерии", cited=True)],
                )
            ],
            errors=["partial"],
        ), 0


class _FakeGuidelinesStorage:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def all(self):
        return []


@pytest.fixture(autouse=True)
def _no_external_services(monkeypatch):
    async def check_icd_codes(**kwargs):
        return [], 0

    monkeypatch.setattr(pipeline_module, "GuidelinesStorage", _FakeGuidelinesStorage)
    monkeypatch.setattr(pipeline_module, "check_icd_codes", check_icd_codes)


def test_audit_visit_returns_all_diagnoses(monkeypatch):
    monkeypatch.setattr(pipeline_module, "FormalValidator", _FakeFormalValidator)
    monkeypatch.setattr(pipeline_module, "DiagnosisValidator", _FakeDiagnosisValidator)

    visit = {
        "Прием": {"GUID": "visit-1"},
        "Пациент": {"Возраст": 10},
        "Диагнозы": [
            {"КодМКБ": "A01", "НаименованиеМКБ": "First"},
            {"КодМКБ": "B02", "НаименованиеМКБ": "Second"},
        ],
    }

    result = asyncio.run(pipeline_module.AuditPipeline()._audit_visit(visit))

    assert [dx.icd_code for dx in result.diagnosis] == ["A01", "B02"]
    assert result.diagnosis[0].issues[0].aspect == "criteria"
    assert result.diagnosis[0].guideline_sources[0].doc_title == "КР"
    assert result.diagnosis[0].errors == ["partial"]


def test_run_skips_visits_with_done_guids(monkeypatch):
    monkeypatch.setattr(pipeline_module, "FormalValidator", _FakeFormalValidator)
    monkeypatch.setattr(pipeline_module, "DiagnosisValidator", _FakeDiagnosisValidator)

    payload = {
        "appointments": [
            {
                "Прием": {"GUID": "0b4121b2-39e1-11f1-a224-00155daa6107"},
                "Диагнозы": [{"КодМКБ": "A01"}],
            },
            {
                "Прием": {"GUID": "2a3b5100-39e1-11f1-a224-00155daa6107"},
                "Диагнозы": [{"КодМКБ": "B02"}],
            },
        ]
    }

    results = asyncio.run(
        pipeline_module.AuditPipeline().run(
            payload,
            done_guids={"0B4121B2-39E1-11F1-A224-00155DAA6107"},
        )
    )

    assert len(results) == 1
    assert (
        results[0][0].input["Прием"]["GUID"] == "2a3b5100-39e1-11f1-a224-00155daa6107"
    )


def test_run_batched_filters_done_guids_before_processing(monkeypatch):
    monkeypatch.setattr(pipeline_module, "FormalValidator", _FakeFormalValidator)
    monkeypatch.setattr(pipeline_module, "DiagnosisValidator", _FakeDiagnosisValidator)

    payload = {
        "appointments": [
            {
                "Прием": {"GUID": "11111111-39e1-11f1-a224-00155daa6107"},
                "Диагнозы": [{"КодМКБ": "A01"}],
            },
            {
                "Прием": {"GUID": "22222222-39e1-11f1-a224-00155daa6107"},
                "Диагнозы": [{"КодМКБ": "B02"}],
            },
            {
                "Прием": {"GUID": "33333333-39e1-11f1-a224-00155daa6107"},
                "Диагнозы": [{"КодМКБ": "C03"}],
            },
        ]
    }

    results = asyncio.run(
        pipeline_module.AuditPipeline().run_batched(
            payload,
            num_batches=2,
            done_guids={"22222222-39e1-11f1-a224-00155daa6107"},
        )
    )

    guids = [r[0].input["Прием"]["GUID"] for r in results]
    assert set(guids) == {
        "11111111-39e1-11f1-a224-00155daa6107",
        "33333333-39e1-11f1-a224-00155daa6107",
    }


def test_parallel_card_audits_have_separate_complete_traces(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_module, "FormalValidator", _FakeFormalValidator)
    monkeypatch.setattr(pipeline_module, "DiagnosisValidator", _FakeDiagnosisValidator)
    trace_path = tmp_path / "graphtraces.jsonl"
    monkeypatch.setenv("GRAPH_TRACE_PATH", str(trace_path))

    visits = [
        {
            "Прием": {"GUID": "trace-card-a", "DATE": "2026-08-21"},
            "Диагнозы": [{"КодМКБ": "A01"}],
        },
        {
            "Прием": {"GUID": "trace-card-b", "DATE": "2026-08-21"},
            "Диагнозы": [{"КодМКБ": "B02"}],
        },
    ]

    async def run():
        pipeline = pipeline_module.AuditPipeline()
        await asyncio.gather(*(pipeline._audit_visit(visit) for visit in visits))

    asyncio.run(run())

    records = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    starts = [row for row in records if row["event"] == "audit.started"]
    assert len(starts) == 2
    assert {row["card_guid"] for row in starts} == {"trace-card-a", "trace-card-b"}
    assert len({row["correlation_id"] for row in starts}) == 2
    assert {row["card_data_priem"]["GUID"] for row in starts} == {
        "trace-card-a",
        "trace-card-b",
    }

    for start in starts:
        card_records = [
            row for row in records if row["correlation_id"] == start["correlation_id"]
        ]
        assert {row["card_guid"] for row in card_records} == {start["card_guid"]}
        assert card_records[0]["event"] == "audit.started"
        assert card_records[-1]["event"] == "audit.completed"
        assert any(
            row["event"] == "checker.completed" and row["checker"] == "formal"
            for row in card_records
        )
        assert any(
            row["event"] == "checker.completed" and row["checker"] == "icd"
            for row in card_records
        )
        assert any(
            row["event"] == "checker.completed" and row["checker"] == "diagnosis"
            for row in card_records
        )


def test_failed_card_audit_closes_the_same_trace(monkeypatch, tmp_path):
    class FailingFormalValidator:
        async def validate(self, visit):
            raise RuntimeError("formal unavailable")

    monkeypatch.setattr(pipeline_module, "FormalValidator", FailingFormalValidator)
    trace_path = tmp_path / "graphtraces.jsonl"
    monkeypatch.setenv("GRAPH_TRACE_PATH", str(trace_path))
    visit = {"Прием": {"GUID": "trace-failed-card"}}

    with pytest.raises(RuntimeError, match="formal unavailable"):
        asyncio.run(pipeline_module.AuditPipeline()._audit_visit(visit))

    records = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in records] == [
        "audit.started",
        "checker.started",
        "checker.failed",
        "audit.failed",
    ]
    assert len({row["correlation_id"] for row in records}) == 1
    assert {row["card_guid"] for row in records} == {"trace-failed-card"}
    assert records[-1]["exception"]["message"] == "formal unavailable"


def test_audit_visit_keeps_the_guideline_snapshot(monkeypatch):
    # Снимок редакции идёт с результатом проверки до самой записи карты: без
    # этого шага в done_cards уезжает голый file_id.
    monkeypatch.setattr(pipeline_module, "FormalValidator", _FakeFormalValidator)
    monkeypatch.setattr(pipeline_module, "DiagnosisValidator", _FakeDiagnosisValidator)

    visit = {
        "Прием": {"GUID": "visit-2"},
        "Диагнозы": [{"КодМКБ": "A01", "НаименованиеМКБ": "First"}],
    }

    result = asyncio.run(pipeline_module.AuditPipeline()._audit_visit(visit))

    assert result.diagnosis[0].guideline_meta == {
        "name": "КР",
        "date": "2024",
        "age_group": "Взрослые",
    }
