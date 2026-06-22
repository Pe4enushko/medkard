import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import audit.pipeline as pipeline_module
from audit.models import DiagnosisAuditResult


class _FakeFormalValidator:
    async def validate(self, visit):
        return [], 0


class _FakeDiagnosisValidator:
    def __init__(self, visit):
        self.visit = visit

    async def validate_diagnosis(self, diagnosis):
        return DiagnosisAuditResult(
            guideline_file_id=f"guideline-{diagnosis['КодМКБ']}",
            icd_code=diagnosis["КодМКБ"],
        ), 0


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
    assert results[0][0].input["Прием"]["GUID"] == "2a3b5100-39e1-11f1-a224-00155daa6107"


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
