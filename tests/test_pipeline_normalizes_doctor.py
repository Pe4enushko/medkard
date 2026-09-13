"""The nightly pipeline stores visits in our doctor form (parsers/doctor.py)."""

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
        return DiagnosisAuditResult(guideline_file_id=None, icd_code=diagnosis["КодМКБ"]), 0


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
    monkeypatch.setattr(pipeline_module, "FormalValidator", _FakeFormalValidator)
    monkeypatch.setattr(pipeline_module, "DiagnosisValidator", _FakeDiagnosisValidator)


def test_audited_visit_is_in_our_doctor_form():
    payload = {"appointments": [{
        "Прием": {"GUID": "9491a742-59b0-429a-a05d-2d5501c39b91", "DATE": "09.09.2026"},
        "Врач": {"GUID": "0a99d563-9ac7-11e8-ba9c-00155d8da706", "FIO": "Правкина Ирина Григорьевна",
                 "SPECIALIZATION": "Педиатр"},
        "Пациент": {"CODE": "к0162184", "AGE": "1"},
        "Диагнозы": [{"КодМКБ": "J06.9"}],
    }]}

    results = asyncio.run(pipeline_module.AuditPipeline().run(payload, done_guids=set()))

    stored = results[0][0].input
    assert stored["Прием"]["Врач_код"] == "0a99d563-9ac7-11e8-ba9c-00155d8da706"
    assert stored["Прием"]["Врач"] == "Правкина Ирина Григорьевна"
    assert stored["Врач"] == {"SPECIALIZATION": "Педиатр"}
