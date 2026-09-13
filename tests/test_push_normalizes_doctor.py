"""POST /visits/push stores the card in our doctor form, before the demo stamp.

No database: the storage is replaced by a recorder and the org gate by a
stub, so what is checked is the order inside the route — normalize first,
stamp second. The stamp keys off Прием.Врач_код, so once the real code is
in place it must stay out.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api import demo_doctors
from api.app import create_app
from api.auth import require_org_access
from api.routes import visits as visits_route

ALENKA_CARD = {
    "Прием": {"GUID": "9491a742-59b0-429a-a05d-2d5501c39b91", "NUM": "ДКА-00223576", "DATE": "09.09.2026"},
    "Врач": {"GUID": "0a99d563-9ac7-11e8-ba9c-00155d8da706", "FIO": "Правкина Ирина Григорьевна",
             "SPECIALIZATION": "Педиатр"},
    "Пациент": {"CODE": "к0162184", "GENDER": "Мужской", "AGE": "1"},
    "Диагнозы": [],
}


class _Recorder:
    stored: dict | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get_priem(self, guid):
        return None

    async def upsert_pending(self, *, card_guid, card_data, organization_id):
        _Recorder.stored = json.loads(card_data)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(visits_route, "DoneCardsStorage", _Recorder)
    app = create_app()
    app.dependency_overrides[require_org_access] = lambda: ("org-1", "Alenka")
    _Recorder.stored = None
    return TestClient(app)


def _push(client):
    response = client.post("/visits/push?org=Alenka", json=ALENKA_CARD)
    assert response.status_code == 200, response.text
    return _Recorder.stored


def test_stored_card_is_in_our_form(client, monkeypatch):
    monkeypatch.delenv("DEMO_DOCTOR_STAMP_ORG", raising=False)
    stored = _push(client)
    assert stored["Прием"]["Врач"] == "Правкина Ирина Григорьевна"
    assert stored["Прием"]["Врач_код"] == "0a99d563-9ac7-11e8-ba9c-00155d8da706"
    assert stored["Врач"] == {"SPECIALIZATION": "Педиатр"}


def test_real_doctor_wins_over_stamp(client, monkeypatch):
    monkeypatch.setenv("DEMO_DOCTOR_STAMP_ORG", "Alenka")
    demo_doctors.load_doctors.cache_clear()
    stored = _push(client)
    assert stored["Прием"]["Врач_код"] == "0a99d563-9ac7-11e8-ba9c-00155d8da706"
