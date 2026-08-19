from datetime import date

import pytest

from grls import status as st
from grls.format import MedicineLookup
from grls.parser import build_record
from tests.grls_fixtures import sample_row
import LLM.tools as tools


async def test_search_medicine_tool_delegates_to_lookup(monkeypatch):
    seen = {}

    async def fake_lookup(query, *, on=None):
        seen["query"], seen["on"] = query, on
        return MedicineLookup(query=query, on=on, registry_date=date(2026, 8, 17),
                              trade_records=[build_record(st.STATUS_ACTIVE, sample_row())])

    monkeypatch.setattr(tools, "lookup_medicine", fake_lookup)
    text = await tools.SearchMedicineTool()._arun("тестин")
    assert seen == {"query": "тестин", "on": None}
    assert text.startswith("Найдено в ГРЛС (1; реестр от 2026-08-17):")
    assert "Статус РУ: Действующий (РУ ЛП-000001, бессрочно)" in text


def test_tool_description_mentions_grls_and_status():
    d = tools.SearchMedicineTool().description
    assert "ГРЛС" in d and "статус" in d.lower()
    assert "ЕСКЛП" not in d


def test_drugs_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        import storage.drugs_storage  # noqa: F401
    import storage.models as m
    assert not hasattr(m, "Drug")
