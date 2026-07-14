"""decider/icd_prefix_picker принимают list[Guideline] напрямую (без dict-адаптера)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from storage.models.guideline import Guideline

_CANDIDATES = [
    Guideline(file_id="581_2", name="Острый бронхит", mkb=["J20.0"], age_category=["Взрослые"]),
    Guideline(file_id="581_3", name="Хронический бронхит", mkb=["J20.1"], age_category=["Взрослые"]),
]


async def test_decide_file_id_accepts_guideline_list_and_serializes_them():
    from LLM.decider import decide_file_id

    with patch("LLM.decider._client") as mock_client:
        mock_client.call = AsyncMock(return_value=("581_2", 10))
        chosen, tokens = await decide_file_id({"Возраст": 30}, {"КодМКБ": "J20.0"}, _CANDIDATES)

    assert chosen == "581_2"
    assert tokens == 10
    # candidates были сериализованы в промпт (JSON с их именами)
    user_msg = mock_client.call.call_args.kwargs["messages"][1]["content"]
    assert "Острый бронхит" in user_msg
    assert "581_2" in user_msg


async def test_decide_file_id_rejects_answer_outside_candidates():
    from LLM.decider import decide_file_id

    with patch("LLM.decider._client") as mock_client:
        mock_client.call = AsyncMock(return_value=("not_a_real_id", 5))
        chosen, tokens = await decide_file_id({}, {"КодМКБ": "J20.0"}, _CANDIDATES)

    assert chosen is None
    assert tokens == 5


async def test_icd_prefix_picker_accepts_guideline_list():
    from LLM.icd_prefix_picker import IcdPrefixPicker

    picker = IcdPrefixPicker()
    with patch.object(picker, "_client") as mock_client:
        mock_client.call = AsyncMock(return_value=("581_3", 7))
        chosen, tokens = await picker.pick({}, {"КодМКБ": "J20.1"}, _CANDIDATES)

    assert chosen == "581_3"
    assert tokens == 7
