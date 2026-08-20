from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from LLM.client import LLMClient
from LLM.graphs.diagnosis_state import JudgeOutput


def test_raw_llm_call_returns_judge_output_schema() -> None:
    raw, tokens = asyncio.run(
        LLMClient(max_retries=0).call(
            messages=[
                {
                    "role": "system",
                    "content": "Return only the requested structured response.",
                },
                {
                    "role": "user",
                    "content": "Return exactly one issue named SCHEMA_TEST_OK with chunk_refs [1].",
                },
            ],
            temperature=0,
            response_model=JudgeOutput,
            reasoning_effort="low",
        )
    )
    output = JudgeOutput.model_validate_json(raw)

    assert isinstance(output, JudgeOutput)
    assert tokens > 0
    assert len(output.issues) == 1
    assert output.issues[0].issue == "SCHEMA_TEST_OK"
    assert output.issues[0].chunk_refs == [1]
