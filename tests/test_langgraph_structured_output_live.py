from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from LLM.client import LLMClient
from audit.diagnosis.schemas import CheckerOutput


def test_langgraph_react_agent_returns_checker_output_schema() -> None:
    output, tokens = asyncio.run(
        LLMClient(max_retries=0).call_agent(
            system_prompt=(
                "You are a structured-output test. Do not use markdown. "
                "Return only the requested structured response."
            ),
            tools=[],
            human_message=(
                "Return exactly one issue. "
                "The issue text must be exactly SCHEMA_TEST_OK. "
                "Include exactly one source with doc_title=unit-test, "
                "section=structured-output, cite=schema enforced."
            ),
            response_format=CheckerOutput,
        )
    )

    assert isinstance(output, CheckerOutput)
    assert tokens > 0
    assert len(output.issues) == 1
    assert output.issues[0].issue == "SCHEMA_TEST_OK"
    assert len(output.issues[0].sources) == 1
    assert output.issues[0].sources[0].doc_title == "unit-test"
    assert output.issues[0].sources[0].section == "structured-output"
    assert output.issues[0].sources[0].cite == "schema enforced"
