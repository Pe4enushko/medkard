"""Live regression checks for individual formal rules; no database writes.

Run with the configured LLM environment, or --env /path/to/dev/.env.
Unlike full-audit fixtures, these records isolate a rule. A parse failure
always fails the case, including cases expecting no finding.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


from formal_false_positive_cases import CASES



async def run() -> None:
    from audit.formal_structure.validator import FormalValidator, _RULES
    from LLM.client import LLMClient
    from LLM.validations import _RuleVerdict, _parse_verdict, _visit_message

    rules = {r["rule_id"]: r for r in _RULES}
    validator = FormalValidator()
    client = LLMClient()
    semaphore = asyncio.Semaphore(3)

    async def check(case: Case) -> dict:
        visit = {
            "Прием": {"GUID": f"regression-{case.name}", "DATE": "24.09.2026"},
            "Пациент": {"AGE": 1},
            "Врач": {"SPECIALIZATION": "Педиатр"},
            "Диагнозы": [{"КодМКБ": case.diagnosis[0], "НаименованиеМКБ": case.diagnosis[1]}],
            "ДанныеОсмотра": [{"Параметр": k, "Значение": v} for k, v in case.fields],
        }
        async with semaphore:
            raw, tokens = await client.call(
                messages=[
                    {"role": "system", "content": validator._render_prompt()},
                    {"role": "user", "content": _visit_message(visit)},
                    {"role": "user", "content": "## Единственное проверяемое правило\n\n" + validator._format_rules([rules[case.rule]])},
                ],
                temperature=0.0, response_model=_RuleVerdict,
            )
        verdict = _parse_verdict(raw)
        issue = verdict.issue.lower() if verdict else ""
        ok = (verdict is not None and verdict.violated == case.violated
              and (not verdict.violated or bool(issue.strip()))
              and all(t in issue for t in case.required)
              and all(t not in issue for t in case.forbidden))
        result = {"case": case.name, "ok": ok, "tokens": tokens,
                  "verdict": verdict.model_dump() if verdict else None}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result

    results = await asyncio.gather(*(check(case) for case in CASES))
    passed = sum(r["ok"] for r in results)
    print(f"{passed}/{len(results)} passed; tokens={sum(r['tokens'] for r in results)}")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path)
    args = parser.parse_args()
    if args.env:
        from dotenv import load_dotenv
        load_dotenv(args.env)
    asyncio.run(run())
