"""
Run _audit_visit on a single visit parsed from a JSON file.

Change VISIT_JSON_PATH to point at the JSON file containing one visit dict.
"""

import asyncio
import json
from pathlib import Path

from audit.pipeline import AuditPipeline

VISIT_JSON_PATH = r"visit.json"


async def main() -> None:
    path = Path(VISIT_JSON_PATH)
    with open(path, encoding="utf-8") as f:
        visit = json.load(f)

    pipeline = AuditPipeline(excel_path="audit_results.xlsx")
    results = await pipeline._audit_visit(visit)

    for i, result in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Flags : {result.flags}")
        print(f"Issues: {result.issues}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
