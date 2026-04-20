"""
Run _audit_visit on a single visit parsed from a JSON file.

Change APPOINTMENTS_JSON_PATH to point at the JSON file containing an
appointments payload. The first appointment in the array will be audited.
"""

import asyncio
from pathlib import Path

from audit.pipeline import AuditPipeline
from parsers.json_parser import AppointmentsParser

APPOINTMENTS_JSON_PATH = r"visit.json"


async def main() -> None:
    path = Path(APPOINTMENTS_JSON_PATH)
    visits = AppointmentsParser.split_file(path)
    visit = visits[0]

    pipeline = AuditPipeline(excel_path="audit_results.xlsx")
    results = await pipeline._audit_visit(visit)

    for i, result in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Flags : {result.flags}")
        print(f"Issues: {result.issues}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
