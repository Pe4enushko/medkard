"""check-guidelines-integrity.py — сверить resources/manifest.csv с БД и pdfs/.

Для каждой строки манифеста проверяет:
  - есть ли PDF-файл на диске (resolve_pdf_path: новая конвенция "КР{base}.pdf",
    затем старая "{file_id}.pdf");
  - есть ли запись в таблице guidelines (file_id);
  - есть ли хотя бы один чанк в таблице docs (RAG) для этого file_id, и сколько
    их всего;
  - нет ли дублирующихся чанков (одинаковый текст chunk дважды при одном
    file_id) — обычно след повторного инжеста без replace.

Подсвечивает несовпадения (отсутствие в любом из трёх мест) и печатает сводку.

    python scripts/checks/check-guidelines-integrity.py [--data-dir DIR] [--manifest-path FILE]
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from RAG.ingestion.data_loader import MANIFEST_PATH, PDFS_DIR, resolve_pdf_path
from storage import DocsStorage
from storage.guidelines_storage import GuidelinesStorage


def _read_manifest_ids(manifest_path: Path) -> list[str]:
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        return [fid for r in csv.DictReader(fh) if (fid := (r.get("ID") or "").strip())]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Check manifest vs. pdfs/ vs. guidelines vs. docs consistency.")
    parser.add_argument("--data-dir", type=Path, help="folder holding the PDFs (default: project pdfs/)")
    parser.add_argument("--manifest-path", type=Path, help="explicit path to manifest.csv")
    args = parser.parse_args()

    pdfs_dir = args.data_dir if args.data_dir is not None else PDFS_DIR
    manifest_path = args.manifest_path if args.manifest_path is not None else MANIFEST_PATH

    manifest_ids = _read_manifest_ids(manifest_path)
    manifest_id_set = set(manifest_ids)

    async with GuidelinesStorage() as guidelines_storage, DocsStorage() as docs_storage:
        guideline_ids = {g.file_id for g in await guidelines_storage.all()}
        chunk_counts = await docs_storage.get_chunk_counts()
        duplicate_counts = await docs_storage.get_duplicate_chunk_counts()
    docs_ids = set(chunk_counts)

    problems: list[tuple[str, list[str]]] = []
    for file_id in manifest_ids:
        missing = []
        if resolve_pdf_path(file_id, pdfs_dir) is None:
            missing.append("PDF")
        if file_id not in guideline_ids:
            missing.append("guidelines")
        if file_id not in docs_ids:
            missing.append("docs (RAG)")
        if missing:
            problems.append((file_id, missing))

    orphan_guidelines = sorted(guideline_ids - manifest_id_set)
    orphan_docs = sorted(docs_ids - manifest_id_set)

    print(f"Manifest: {len(manifest_ids)} file_id(s) | guidelines: {len(guideline_ids)} | "
          f"docs (distinct file_id): {len(docs_ids)}")
    print()

    if problems:
        print(f"❌ {len(problems)} manifest entr(y/ies) with missing data:")
        for file_id, missing in problems:
            print(f"  {file_id:<14} missing: {', '.join(missing)}")
    else:
        print("✅ Every manifest entry has a PDF, a guidelines row, and RAG chunks.")

    print()
    print("Chunk counts (manifest file_ids with docs entries):")
    for file_id in manifest_ids:
        n = chunk_counts.get(file_id)
        if n is None:
            continue
        dup = duplicate_counts.get(file_id, 0)
        dup_note = f"  ⚠️ {dup} duplicate row(s)" if dup else ""
        print(f"  {file_id:<14} {n} chunk(s){dup_note}")

    print()
    if orphan_guidelines:
        print(f"⚠️  {len(orphan_guidelines)} guidelines row(s) not in manifest: {', '.join(orphan_guidelines)}")
    if orphan_docs:
        print(f"⚠️  {len(orphan_docs)} file_id(s) with docs chunks but not in manifest: {', '.join(orphan_docs)}")

    if problems or duplicate_counts:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
