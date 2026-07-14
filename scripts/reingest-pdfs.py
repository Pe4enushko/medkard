#!/usr/bin/env python3
"""reingest-pdfs.py — re-sync docs + guidelines to the current manifest & PDFs, resumably.

Unlike ingest-pdfs.py, does NOT skip already-ingested files. Per file it decides
(via reingest_planner) between a full re-chunk, a cheap metadata-only guidelines
upsert, or skip — driven by the ingest_runs resume table (status + last-done PDF
hash) and a manifest-vs-guidelines diff.

    python scripts/reingest-pdfs.py [--only-failed] [--file-id ID]
"""
import argparse
import asyncio
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from RAG.ingestion.data_loader import MANIFEST_PATH, PDFS_DIR, PDF_EXTENSION, load_documents
from RAG.ingestion.pipeline import process_batch
from RAG.ingestion.reingest_planner import build_worklist, sha256_file
from storage import DocsStorage, IngestRunsStorage
from storage.guidelines_storage import GuidelinesStorage
from storage.models.guideline import Guideline

QUERY_GENERATION_BATCH_SIZE = 3

LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
log_filename = LOGS_DIR / f"reingest-pdfs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_filename, encoding="utf-8")],
)
log = logging.getLogger(__name__)


def _resolve_paths(data_dir: Path | None, manifest_path: Path | None) -> tuple[Path, Path]:
    """(pdfs_dir, manifest). --data-dir sets the PDFs folder (and manifest.csv inside it,
    unless --manifest-path overrides). Without either, use the project layout
    (pdfs/ + resources/manifest.csv). --manifest-path always wins for the manifest."""
    pdfs_dir = data_dir if data_dir is not None else PDFS_DIR
    if manifest_path is not None:
        manifest = manifest_path
    elif data_dir is not None:
        manifest = data_dir / "manifest.csv"
    else:
        manifest = MANIFEST_PATH
    return pdfs_dir, manifest


def _pdf_path(file_id: str, pdfs_dir: Path) -> Path:
    return pdfs_dir / (file_id + PDF_EXTENSION)


def _current_hash(file_id: str, pdfs_dir: Path):
    p = _pdf_path(file_id, pdfs_dir)
    return sha256_file(p) if p.exists() else None


def _read_manifest_rows(manifest_path: Path) -> dict:
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        return {(r.get("ID") or "").strip(): r
                for r in csv.DictReader(fh) if (r.get("ID") or "").strip()}


async def _full_reingest(file_id, row, pdfs_dir, manifest_path,
                         docs_storage, guidelines_storage, runs_storage):
    await runs_storage.upsert_pending(file_id)
    try:
        readers = list(load_documents(manifest_path=manifest_path, pdfs_dir=pdfs_dir, only={file_id}))
        if not readers:
            raise FileNotFoundError(f"no reader for {file_id} (missing PDF?)")
        chunks = list(readers[0].iter_chunks())

        docs = []
        for start in range(0, len(chunks), QUERY_GENERATION_BATCH_SIZE):
            batch = chunks[start:start + QUERY_GENERATION_BATCH_SIZE]
            docs.extend(d for d in await process_batch(batch, file_id) if d is not None)

        await docs_storage.replace_by_file_id(file_id, docs)
        await guidelines_storage.upsert_many([Guideline.from_manifest_row(row)])
        await runs_storage.mark_done(file_id, sha256_file(_pdf_path(file_id, pdfs_dir)))
        log.info("Reingested %s — %d chunk(s)", file_id, len(docs))
    except Exception as exc:  # per-file: never halt the whole run
        await runs_storage.mark_failed(file_id, str(exc))
        log.error("FAILED %s: %s", file_id, exc)


async def _metadata_only(file_id, row, guidelines_storage):
    await guidelines_storage.upsert_many([Guideline.from_manifest_row(row)])
    log.info("Metadata-only update for %s (PDF unchanged)", file_id)


def _summarize(worklist) -> dict:
    summary: dict[str, int] = {}
    for _, decision in worklist:
        summary[decision] = summary.get(decision, 0) + 1
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description="Reingest PDFs / sync guidelines with resume.")
    parser.add_argument("--data-dir", type=Path,
                        help="folder holding the PDFs (and manifest.csv unless --manifest-path is set) "
                             "(default: project pdfs/ + resources/manifest.csv)")
    parser.add_argument("--manifest-path", type=Path,
                        help="explicit path to manifest.csv (overrides --data-dir/manifest.csv)")
    parser.add_argument("--only-failed", action="store_true", help="only files with status='failed'")
    parser.add_argument("--file-id", help="force full reingest of one file_id (bypass diff logic)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the work-list decisions and exit without writing to the DB")
    args = parser.parse_args()

    pdfs_dir, manifest_path = _resolve_paths(args.data_dir, args.manifest_path)
    log.info("PDFs: %s | manifest: %s", pdfs_dir, manifest_path)
    manifest_rows = _read_manifest_rows(manifest_path)

    async with DocsStorage() as docs_storage, \
            GuidelinesStorage() as guidelines_storage, \
            IngestRunsStorage() as runs_storage:

        runs = await runs_storage.get_all()
        guidelines_by_id = {g.file_id: g for g in await guidelines_storage.all()}

        if args.file_id:
            worklist = [(args.file_id, "full")]
        else:
            worklist = build_worklist(manifest_rows, runs, guidelines_by_id,
                                      lambda fid: _current_hash(fid, pdfs_dir))
            if args.only_failed:
                worklist = [(fid, "full") for fid, _ in worklist
                            if runs.get(fid, (None, None))[0] == "failed"]

        log.info("Work-list: %d file(s) — %s", len(worklist), _summarize(worklist))

        if args.dry_run:
            for file_id, decision in worklist:
                log.info("[dry-run] %-14s %s", decision, file_id)
            log.info("[dry-run] no changes written.")
            return

        for file_id, decision in worklist:
            row = manifest_rows.get(file_id)
            if decision == "skip" or row is None:
                continue
            if decision == "metadata_only":
                await _metadata_only(file_id, row, guidelines_storage)
            else:
                await _full_reingest(file_id, row, pdfs_dir, manifest_path,
                                     docs_storage, guidelines_storage, runs_storage)

    log.info("Reingest complete. Log: %s", log_filename)


if __name__ == "__main__":
    asyncio.run(main())
