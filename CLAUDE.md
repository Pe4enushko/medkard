# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**medkard** is a Russian ambulatory card (амбулаторная карта) audit system. It fetches outpatient visit records from a 1C medical information system, audits them for compliance and quality across two active dimensions, and exports findings to Excel.

Active audit dimensions:
1. **Formal structure** (`src/audit/formal_structure/`) — presence and completeness of required sections, per visit type and patient age
2. **Diagnosis check** (`src/audit/diagnosis/`) — clinical-guideline compliance via three parallel checker agents (anamnesis / inspection / treatment)

Core technology: LLM-based analysis (OpenAI-compatible API) combined with RAG — chunk contextual text (section + body) is embedded at ingest and matched against the query embedding, with a BM25 + vector RRF hybrid.

## Environment

Copy `.env.example` to `.env` and fill in:
- `OPENAI_API_KEY` + `LLM_MODEL` (default `gpt-4o-mini`) + optional `OPENAI_BASE_URL`
- `POSTGRES_*` — PostgreSQL connection (pgvector extension required)
- `EMBEDDING_PROVIDER` — `openai` (REST) or `st` (sentence-transformers local) or `fastembed`
- `EMBEDDING_MODEL` + `EMBEDDING_DIM` — must match the chosen provider's output dimension
- `ALENKA_ONE_C_*` / `MDS_ONE_C_*` — 1C integration credentials

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (pytest-asyncio, asyncio_mode=auto, pythonpath=src)
pytest

# Run a single test
pytest tests/test_validations.py

# Ingest clinical-guideline PDFs (manifest.csv + pdfs/ → Postgres docs table)
python scripts/ingest-pdfs.py

# Run a full audit for a 1C org and export to Excel
python scripts/audit-one-c-period.py Alenka [--days N] [--date DD.MM.YYYY] [--ignore-icd CODE ...] [--num-batches N] [--ftpcreds FILE]

# Audit a single local JSON file
python scripts/audit-file.py <path-to-visit-json>

# Replay today's cached 1C data (for development)
python scripts/mock-run-today.py
```

## Architecture

```
src/
├── audit/
│   ├── pipeline.py          # AuditPipeline: entry point, concurrency via asyncio.Semaphore
│   ├── models.py            # Result dataclasses (DiagnosisAuditResult, etc.)
│   ├── excel_formatter.py   # ExcelFormatter: DB → Excel export
│   ├── formal_structure/    # FormalStructureValidator: rules.json-driven LLM check
│   └── diagnosis/
│       ├── validator.py     # DiagnosisValidator: 3 parallel LLM checker agents
│       └── clinic_recs.py   # ClinicRecs: ICD → guideline file_id lookup
├── LLM/
│   ├── base.py              # Shared OpenAI client + MODEL constant
│   ├── rag_agent.py         # create_checker_agent() — LangChain ReAct agent
│   ├── tools.py             # Per-file-id RAG retrieval tools for checker agents
│   ├── visit_classifier.py  # VisitClassifier: primary / repeat / prophylactic
│   ├── icd_prefix_picker.py # ICD prefix matching for guideline lookup
│   ├── decider.py           # LLM-based binary decision helper
│   └── chinese_detector.py  # Detects and repairs hallucinated Chinese characters in LLM output
├── RAG/
│   ├── ingestion/
│   │   └── data_loader.py   # load_documents(): manifest.csv + pdfs/ → chunk generator
│   └── retrieval/
│       ├── vector_store.py  # Postgres/pgvector pool, hybrid search (HNSW + BM25 via RRF)
│       ├── embeddings.py    # embed(): async embedding via configured provider
│       └── searches.py      # file/section-scoped hybrid search
├── storage/                 # psycopg3 async storage classes (BaseStorage pattern)
│   ├── docs_storage.py      # Clinical guideline chunks (docs table)
│   ├── done_cards_storage.py# Deduplication: already-audited visit GUIDs
│   ├── drugs_storage.py     # Drug reference data
│   └── ...
├── integrations/
│   ├── one_c.py             # AlenkaOneCClient / MdsOneCClient: fetch visits from 1C
│   └── ftp.py               # FTP upload for Excel reports
└── parsers/
    ├── excel.py             # Input Excel parser
    └── json_parser.py       # Visit JSON normalization
```

## Key Data Flows

**Ingestion** (`scripts/ingest-pdfs.py`):
`manifest.csv` + `pdfs/` → `load_documents()` → chunks → embed contextual text (section + body) → insert into `docs` table with one embedding column.

**Audit pipeline** (`AuditPipeline.run_batched`):
1C JSON → parse visits → deduplicate via `done_cards_storage` → concurrent audit with `asyncio.Semaphore` → for each visit: `FormalStructureValidator.validate()` + `DiagnosisValidator.validate_diagnosis()` per ICD code → persist `Result` to DB → `ExcelFormatter.export_period()` → optional FTP upload.

**Diagnosis checker** (per ICD code):
`ClinicRecs.pick_recs()` → guideline `file_id` → three parallel LangChain ReAct agents (anamnesis / inspection / treatment), each equipped with file-scoped RAG retrieval tools from `LLM/tools.py` → issues parsed, Chinese-character hallucinations repaired by `ChineseDetector`.

## Key Design Notes

- **Two storage backends coexist**: `asyncpg` (used in `RAG/retrieval/vector_store.py` for pgvector operations) and `psycopg3` (used in `storage/` for all other DB access). They share the same Postgres DB but use separate connection pools.
- **Contextual embeddings**: each chunk is embedded from its section header + body; at retrieval the query embedding is matched against those chunk embeddings (BM25+RRF hybrid).
- **Chinese character detection**: `ChineseDetector` (`LLM/chinese_detector.py`) post-processes every LLM output to detect and repair hallucinated CJK characters, which can appear in Russian-language responses from some models.
- **Deduplication**: `DoneCardsStorage` tracks audited visit GUIDs so re-runs skip already-processed cards. Visits with unrecognised visit types are upserted as "ignored" rather than audited.
- **Prompt files** live under `src/LLM/prompts/` as `.txt` files loaded at import time.
- **`pythonpath = src`** is set in `pytest.ini`, so imports like `from audit.pipeline import AuditPipeline` work in tests without installation.
