# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**medkard** is a Russian ambulatory card (амбулаторная карта) audit system. It analyzes outpatient medical records for compliance and quality across four check dimensions:

1. **Formal structure** — presence and completeness of required sections
2. **Diagnosis check** — ICD coding correctness, consistency between diagnosis and documented symptoms
3. **Anamnesis check** — completeness of medical history, required fields per specialty
4. **Recommendation check** — appropriateness of prescribed treatments and follow-up plans

Core technology: LLM-based analysis combined with RAG using **reverse HyDE** (Hypothetical Document Embeddings in reverse — generating hypothetical queries from documents rather than hypothetical answers from queries).

## Architecture

```
medkard/
├── RAG/
│   ├── injection/
│   │   └── data_loader.py   # PDF ingestion: manifest → PDFContentReader generator
│   └── retrieval/           # (upcoming) reverse HyDE retrieval
├── LLM/                     # (upcoming) LLM orchestration / prompts
├── audit/
│   ├── pipelines/           # (upcoming) per-dimension audit pipelines
│   └── invalidation_flags/  # (upcoming) structured finding flags
├── pdfs/                    # source PDFs (not committed)
└── manifest.csv             # ID → metadata map (ID = PDF filename stem)
```

**Data flow**: `manifest.csv` + `pdfs/` → `load_documents()` → `PDFContentReader.iter_chunks()` → text/table chunks with metadata → vector store → reverse HyDE retrieval → LLM audit checks.

**Chunk format** (from `data_loader.py`):
```python
{
    "type": "text" | "table",
    "content": str | list[dict],   # str for text, list of row dicts for table
    "metadata": {
        "ID": str, "page": int, "section": str | None,
        "content_type": "text" | "table",
        # table-only: "table_index", "chunk_index"
        # + all other manifest.csv columns
    }
}
```

- Text extraction uses pymupdf `clip` rects to exclude table areas (horizontal slices between table bboxes)
- TOC sections (from `doc.get_toc()`) are mapped to pages and attached to every chunk's metadata
- Tables are extracted with tabula-py per detected bbox, converted to lists of row dicts, then split into batches of `TABLE_ROW_CHUNK_SIZE` (default 6) rows

## Development Setup

_Update this section once the stack is decided and dependencies are defined._

## Commands

_Update this section once the build system is in place._

## Key Design Notes

- Audit checks are independent and should be composable — a card can be checked on any subset of dimensions
- Reverse HyDE: embed the document chunk, generate hypothetical queries it would answer, use those query embeddings for retrieval — this improves recall for domain-specific medical terminology
- Russian-language medical text requires models/embeddings with strong Russian support
