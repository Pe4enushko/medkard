# Migrations runbook

`migrate.sh` keeps an applied-migrations ledger (`schema_migrations`) and applies only
files not yet recorded. It never re-runs history.

## Fresh database

```bash
bash migrations/migrate.sh
```

Applies every migration in order and records each. No baseline needed.

## Existing / drifted database (one-time baseline)

The stand DB predates the ledger and its `docs` schema drifted from the files (it has an
untracked `chunk_embedding` + `hyde_reembedded`, not the files' `fact_q_*`). Baseline it so
history is not replayed, then run the reconcile:

```bash
# 1. Deploy the remove-hyde code (reads/writes docs.embedding).
# 2. Mark 001–023 applied WITHOUT running them:
bash migrations/migrate.sh --skip-until 024_docs_reconcile.sql
# 3. Fill docs.embedding for every row:
python scripts/knowledge/reingest-pdfs.py --force-all
# 4. Apply 024 (drops chunk_embedding/hyde_reembedded, keeps embedding):
bash migrations/migrate.sh
```

Order 3→4 is mandatory: `024_docs_reconcile.sql` aborts with a clear error if any row still
holds its vector only in `chunk_embedding`. Skipping the re-embed fails step 4 loudly rather
than dropping live vectors.

## Verify

```sql
\d docs
```

Expect: no `chunk_embedding` / `hyde_reembedded` / `fact_q_*`; `embedding VECTOR(1024)` present
with index `docs_embedding_idx`. Then spot-check retrieval.
