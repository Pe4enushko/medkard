# medkard — docs

| Document | What it covers |
|---|---|
| [pipeline.md](pipeline.md) | `AuditPipeline` — batch orchestration, filtering, Excel + DB persistence |
| [formal_validator.md](formal_validator.md) | `FormalValidator` — visit type detection, rule filtering, LLM call |
| [storage.md](storage.md) | Хранилища, состояния `done_cards` и переаудит broken-карт |
| [revision-log.md](revision-log.md) | Ревизии логики аудита, справочников и эксплуатационные прогоны |
| [formal-rules-sources.md](formal-rules-sources.md) | реестр НПА, на которые опираются правила формальной проверки |
| [formal-rules-revision-log.md](formal-rules-revision-log.md) | Журнал ревизий нормативной базы формальных правил |
| [diagnosis_validator.md](diagnosis_validator.md) | `DiagnosisValidator` — three parallel checker agents per diagnosis |
| [clinic_recs.md](clinic_recs.md) | `ClinicRecs` — ICD → guideline file_id lookup (exact, prefix, BM25, LLM) |
| [llm_calls.md](llm_calls.md) | Every LLM call site: what is sent, what comes back, how to access token counts |
| [rag.md](rag.md) | Embeddings, vector store, hybrid search, LangChain tools, reverse HyDE |
| [clinic-data-requirements.md](clinic-data-requirements.md) | Формат пакета карт, который клиника передаёт в MedCheck |
| [iskra-integration.md](iskra-integration.md) | Push карт из МИС или 1С для работы с «Искрой» |
