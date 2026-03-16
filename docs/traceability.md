Here’s a ready-to-paste docs/traceability.md starter that is SPEC-accurate, reviewer-complete, and designed to prevent exactly the kind of feedback loss you’re worried about.

This file becomes your control plane: every refactor, reviewer comment, and PR must map here.

You can paste this verbatim.

⸻

docs/traceability.md

Sovereign IDP – SPEC Traceability Matrix

This document provides end-to-end traceability from SPEC.md + SPEC_ADDENDUM.md to:
	•	implementation modules,
	•	tests,
	•	demo/acceptance evidence.

If a requirement is not mapped here, it is considered not implemented.

⸻

Legend
	•	SPEC → original system specification (Sections 1–12)
	•	ADD → SPEC_ADDENDUM.md (v1.1)
	•	Impl → primary code location(s)
	•	Tests → required automated or manual verification
	•	Status → Planned | In-Progress | Complete

⸻

1. Purpose (SPEC §1)

Requirement	Spec Ref	Impl	Tests / Evidence	Status
600+ page PDF ingestion without crash	§1, §10	ingestion/, extraction/	Integration test: large PDF	Planned
Evidence-grounded answers	§1, §8	retrieval/, synthesis/	QA test with citations	Planned
Visual PDF highlights	§1, §9.2	grounding/, ui/	Manual UI demo	Planned
Late chunking preserved	§1, §6	embedding/	Unit test: global pass	Planned
Selective Azure DI	§1, §5	ingestion/triage.py	Triage tests	Planned


⸻

2. Goals (SPEC §2)

Requirement	Spec Ref	Impl	Tests / Evidence	Status
Mixed-content ingestion (text/table/scan)	§2.1	extraction/, ingestion/	Test PDFs	Planned
Policy-based selective DI	§2.2, §5	ingestion/triage.py	Unit + audit tests	Planned
Full lineage preserved	§2.3, §4.2	storage/, retrieval/	Lineage invariant tests	Planned
Attribute extraction	§2.4	synthesis/, retrieval/	Attribute tests	Planned
Free-form analytical queries	§2.4	retrieval/router.py	QA tests	Planned
Local Mac CPU execution	§2.6	embedding/, reranker	Runtime validation	Planned


⸻

3. Non-Goals (SPEC §3)

Constraint	Spec Ref	Enforcement	Evidence	Status
No perfect table cell semantics	§3	Table chunks atomic	Code review	Planned
No chart vision reasoning	§3	No vision models	Code review	Planned
No production hardening	§3	No auth/HA	Code review	Planned
Azure DI retained	§3	DI modules present	Code review	Planned


⸻

4. Architectural Invariants (SPEC §4)

4.1 Separation of Concerns

Invariant	Spec Ref	Impl	Tests	Status
No layer bypass	§4.1	package boundaries	Lint / review	Planned

4.2 Deterministic Lineage

Field	Spec Ref	Impl	Tests	Status
doc_id, page_numbers	§4.2	chunks table	Insert tests	Complete
char_start / char_end	§4.2	canonicalization	Offset tests	Complete
polygons	§4.2	extraction/grounding	UI highlight	Complete
macro_id / child_id	§4.2	embedding	Unit tests	Complete
anchor decision debug reasons	§4.2	retrieval/router.py	tests/test_items_of_note_anchor_not_adjusted_measures.py; tests/test_items_of_note_anchor_rejects_reconciliation_reference_only.py	Complete

4.3 Embedding Invariant

Requirement	Spec Ref	Impl	Tests	Status
ModernBERT only	§4.3	embedding/config	Runtime assert	Planned
768-dim vectors	§4.3	storage	Unit test	Planned
Global pass + pooling	§4.3	embedding/late_chunk.py	Behavioral test	Planned


⸻

5. Selective Azure DI Policy (SPEC §5)

Requirement	Spec Ref	Impl	Tests	Status
Page triage metrics	§5.2	ingestion/metrics.py	Unit tests	Planned
DI decision rules	§5.3	ingestion/policy.py	Rule tests	Planned
Audit persistence	§5.4	pages table	DB tests	Planned
DI disable behavior	§5.5	config + ingestion	Toggle test	Planned


⸻

6. Late Chunking (SPEC §6)

Requirement	Spec Ref	Impl	Tests	Status
Macro chunking (8192)	§6.1	embedding	Token tests	Planned
Token offset mapping	§6.3	tokenizer logic	Span tests	Planned
Context-sensitive embeddings	§6.4	embedding	Context diff test	Planned


⸻

7. Database Contract (SPEC §7)

Requirement	Spec Ref	Impl	Tests	Status
documents/pages/chunks tables	§7.1	storage/schema.sql; storage/migrations/001_add_chunk_type.sql; storage/schema_contract.py	tests/test_migrations_and_contract.py; tests/test_ingestion_smoke.py	In-Progress
document_facts table	§7.1	storage/schema.sql; storage/migrations/002_document_facts.sql; storage/schema_contract.py	tests/test_document_facts.py	In-Progress
HNSW + B-tree indexes	§7.2	migrations	Index existence	Planned


⸻

8. Retrieval & Answering (SPEC §8)

Requirement	Spec Ref	Impl	Tests	Status
Lineage-rich retrieval	§8.1	retrieval/	Retrieval tests	Planned
Document metadata shortcut	§8.1	retrieval/metadata.py; app/app.py	tests/test_document_facts.py	In-Progress
LLM constrained to evidence	§8.2	synthesis/prompt	Hallucination test	Planned
Items-of-note numeric_list anchor hardening	§8.1	retrieval/router.py	tests/test_items_of_note_anchor_not_adjusted_measures.py; tests/test_items_of_note_anchor_rejects_reconciliation_reference_only.py	Complete
BM25 index caching for hybrid retrieval	§8.1	retrieval/bm25_index.py; retrieval/hybrid.py; app/app.py	tests/test_bm25_index_manager.py	Complete


⸻

9. UI Requirements (SPEC §9)

Feature	Spec Ref	Impl	Evidence	Status
Streamlit UI only	§9	ui/	Manual check	Planned
Triage summary	§9.1	ui/triage	UI demo	Planned
Clickable highlights	§9.2	grounding/ui	Visual demo	Planned


⸻

10. Acceptance Criteria (SPEC §10)

Criterion	Spec Ref	Evidence	Status
600-page ingest	§10.1	Demo / test	Planned
Inspectable triage	§10.2	UI screenshot	Planned
Answer + citation	§10.3	docs/demo_evidence.md; docs/demo_evidence_di.md	Complete
Highlight on click	§10.4	UI demo	Planned
Lineage preserved	§10.5	End-to-end test	Planned


⸻

11. Change Management (SPEC §11)

Rule	Enforcement	Evidence	Status
Schema change → spec update	PR template	storage/schema_contract.py; tests/test_schema_contract.py	In-Progress
DI policy change → §5 update	Review gate	SPEC diff	Planned
Decisions logged	docs/decisions.md	Entry present	Complete


⸻

12. Final Authority (SPEC §12)

Rule	Enforcement	Status
SPEC wins on conflict	PR review gate	Enforced


⸻

SPEC ADDENDUM TRACEABILITY

Addendum Requirement	Addendum Ref	Impl	Tests	Status
Query intent routing + subtypes (list/attribute/numeric_list/pointer)	A1	retrieval/router.py; synthesis/openai_client.py; app/app.py	tests/test_coverage_query.py; tests/test_router_intent_subtypes.py; tests/test_pointer_query_note_reference.py; tests/test_coverage_attribute_range_losses.py	Complete
Page-filtered retrieval	A1	retrieval/vector_search.py	Page query test	Complete
Section expansion (Locate → Then Read)	A1	retrieval/router.py; retrieval/vector_search.py	tests/test_closed_matters_plan.py; tests/test_coverage_query_integration.py	Complete
heading_path persistence	A2	canonicalization	tests/test_coverage_query_integration.py	Complete
Atomic table chunks	A3	ingestion/canonicalize.py; embedding/late_chunking.py	tests/test_table_atomicity.py	Complete
Table-aware filtering for MD&A	A3	retrieval/router.py	tests/test_table_filtering_policy.py; tests/test_items_of_note_anchor_never_table.py	Complete
Hybrid retrieval (RRF)	A4	retrieval/hybrid.py	Fusion test	Complete
Verifier (Yes/No)	A4	synthesis/verifier.py	Support test	Complete
Cross-encoder reranker	A4	retrieval/rerank.py	Rerank tests	Complete
Document-level fact caching	A6	ingestion/document_facts.py; retrieval/metadata.py; storage/schema.sql	tests/test_document_facts.py	In-Progress


⸻

⸻

13. Engineering Quality & Runtime Safety (SPEC §13, WO-010)

Requirement	Spec Ref	Impl	Tests	Status
Model loads once per process	§13	embedding/model_registry.py; retrieval/vector_search.py; embedding/late_chunking.py	tests/test_wo010_model_singleton.py	Complete
Connection pooling	§13	storage/db_pool.py; storage/db.py	tests/test_wo010_pool_usage.py	Complete
No unsafe deserialization	§13	retrieval/bm25_index.py (JSON cache)	tests/test_wo010_no_pickle.py	Complete
Regex extraction correct	§13	ingestion/document_facts.py	tests/test_wo010_regex_document_facts.py	Complete
Idempotent chunk writes	§13	storage/migrations/003_chunks_idempotency.sql; storage/repo.py	tests/test_wo010_chunk_idempotency.py	Complete

⸻

14. Preprocessor Agent — Chunking Strategy Selection (§4.9)

Requirement	Spec Ref	Impl	Tests	Status
Deterministic doc-type → strategy mapping	§4.9	agents/preprocessor_agent.py (_STRATEGY_RULES)	tests/test_preprocessor_agent.py::TestDeterministicStrategy	Complete
Learned strategy from past outcomes	§4.9	agents/preprocessor_agent.py (OutcomeStore)	tests/test_preprocessor_agent.py::TestLearnedStrategy	Complete
Skip decision for empty/zero-page docs	§4.9	agents/preprocessor_agent.py (Tier 0)	tests/test_preprocessor_agent.py::TestSkipDecision	Complete
Triage-based strategy adjustment	§4.9	agents/preprocessor_agent.py (_adjust_for_triage)	tests/test_preprocessor_agent.py::TestTriageAdjustments	Complete
Typed contracts (PreprocessorInput/Result/Strategy/Outcome)	§4.9	agents/contracts.py	tests/test_preprocessor_agent.py::TestMessageBusIntegration	Complete
Feature flag ENABLE_PREPROCESSOR	§4.9	core/config.py	Manual toggle test	Complete
Ingestion pipeline integration	§4.9	ingestion/ingest_pipeline.py (_run_preprocessor, _record_chunking_outcome)	tests/test_preprocessor_agent.py	Complete
Chunking outcome learning loop	§4.9	agents/preprocessor_agent.py (record_outcome)	tests/test_preprocessor_agent.py::TestOutcomeStore	Complete

⸻

15. Schema-Driven Extraction (§10)

Requirement	Spec Ref	Impl	Tests	Status
ExtractionSchema + FieldDefinition contracts	§10.1	agents/contracts.py	tests/test_extractor_agent.py	Complete
Data Extractor Agent (regex + LLM)	§10.1	agents/extractor_agent.py	tests/test_extractor_agent.py	Complete
Per-field confidence scoring	§10.1	agents/contracts.py (ExtractedField.confidence)	tests/test_extractor_agent.py::test_overall_confidence	Complete
Field validation (regex, allowed_values, required)	§10.1	agents/extractor_agent.py (_validate_field)	tests/test_extractor_agent.py::test_allowed_values_validation	Complete
Schema registry (register/get/list)	§10.1	agents/extractor_agent.py	tests/test_extractor_agent.py::test_register_and_get_schema	Complete
Feature flag ENABLE_EXTRACTOR	§10.1	core/config.py	Manual toggle	Complete
DB migration (extraction tables)	§10.1	storage/migrations/008_extraction_schema.sql	Schema test	Complete

⸻

16. Transformer Agent & MCP Reference Data (§10.2, §10.3)

Requirement	Spec Ref	Impl	Tests	Status
MCPLookupRequest/Response contracts	§10.3	agents/contracts.py	tests/test_transformer_agent.py::test_mcp_*	Complete
MCP Reference Server (reference impl)	§10.3	agents/mcp_reference_server.py	tests/test_transformer_agent.py::test_direct_lookup_*	Complete
Transformer Agent (MCP lookup + transforms)	§10.2	agents/transformer_agent.py	tests/test_transformer_agent.py	Complete
Date/case/regex transforms	§10.2	agents/transformer_agent.py	tests/test_transformer_agent.py::test_transformer_date_format	Complete
TransformationRule registry	§10.2	agents/transformer_agent.py	tests/test_transformer_agent.py::test_register_and_get_rules	Complete
Feature flag ENABLE_TRANSFORMER	§10.2	core/config.py	Manual toggle	Complete

⸻

17. Parser Abstraction Layer (§10.4)

Requirement	Spec Ref	Impl	Tests	Status
BaseParser abstract class	§10.4	ingestion/parser_base.py	tests/test_parser_abstraction.py	Complete
PyMuPDF parser backend	§10.4	ingestion/pymupdf_parser.py	tests/test_parser_abstraction.py::test_pymupdf_*	Complete
Docling parser backend (optional)	§10.4	ingestion/docling_parser.py	Requires docling install	Complete
Parser registry (register/get/list)	§10.4	ingestion/parser_base.py	tests/test_parser_abstraction.py::test_register_*	Complete
Config PARSER_BACKEND	§10.4	core/config.py	Manual toggle	Complete

⸻

18. Azure OpenAI Support (§7.3)

Requirement	Spec Ref	Impl	Tests	Status
Azure OpenAI via AzureOpenAI client	§7.3	agents/model_gateway.py (_execute_azure_openai_call)	tests/test_azure_openai.py	Complete
LLM_PROVIDER config flag	§7.3	core/config.py	tests/test_azure_openai.py::test_azure_openai_provider_selection	Complete
Fallback to vanilla OpenAI	§7.3	agents/model_gateway.py	tests/test_azure_openai.py::test_fallback_to_vanilla	Complete
Azure deployment ID override	§7.3	agents/model_gateway.py	tests/test_azure_openai.py	Complete

⸻

19. Multi-Format Document Ingestion (§10.5)

Requirement	Spec Ref	Impl	Tests	Status
DOCX parser (python-docx)	§10.5	ingestion/multi_format_parser.py (DocxParser)	Requires python-docx	Complete
Excel parser (openpyxl)	§10.5	ingestion/multi_format_parser.py (ExcelParser)	Requires openpyxl	Complete
CSV parser (stdlib)	§10.5	ingestion/multi_format_parser.py (CsvParser)	tests/test_parser_abstraction.py::test_csv_*	Complete
JSON parser (stdlib)	§10.5	ingestion/multi_format_parser.py (JsonParser)	tests/test_parser_abstraction.py::test_json_*	Complete
HTML parser (beautifulsoup4)	§10.5	ingestion/multi_format_parser.py (HtmlParser)	Requires bs4	Complete
Config ENABLE_MULTI_FORMAT	§10.5	core/config.py	Manual toggle	Complete

⸻

20. Feedback Loop Subsystem — Canonical Feedback/Retraining Architecture

Requirement	Spec Ref	Impl	Tests	Status
Typed feedback domain models (Pydantic v2)	§4.10	feedback_loop/models.py (25 models, 6 enums)	tests/test_feedback_loop.py::TestBoundaryKey	Complete
BoundaryKey isolation: B = (client, division, jurisdiction)	§4.10	feedback_loop/models.py (BoundaryKey, min_length=1)	tests/test_feedback_loop.py::TestBoundaryPolicyGuard	Complete
Feedback ingestion with boundary validation	§4.10	feedback_loop/services.py (InMemory + Postgres)	tests/test_feedback_loop.py::TestEndToEndPipeline	Complete
Prediction trace join service	§4.10	feedback_loop/services.py (InMemory + Postgres)	tests/test_feedback_loop.py::TestEndToEndPipeline	Complete
6-rule deterministic attribution engine	§4.10	feedback_loop/attribution.py (RuleBasedAttributionEngine)	tests/test_feedback_loop.py::TestRule1-6	Complete
Feedback normalization (comment → reason codes)	§4.10	feedback_loop/normalizer.py (DefaultFeedbackNormalizer)	tests/test_feedback_loop.py::TestNormalizer	Complete
Layer-specific training row builders (5 layers)	§4.10	feedback_loop/training_rows.py (DefaultTrainingRowBuilder)	tests/test_feedback_loop.py::TestTrainingRowBuilder	Complete
Boundary policy guard with sanitization	§4.10	feedback_loop/boundary.py (DefaultBoundaryPolicyGuard)	tests/test_feedback_loop.py::TestBoundaryPolicyGuard	Complete
End-to-end pipeline orchestration	§4.10	feedback_loop/pipeline.py (FeedbackLoopPipeline)	tests/test_feedback_loop.py::TestEndToEndPipeline	Complete
Quarantine semantics for missing traces	§4.10	feedback_loop/pipeline.py (quarantined=True)	tests/test_feedback_loop.py::test_pipeline_no_trace_quarantine	Complete
Model lifecycle management (shadow→canary→approved)	§4.10	feedback_loop/services.py (InMemoryModelPromotionController)	tests/test_feedback_loop.py::TestModelPromotion	Complete
Model evaluation with baseline comparison	§4.10	feedback_loop/services.py (DefaultModelEvaluator)	tests/test_feedback_loop.py::TestModelEvaluator	Complete
Abstract service interfaces (pluggable backends)	§4.10	feedback_loop/interfaces.py (9 ABCs)	tests/test_feedback_loop.py	Complete
DB-backed feedback services (Postgres)	§4.10	feedback_loop/services.py (Postgres*)	Integration test (requires DB)	Complete
DB migration for feedback loop tables	§4.10	storage/migrations/009_feedback_loop_subsystem.sql (14 tables)	Schema test (requires DB)	Complete
Deprecation of agents/feedback_agent.py	§4.10	agents/feedback_agent.py (DeprecationWarning)	Import warning test	Complete
Deprecation of agents/retraining_agent.py	§4.11	agents/retraining_agent.py (DeprecationWarning)	Import warning test	Complete

⸻

21. Enterprise State Model

Requirement	Spec Ref	Impl	Tests	Status
DocumentLifecycle enum	§4	core/contracts.py (10 states)	Code review	Complete
QueryState enum	§4	core/contracts.py (7 states)	Code review	Complete
FeedbackState enum	§4.10	core/contracts.py (6 states)	Code review	Complete

⸻

22. Schema Contract Validation — Enterprise Coverage

Requirement	Spec Ref	Impl	Tests	Status
Core table validation (documents, pages, chunks, document_facts)	§7	storage/schema_contract.py	tests/test_migrations_and_contract.py	Complete
Enterprise field validation (document_type, classification_label, updated_at)	§7	storage/schema_contract.py	Schema test (requires DB)	Complete
Lineage field validation (heading_path, section_id)	§2.1	storage/schema_contract.py	Schema test (requires DB)	Complete
Audit log table validation	§2.4	storage/schema_contract.py	Schema test (requires DB)	Complete
Feedback loop table validation (prediction_traces, feedback_events, etc.)	§4.10	storage/schema_contract.py	Schema test (requires DB)	Complete

⸻

23. Audit Log Repository Abstraction

Requirement	Spec Ref	Impl	Tests	Status
Audit writes through repo layer (§2.5 SoC)	§2.4, §2.5	storage/repo.py (insert_audit_entry, insert_audit_entries)	Code review	Complete
audit.py delegates to repo.py	§2.5	agents/audit.py	Code review	Complete

⸻

Usage Rule
	•	Every PR MUST update this file if it adds, completes, or modifies a requirement.
	•	A PR that touches code without updating traceability must be rejected.

