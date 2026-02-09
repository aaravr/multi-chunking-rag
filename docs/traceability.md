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
anchor decision debug reasons	§4.2	retrieval/router.py	tests/test_items_of_note_anchor_not_adjusted_measures.py	Complete

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
Document metadata shortcut	§8.1	retrieval/metadata.py; app/poc_app.py	tests/test_document_facts.py	In-Progress
LLM constrained to evidence	§8.2	synthesis/prompt	Hallucination test	Planned
Items-of-note numeric_list anchor hardening	§8.1	retrieval/router.py	tests/test_items_of_note_anchor_not_adjusted_measures.py	Complete


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
Query intent routing + subtypes (list/attribute/numeric_list/pointer)	A1	retrieval/router.py; synthesis/openai_client.py; app/poc_app.py	tests/test_coverage_query.py; tests/test_router_intent_subtypes.py; tests/test_pointer_query_note_reference.py; tests/test_coverage_attribute_range_losses.py	Complete
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

Usage Rule
	•	Every PR MUST update this file if it adds, completes, or modifies a requirement.
	•	A PR that touches code without updating traceability must be rejected.

