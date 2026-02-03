
1. Purpose

This specification defines the non-negotiable behavior, invariants, and interfaces for the Sovereign Intelligent Document Processing (IDP) RAG Proof-of-Concept.

The system must demonstrate:
	•	Accurate extraction from large, complex PDFs (600+ pages)
	•	Evidence-grounded answers with visual highlights
	•	Late Chunking for semantic precision
	•	Selective Azure Document Intelligence (DI) for cost-aware completeness
	•	A design that can scale conceptually to millions of documents

This document is the single source of truth.
All code changes must be traceable to this spec.

⸻

2. Goals

The system MUST:
	1.	Ingest PDFs containing mixed content:
	•	native text
	•	tables
	•	images / scanned pages
	2.	Use policy-based selective DI, not blanket OCR.
	3.	Preserve full lineage from answer → chunk → page → coordinates.
	4.	Support attribute extraction and free-form analytical queries.
	5.	Render clickable evidence highlights on the PDF.
	6.	Run locally on Mac CPU (PoC environment).

⸻

3. Non-Goals

The system explicitly does NOT aim to:
	•	Achieve perfect table cell semantics
	•	Perform full vision reasoning on charts
	•	Be production hardened (auth, HA, SLA)
	•	Replace Azure DI with a pure OSS pipeline (out of scope for PoC)

⸻

4. Architectural Invariants (MUST NEVER BREAK)

4.1 Separation of Concerns

The following layers MUST remain distinct:

Layer	Responsibility
UI	Streamlit only
Ingestion	Page triage, DI orchestration
Extraction	Native PDF / DI parsing
Canonicalization	Text + coordinate normalization
Embedding	Late chunking + vector creation
Storage	DB schema, persistence
Retrieval	Vector search
Synthesis	LLM answer generation
Grounding	Highlight polygon mapping

No layer may bypass another.

⸻

4.2 Deterministic Lineage (First-Class)

Every stored chunk MUST contain:
	•	doc_id
	•	page_number(s)
	•	char_start, char_end
	•	polygons (bounding boxes or polygons)
	•	source_type (di or native)
	•	macro_id, child_id
	•	embedding_model, embedding_dim

If any of these are missing, the system is invalid.

⸻

4.3 Embedding Invariant
	•	Embedding model: nomic-ai/modernbert-embed-base
	•	Dimension: 768
	•	Device: CPU
	•	Late chunking MUST pool embeddings after a global attention pass.

No early chunking shortcuts are allowed.

⸻

5. Selective Azure DI Policy

5.1 Principle

DI usage must be:
	•	Defensible
	•	Auditable
	•	Explainable per page

Selective DI is mandatory to control cost at scale.

⸻

5.2 Page Triage Metrics (Computed Locally)

For every page:
	•	text_length
	•	text_density
	•	image_coverage_ratio
	•	layout_complexity_score (heuristic)

⸻

5.3 Decision Rules (Example)

A page MUST be sent to DI if any of the following are true:
	•	Very low or no extractable text
	•	High image coverage (likely scanned)
	•	High table likelihood / layout complexity

Otherwise:
	•	Native extraction is allowed

⸻

5.4 Triage Auditability

For each page, persist:
	•	triage_metrics (json)
	•	triage_decision
	•	reason_codes
	•	di_json_path (if applicable)

This enables post-hoc justification of DI cost.

5.5 Temporary DI Disable (Testing Only)

DI may be temporarily disabled via configuration for local testing.
When disabled:
	•	triage_decision remains unchanged
	•	reason_codes MUST include "di_disabled"
	•	di_json_path remains null


⸻

6. Late Chunking Specification

6.1 Macro Chunking
	•	Chunk text by tokens
	•	Target max length: configurable (default 8192)
	•	Overlap: configurable

⸻

6.2 Global Attention Pass
	•	Run ModernBERT once per macro chunk
	•	Capture last_hidden_state (token embeddings)

⸻

6.3 Child Chunk Pooling
	•	Define child spans (sentences or ~256 tokens)
	•	Use tokenizer offset mappings
	•	Pool token embeddings within child span
	•	Mean pooling is acceptable

⸻

6.4 Guarantees
	•	Identical labels in different contexts must embed differently
	•	Global context must influence local retrieval units

⸻

7. Database Contract

7.1 Required Tables

documents
	•	doc_id (uuid)
	•	filename
	•	sha256
	•	page_count
	•	created_at

pages
	•	doc_id
	•	page_number
	•	triage_metrics
	•	triage_decision
	•	reason_codes
	•	di_json_path
	•	created_at

chunks
	•	chunk_id
	•	doc_id
	•	page_numbers
	•	macro_id
	•	child_id
	•	text_content
	•	char_start
	•	char_end
	•	polygons
	•	source_type
	•	embedding_model
	•	embedding_dim
	•	embedding (vector(768))
	•	created_at

⸻

7.2 Indexing
	•	HNSW index on chunks.embedding
	•	B-tree index on doc_id

⸻

8. Retrieval & Answering

8.1 Retrieval
	•	Embed query
	•	Retrieve top-K chunks (default K=3)
	•	Retrieval must return lineage metadata

⸻

8.2 Synthesis
	•	Use OpenAI API as decoder
	•	LLM MUST:
	•	use only retrieved content
	•	return answer + citations
	•	never hallucinate unseen facts

⸻

9. UI Requirements

9.1 Features
	•	PDF upload
	•	Process button
	•	Triage summary
	•	Attribute buttons (CET1 Ratio, Net Income, Risk Exposure)
	•	Free-form query input

⸻

9.2 Grounding
	•	Every citation must be clickable
	•	Clicking highlights exact polygons on the PDF
	•	Highlighting is mandatory for PoC acceptance

⸻

10. Acceptance Criteria

The system is considered complete only if:
	1.	A 600-page financial PDF ingests without crashing
	2.	Triage decisions are stored and inspectable
	3.	Queries return:
	•	a concise answer
	•	at least one grounded citation
	4.	Clicking a citation highlights evidence on the PDF
	5.	All lineage invariants are preserved end-to-end

⸻

11. Change Management Rules
	•	Any schema change → update this spec + migrations
	•	Any ingestion policy change → update Section 5
	•	Any embedding change → update Section 6
	•	All architectural decisions → log in docs/decisions.md

⸻

12. Final Authority

If code behavior and SPEC.md disagree,
SPEC.md wins.

