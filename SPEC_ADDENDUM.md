A1. Query Intent Routing (MUST)

The system MUST classify user queries into:
	•	LocationQuery: references page number (“page 22”, “p. 22”, “on page”)
	•	CoverageQuery: requests exhaustive lists/sections (“list all”, “all significant events”, “summarize section”)
	•	SemanticQuery: normal QA

LocationQuery behavior (MUST):
	•	Retrieval MUST filter candidate chunks by page_number(s) before scoring.
	•	Answer MUST cite evidence from the requested page(s) if present.

CoverageQuery behavior (MUST):
	•	Retrieval MUST locate an anchor chunk and expand scope to all chunks sharing the same heading_path or section_id (ordered by document position).
	•	Answer MUST be derived from expanded scope, not top-K alone.

CoverageQuery subtypes (MUST):
	•	list: exhaustive list extraction from expanded scope (e.g., “list all significant events”).
	•	attribute: numeric span extraction (e.g., “aggregate range of losses”).
	•	numeric_list: list of items with numeric impacts (e.g., “items of note…net income…aggregate impact”).
	•	pointer: section/note reference with citation (e.g., “where can I find…”).

SemanticQuery behavior (MUST):
	•	Default top-K vector retrieval is allowed.

A2. Deterministic Heading Path (MUST)

Each chunk MUST include:
	•	heading_path (e.g., CIBC_2024_AR/MD&A/Significant events)
	•	section_id (coarser grouping)

A3. Table Chunk Atomicity (MUST)

If a table is detected (native or DI):
	•	it MUST be stored as a single atomic chunk (Markdown with caption/footnotes where available)
	•	it MUST NOT be split by recursive or sentence chunking
	•	polygons MUST map to the table region (minimum)

A4. Optional Precision Enhancements (SHOULD)
	•	Hybrid retrieval (lexical + dense) fused via RRF SHOULD be implemented for finance/legal IDs.
	•	A verifier step (Extractor + Auditor YES/NO) SHOULD be used for numeric/attribute extraction.

A5. Regression Scenarios (MUST)

The system MUST pass these scenarios (as integration tests or scripted demos):
	•	“page X …” query returns evidence from that page
	•	“list all significant events” returns complete list for that section
	•	table-based attributes produce citations with polygons that highlight the correct region

A6. Document-Level Fact Caching (MUST SUPPORT; ENABLED BY CONFIG)

The system MUST SUPPORT deterministic, citation-backed document-level metadata facts with lineage.
Fact extraction/caching MAY be enabled/disabled via configuration. When disabled, behavior falls back to the document-metadata retrieval plan.

Facts to extract (initial scope):
	•	default_currency
	•	reporting_period
	•	accounting_framework
	•	units
	•	consolidation_basis

Data contract (MUST):
Each fact MUST store:
	•	value (string or null)
	•	status (found | not_found | ambiguous)
	•	confidence (float 0–1)
	•	source_chunk_id (nullable if not_found/ambiguous)
	•	page_numbers (searched pages when not_found/ambiguous)
	•	polygons (nullable when not_found/ambiguous)
	•	evidence_excerpt (nullable when not_found/ambiguous)

Storage (MUST):
Persist in a new table document_facts keyed by (doc_id, fact_name).

Extraction rules (MUST):
	•	Facts MUST be derived only from explicit evidence.
	•	default_currency MUST NOT be inferred from transactional tables.
	•	If multiple conflicting candidates exist, set status=ambiguous and do not choose a value.

Retrieval behavior (MUST):
	•	For document-metadata queries, the system MUST consult document_facts first.
	•	If status=not_found/ambiguous OR record missing, perform a front-matter-only locate step and return:
		(a) explicit value with citation, OR
		(b) “not found/ambiguous” + searched pages, with no inference.
	•	Front-matter locate MUST search pages 1–10 and the first occurrence of:
		“Consolidated financial statements”, “Basis of presentation”,
		“Significant accounting policies”, “Presentation currency”, “Functional currency”.

Lineage (MUST):
Each found fact MUST include full lineage per SPEC §4.2.
