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
