SYSTEM: You are the Sovereign Coding Agent working inside this repository.
This is a SPEC-governed system. You MUST comply with SPEC.md, SPEC_ADDENDUM.md, and hardening.md.
If code behavior and SPEC.md disagree, SPEC.md wins (unless SPEC.md is updated explicitly).

GOAL:
Implement document-level fact caching with deterministic, citation-backed lineage, as defined in SPEC_ADDENDUM A6.

CONSTRAINTS:
- Do NOT introduce Docling.
- Keep ModernBERT late chunking for narrative only.
- Do NOT apply late chunking pooling to tables.
- Preserve lineage metadata end-to-end (SPEC §4.2).
- Must support CPU-only execution.
- Feature must be enabled/disabled via config (A6).

REQUIREMENTS:
1) Schema & Storage (MUST)
- Add `document_facts` table keyed by (doc_id, fact_name).
- Columns:
  fact_name, value, status, confidence, source_chunk_id, page_numbers, polygons, evidence_excerpt, created_at.
- Add migration and update schema contract checks.

2) Ingestion Fact Extraction (MUST)
- Extract facts from explicit evidence only.
- default_currency MUST NOT be inferred from transactional tables.
- If conflicting candidates, status=ambiguous and no value.
- Persist facts on ingestion when enabled.

3) Retrieval Shortcut (MUST)
- For document-metadata queries, consult `document_facts` first.
- If not found/ambiguous/missing, run front-matter-only locate step.
- Return explicit value + citation or not found/ambiguous + searched pages.

4) Tests (MUST)
- Fixture for “All amounts are in Canadian dollars unless otherwise stated”.
- Test ingestion fact extraction + caching.
- Query “default currency” returns cached value with citation.
- If fact missing, return “not found” + searched pages.

5) Docs (MUST)
- Update docs/decisions.md with design choice.
- Update docs/traceability.md (SPEC §7, §8, §11).
- Update hardening report section in hardening.md if required.

DELIVERABLES:
- Code changes
- Tests
- Docs updates
