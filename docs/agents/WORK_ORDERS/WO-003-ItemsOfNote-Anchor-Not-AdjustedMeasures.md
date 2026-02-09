SYSTEM: You are the Sovereign Coding Agent working inside the repository /mnt/data/multi-chunking-rag-main.
This is a SPEC-governed system. You MUST comply with SPEC.md, SPEC_ADDENDUM.md, and hardening.md.
If code behavior and SPEC.md disagree, SPEC.md wins (unless SPEC.md is updated explicitly).

WORK ORDER: WO-XXX — Prevent “Adjusted measures ≠ Items of note” anchoring regression (permanent)

GOAL
Fix the anchor-selection regression where the query:
  “List the items of note affecting 2024 net income and the aggregate impact”
incorrectly anchors to “Adjusted measures are non-GAAP…” definition text (e.g., LCR/ratio sections) instead of the MD&A “Items of note” reconciliation.
Make this permanent with a unit test.

SCOPE (NO SPEC CHANGES)
- This is a bugfix + regression hardening.
- Do NOT change embedding model or late chunking behavior.
- Do NOT weaken verifier strictness.
- Do NOT introduce Docling or new external dependencies.
- CPU-only.

REPRO (from debug)
The current system accepts an anchor in heading_path like “.../LCR (12)/118 %” where snippet includes:
  “Adjusted measures are non-GAAP measures…”
This is not an “Items of note” reconciliation section.

REQUIREMENTS
R1. Anchor rejection rule (MUST)
For coverage queries with coverage_type="numeric_list" that match the “items_of_note” intent:
- Reject candidate anchors that are “Adjusted measures / non-GAAP definitions” without an explicit reconciliation/list.
- Specifically reject if:
  a) candidate text contains phrases like:
     - “Adjusted measures are non-GAAP”
     - “Adjusted measures are used to assess”
     - “Common shareholders’ equity divided by…”
  AND
  b) candidate does NOT contain any “items of note” reconciliation signals like:
     - “Items of note”
     - “Specified items”
     - “Reconciliation” + “net income”
     - “impact on reported net income”
  AND/OR
  c) candidate heading_path/section_id indicates a ratio/metric definition context, e.g. contains:
     - “LCR”, “ratio”, “capital”, “common shareholders’ equity”, “ROE”, “efficiency ratio”
Result: these candidates must never be accepted as anchor for items_of_note numeric_list.

R2. Anchor acceptance strengthening (MUST)
For items_of_note numeric_list anchors, require minimum positive evidence:
- Candidate must contain at least ONE of:
  - the phrase “Items of note”
  - OR a reconciliation phrase mentioning net income (reported/adjusted) plus “items”/“specified items”
AND must contain either:
  - >= 2 labeled numeric item patterns (e.g., “FDIC … ($X)” AND “acquisition … ($Y)”), OR
  - “Aggregate impact” (or equivalent) explicitly stated
This prevents accepting generic adjusted-measures definitions.

R3. Debug transparency (MUST)
When rejecting an “Adjusted measures definition” candidate, record an explicit reason code in debug:
- “reject_adjusted_measures_definition” (or similar, consistent naming)
Also ensure front-matter/glossary reasons remain.

R4. Unit test (MUST) — permanent regression
Add a deterministic unit test that:
- Builds a small in-memory candidate set with 3 chunks:
  1) BAD: LCR/ratio definitions containing “Adjusted measures are non-GAAP…”
  2) BAD: front-matter glossary reference (“see the Glossary…”)
  3) GOOD: MD&A “Items of note” paragraph containing at least two items with amounts and an aggregate impact
- Asserts:
  - router classifies query as coverage/numeric_list
  - anchor selector returns the GOOD chunk_id
  - BAD chunks are rejected with reasons including:
    - front_matter_reference
    - reject_adjusted_measures_definition (for the LCR chunk)
File name:
  tests/test_items_of_note_anchor_not_adjusted_measures.py

R5. Knowledge hardening (MUST)
- Add an entry to docs/decisions.md describing:
  “Preventing false anchors: adjusted measures definitions ≠ items of note reconciliation”
- Update docs/traceability.md mapping:
  SPEC §8 (Retrieval) + SPEC §4.2 (Lineage/debug transparency) → anchor selection code + new unit test
- Add a short critical code comment at the decision boundary (per hardening.md):
  “Adjusted measures (ratio definitions) ≠ Items of note (MD&A reconciliation) — reject as anchor.”

IMPLEMENTATION GUIDANCE
- Keep the change surgical: only touch anchor selection / coverage plan anchor logic (likely retrieval/router.py and/or retrieval/anchor.py).
- Prefer a small function:
  is_items_of_note_anchor_candidate(chunk) -> (bool, reasons[])
  with explicit negative-context checks and positive-evidence checks.
- Do NOT hardcode document-specific page numbers. Use textual/heading heuristics only.

VALIDATION (MUST PROVIDE IN PR SUMMARY)
- pytest -k "items_of_note_anchor_not_adjusted_measures"
- Show one debug example proving the LCR candidate is rejected with the new reason code.
- Include Knowledge Hardening Report:
  (a) learning, (b) where hardened, (c) tests added, (d) how to validate.

DELIVERABLES
- Code change implementing R1–R3
- Unit test implementing R4
- docs/decisions.md + docs/traceability.md updates (R5)
- PR summary mapping to SPEC sections + Knowledge Hardening Report

DO NOT
- change embedding model
- weaken verifier
- infer “items of note” from generic adjusted measures text
- accept table chunks as anchors for items_of_note numeric_list