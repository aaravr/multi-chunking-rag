# WO-00X — Tighten Items-of-Note Itemization Detection (Prevent LCR “Reconciliation Reference” Anchors)

## Context
We have a recurring regression for the coverage query:

> “List the items of note affecting 2024 net income and the aggregate impact”

The system correctly rejects:
- front matter glossary references (`front_matter_reference`)
- tables as anchors (`reject_chunk_type`)
- generic “Adjusted measures are non-GAAP…” definitions (`reject_adjusted_measures_definition`)

However, it still sometimes **accepts** an anchor in LCR/ratio contexts that contains *meta-text* like:
- “the calculation of adjusted measures is adjusted to exclude the impact of items of note…”
- “for additional information and a reconciliation of reported results to adjusted…”

This anchor is invalid because it **does not contain the reconciliation** and **does not enumerate any items or impacts**. It is a *pointer/reference*, not a disclosure.

### Root Cause (Observed)
The acceptance path is being triggered because the “itemization detector” (e.g., `_has_multiple_labeled_numbers(text)` or equivalent) is **too permissive** and can treat:
- ratio/footnote numbers (e.g., “(12)”)
- years (e.g., 2020/2021/2022)
- guideline references
as “labeled numeric items,” allowing LCR meta-paragraphs to look like they contain “items”.

This causes incorrect anchoring and prevents correct section targeting to the MD&A “Items of note” reconciliation.

---

## Governance / Constraints
This is a **SPEC-governed change**.

- MUST comply with `SPEC.md`, `SPEC_ADDENDUM.md`, and `hardening.md`.
- NO embedding model changes (ModernBERT, dim=768, CPU-only).
- No weakening of verifier strictness.
- No UI changes required.
- No new external dependencies.

---

## Goal
**Permanently prevent** “adjusted measures / ratio-definition / reconciliation-reference-only” text from being accepted as the anchor for `coverage.numeric_list` items-of-note queries.

---

## Requirements

### R1 — Tighten “itemization / labeled number” detection (MUST)
Modify the itemization detector used during anchor acceptance for `coverage_type="numeric_list"` (items_of_note intent) so that it only counts *financial item impacts*, not arbitrary numbers.

**New rule:**
A “labeled numeric item” must include at least one strong financial-impact signal, such as:
- Currency patterns: `$`, `C$`, `US$`
- Magnitude words: `million|billion`
- Parentheses negatives: `($123)`
- “after tax”, “pre-tax”, “(after tax)”
- Optional: explicit “impact on net income” phrasing

**Explicitly DO NOT count as numeric items:**
- footnote markers like `(12)` or `(6)`
- pure ratio context numbers like “118%”
- standalone years (2020/2021/2022)
- guideline references that include numbers but not money impacts

**Implementation guidance:**
Prefer a small helper (names may vary by codebase):
- `_looks_like_financial_impact_number(text)` and/or
- `_extract_financial_impact_mentions(text)` returning count
Then require:
- `>= 2` distinct labeled financial-impact mentions OR explicit “Aggregate impact” phrase.

---

### R2 — Enforce “reconciliation-reference-only” rejection (MUST)
For items-of-note numeric-list queries, if a candidate contains meta-reference phrases like:
- “for additional information”
- “see the reconciliation”
- “are adjusted to exclude the impact of items of note”
- “reconciliation of reported results to adjusted”

Then it MUST be rejected **unless** either:
- the candidate itself contains enumerated item impacts (as per R1), OR
- an explicit “Aggregate impact” line is present (or equivalent), OR
- (optional) a very small neighbor window in the same section (e.g., next 1–2 narrative chunks) contains the enumerated items/aggregate and is merged into the candidate evidence set.

If rejected, record reason code:
- `reject_reconciliation_reference_only`

---

### R3 — Debug transparency (MUST)
Ensure debug output includes, per anchor candidate:
- chunk_id
- chunk_type
- heading_path / section_id
- reasons (including `reject_reconciliation_reference_only`)
- a short indicator of why itemization passed/failed (e.g., `impact_number_hits=0`)

This is required for diagnosability and hardening.

---

## Tests (MUST)

### T1 — Unit regression: reject LCR reconciliation-reference-only
Create a deterministic unit test file:

**`tests/test_items_of_note_anchor_rejects_reconciliation_reference_only.py`**

It must build a minimal candidate set in-memory with:

1) BAD candidate (LCR/ratio context, meta reference):
- heading_path includes “LCR” or ratio-like section naming
- contains “adjusted measures … exclude the impact of items of note … for additional information and a reconciliation…”
- contains only footnote numbers/years/ratios (no currency impacts)

2) GOOD candidate (MD&A items of note disclosure):
- contains “Items of note affecting 2024 net income…”
- contains at least two item lines with currency impacts
- contains explicit aggregate impact OR enough to derive it

Assertions:
- router classifies query as `coverage.numeric_list`
- anchor selection returns GOOD chunk_id
- BAD LCR candidate appears in `anchor_decisions` with reason including:
  - `reject_reconciliation_reference_only` OR failing itemization signals
- no table chunk is accepted as anchor

---

### T2 — Unit regression: tighten itemization detector
Add a focused unit test (same file or separate) verifying the detector:

- returns **false** for:
  - “LCR (12) … 118% … (6) Adjusted measures are non-GAAP…”
  - “Ratios for 2020, 2021 and 2022 reflect…”
- returns **true** for:
  - “FDIC special assessment ($0.3 billion after tax) … acquisition-related intangibles ($0.1 billion after tax) …”
  - includes “Aggregate impact … ($0.4 billion) …”

---

## Documentation / Hardening (MUST)
1) Update `docs/decisions.md`:
   - Decision: “Strengthen itemization detection to prevent ratio/meta anchors”
   - Alternatives considered: broader lexical anchors; pure vector; larger section search
   - Consequences: fewer false positives; safer refusals; stronger auditability

2) Update `docs/traceability.md` mapping:
   - SPEC §8 Retrieval correctness → anchor selection + itemization detector + tests
   - SPEC §4.2 lineage/debug transparency → anchor_decisions reasons + tests

3) Add a critical code comment at the decision boundary (per `hardening.md`):
   - “Reconciliation reference ≠ reconciliation disclosure — reject meta anchors.”

---

## Acceptance Criteria
This WO is complete only if:

- The query “List the items of note affecting 2024 net income and the aggregate impact”:
  - does **not** anchor to LCR/ratio “adjusted measures” meta-text
  - anchors to an MD&A “items of note” disclosure chunk if present
  - otherwise returns a correct “not found” response with searched sections/pages (no hallucination)

- Tests T1 and T2 pass in CI and locally.

- Debug output clearly shows `reject_reconciliation_reference_only` for the LCR meta candidate.

---

## Deliverables
- Code changes: tightened itemization detection + rejection rule + debug enrichment
- Tests: T1 + T2
- Documentation: `docs/decisions.md` + `docs/traceability.md` updates
- PR summary mapping changes to SPEC sections + “knowledge hardening report” per hardening.md