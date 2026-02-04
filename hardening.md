KNOWLEDGE HARDENING CHECKLIST (MANDATORY)

Any time you discover, confirm, or rely on a non-obvious behavior, failure mode, edge case, heuristic, routing rule, or design trade-off during implementation, you MUST “harden” that knowledge into the repository so it cannot be lost.

Definition: “Knowledge hardening” means converting learnings into durable artifacts in ALL of the following locations (as applicable):

1) SPEC.md / SPEC_ADDENDUM.md
   - If the learning changes required runtime behavior, adds a new query type, changes routing, changes required metadata, or introduces a new invariant → update SPEC or SPEC_ADDENDUM.
   - If the learning is optional/experimental → document as SHOULD with a feature flag.

2) docs/decisions.md
   - Every meaningful design choice MUST be recorded with:
     Context → Decision → Consequences → Alternatives considered.
   - Include explicit “why this exists” rationale for future maintainers.

3) docs/traceability.md
   - Add/modify rows linking:
     Requirement → Spec ref → Implementation files → Tests → Status.

4) Regression tests (mandatory for hard-won learnings)
   - If the learning was discovered through a bug, debugging session, or regression:
     you MUST add a test that fails without the fix.
   - Prefer integration tests for end-to-end failures (routing → retrieval → synthesis → citations).
   - If full PDFs are too heavy for CI, add a deterministic fixture extracted from the relevant section and document how it was derived.

5) Critical code comments at decision boundaries
   - Add short comments ONLY at branch points where future “simplification” would reintroduce bugs.
   - Comments must state the specific failure prevented (e.g., “Items of note ≠ Note 12 (financial statements) — avoid table anchors”).

ENFORCEMENT RULES
- A PR is invalid if it changes behavior without updating traceability.
- A PR is invalid if it fixes a regression without adding/adjusting a regression test.
- A PR is invalid if it introduces new heuristics/routing without documenting in decisions.md.
- If you are unsure whether a change is a “learning”, treat it as one and harden it.

REQUIRED OUTPUT (in every response that includes code changes)
Along with code, you MUST output a “Knowledge Hardening Report” with:

A) Learnings discovered (bullet list)
B) Where hardened:
   - SPEC/SPEC_ADDENDUM updates (file + section)
   - decisions.md entry title
   - traceability rows updated
   - tests added/updated (names)
   - key code comment locations
C) How to validate:
   - exact commands to run tests
   - exact UI steps for any manual acceptance demo