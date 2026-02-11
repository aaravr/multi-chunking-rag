Context

Regulated users require transparent reasoning and evidence.

Objective

Produce structured explainability outputs for every answer.

Required Outputs
	•	Evidence map (chunk → page → polygon)
	•	Decision trace (accepted / rejected anchors with reasons)
	•	Confidence grade:
	•	explicit
	•	inferred
	•	not_found

MUST NOT
	•	Mix inferred with explicit
	•	Hide uncertainty

Tests (MUST)
	•	Snapshot explainability output
	•	Verifier failure explanation test

SPEC Mapping
	•	§4.2 Lineage
	•	§10 Acceptance Criteria