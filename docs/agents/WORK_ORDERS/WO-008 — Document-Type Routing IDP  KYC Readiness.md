Context

Platform must support KYC documents beyond financial reports.

Objective

Introduce explicit document-type routing without breaking financial logic.

Supported Document Types
	•	narrative_financial
	•	structured_financial
	•	identity_document
	•	address_proof
	•	receipt
	•	generic_form

Requirements (MUST)
	•	Routing MUST be explicit and logged
	•	Financial documents MUST retain existing behavior
	•	Table atomicity rules MUST still apply

Tests (MUST)
	•	Passport fixture
	•	Utility bill fixture
	•	Annual report regression test

SPEC Mapping
	•	§2 Goals
	•	§4.1 Separation of Concerns