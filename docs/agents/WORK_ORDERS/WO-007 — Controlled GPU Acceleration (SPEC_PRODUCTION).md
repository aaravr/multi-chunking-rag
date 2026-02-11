Context

RTX-4090 available for acceleration.

Objective

Enable GPU acceleration without changing answers.

Allowed GPU Usage
	•	Embedding forward pass
	•	Cross-encoder reranking

Requirements (MUST)
	•	GPU usage gated by config: USE_GPU=true|false
	•	CPU and GPU outputs MUST be equivalent:
	•	anchor chunk_id
	•	retrieved chunk_ids
	•	numeric spans
	•	GPU MUST auto-disable on equivalence failure

Prohibited
	•	GPU-only execution
	•	Silent model swaps

Tests (MUST)
	•	CPU vs GPU anchor equivalence
	•	CPU vs GPU reranker equivalence
	•	Numeric extraction equivalence

SPEC Mapping
	•	SPEC_PRODUCTION.md
	•	§6 Embeddings
	•	§10 Acceptance Criteria
