Context

Hybrid retrieval currently rebuilds BM25 indexes at query time, causing unacceptable latency and instability.

Objective

Make hybrid retrieval deterministic, cached, and performant without changing retrieval semantics.

Requirements (MUST)
	•	BM25 index MUST be built once per document set.
	•	Index MUST be cached (memory or disk).
	•	Query-time retrieval MUST NOT rebuild the index.
	•	Hybrid search (BM25 + vector + RRF) MUST preserve ranking equivalence.

Non-Goals
	•	No learned rankers
	•	No infra services required (Redis optional, not mandatory)

Implementation Notes
	•	Introduce BM25IndexManager
	•	Cache keyed by (doc_id, corpus_version)
	•	Fail fast if index missing or stale

Tests (MUST)
	•	CPU baseline vs cached retrieval equivalence
	•	Latency regression test (<200ms per query)

SPEC Mapping
	•	§8 Retrieval & Answering
	•	§11 Change Management
