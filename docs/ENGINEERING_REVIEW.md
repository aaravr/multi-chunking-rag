# Engineering Review — multi-chunking-rag IDP Platform

**Date:** 2026-03-14
**Scope:** Full architectural review, schema audit, feedback subsystem convergence, packaging, and remediation plan.

---

## SECTION 1 — EXECUTIVE REVIEW

### Current State

The platform is an advanced, ambitious enterprise-IDP / RAG system with substantial architecture spanning agent decomposition, late chunking, hybrid retrieval, evidence-grounded synthesis, lineage tracking, and a layered feedback/retraining pipeline. It has evolved from a validated PoC through multiple implementation phases (1–5a complete) into a system that genuinely attempts enterprise-grade behavior.

### What Is Strong

1. **Architecture decomposition.** Clear layer separation: ingestion → embedding → retrieval → synthesis → grounding → agents → storage → feedback. Each layer has a defined responsibility.
2. **Lineage mindset.** Chunk-level lineage with page numbers, char offsets, polygons, heading paths, section IDs, and macro/child span relationships. Evidence grounding is carried through agent hops.
3. **Chunking and retrieval.** Late chunking with macro/child spans, BM25 hybrid search, cross-encoder reranking, section-aware expansion, and intent-based query routing. Richer than typical RAG implementations.
4. **Test surface.** 47 test files covering contracts, agents, routing, coverage, schema, regression, and the feedback loop subsystem.
5. **Feedback loop design.** The newer `feedback_loop/` subsystem is directionally correct: typed models, 6-rule attribution engine, boundary-safe training isolation, layer-specific training row generation, and model lifecycle management.
6. **Contract discipline.** 85 typed dataclasses/models across three contract namespaces (agents: 50, core: 10, feedback_loop: 25).

### What Prevents Enterprise-Grade Status

1. **Duplicate feedback/retraining architecture.** Two parallel implementations coexist without a declared canonical path.
2. **Schema/repository/contract drift.** Repository code inserts fields not validated by the schema contract checker. Base schema lacks fields that repo.py depends on.
3. **Missing dependency declarations.** `langchain-core`, `langgraph`, and `scikit-learn` are imported but not declared in `requirements.txt`.
4. **Documentation drift.** GAP_ANALYSIS.md claims Extractor/Transformer are unimplemented despite complete implementations. README still frames the project as a PoC skeleton.
5. **In-memory feedback services.** The feedback loop pipeline uses `InMemory*` service implementations that lose state on restart.
6. **BoundaryKey permissiveness.** Empty `division` and `jurisdiction` are allowed, weakening the stated isolation principle.
7. **Synthetic trace fallback.** Missing prediction traces are silently replaced with minimal synthetic traces rather than quarantined.

---

## SECTION 2 — ARCHITECTURAL FINDINGS

### 2.1 Agent Decomposition

The agent layer contains 11 specialized agents plus supporting infrastructure (model gateway, message bus, prompt registry, audit, evaluation). This is well-structured. Each agent has typed input/output contracts via `agents/contracts.py`.

**Retained:** All current agents.
**No changes needed** to the decomposition itself.

### 2.2 Duplicate Feedback/Retraining Paths

This is the single biggest architectural issue.

**Old agent-level path** (`agents/feedback_agent.py`, `agents/retraining_agent.py`):
- In-memory TTL cache for feedback storage (lost on restart)
- Fixed routing to 2-3 targets (classifier, preprocessor, retriever)
- No multi-layer attribution
- No training row generation
- No boundary isolation
- No model lifecycle management
- DB tables: `feedback_entries`, `retraining_events`, `chunking_outcomes` (migration 007)

**New feedback_loop subsystem** (`feedback_loop/`):
- 9 abstract service interfaces
- 25 Pydantic v2 typed models
- 6-rule attribution engine mapping feedback → impacted decision layers
- 5 layer-specific training row builders with reward signals
- Boundary-safe isolation with approval governance
- Model lifecycle: shadow → canary → approved → rollback
- DB tables: 14 tables across prediction traces, feedback events, attributions, training rows (5 layer-specific), model candidates, evaluation reports, boundary approvals, retraining jobs (migration 009)

**Verdict:** The new `feedback_loop/` subsystem is categorically stronger in every dimension. It must become the canonical path. The old agent-level path must be deprecated.

### 2.3 Contract Fragmentation

Three contract namespaces exist:

| Namespace | Count | Technology | Scope |
|-----------|-------|-----------|-------|
| `agents/contracts.py` | 50 dataclasses | `@dataclass(frozen=True)` | Agent communication, query routing, synthesis, extraction |
| `core/contracts.py` | 10 dataclasses | `@dataclass(frozen=True)` | Storage records, retrieval results, triage |
| `feedback_loop/models.py` | 25 models | Pydantic v2 | Feedback, attribution, training rows, model lifecycle |

**Bridge point:** `core.contracts.RetrievedChunk` is used by both core and agents (via `RankedEvidence.chunks`).

**Assessment:** The three namespaces are largely non-overlapping and serve different domains. This is acceptable isolation, not harmful fragmentation. The `FeedbackEntry`/`FeedbackResult` types in `agents/contracts.py` (lines ~580-610) overlap with `feedback_loop/models.py` and should be deprecated in favor of the feedback_loop types.

### 2.4 State Model Gaps

The platform lacks an explicit canonical state model for:
- **Case/query state:** Implicit in orchestrator ReAct loop, not formally defined
- **Document state:** Partially captured via `document_facts` and classification, but no lifecycle enum
- **Feedback state:** Defined in feedback_loop models but not connected to agent contracts
- **Training state:** Defined in feedback_loop but not integrated with working memory

**Recommendation:** Define a `DocumentLifecycle` enum and `QueryState` enum in `core/contracts.py` as the canonical state model. Defer full state machine formalization to Phase 6 (Memory System).

---

## SECTION 3 — STORAGE / SCHEMA FINDINGS

### 3.1 Schema Drift: Base Schema vs. Migrations

**Root cause:** The base `schema.sql` defines the PoC-era tables. Enterprise fields were added via migrations (004+). Repository code (`repo.py`) assumes migrations have been applied but the schema contract checker (`schema_contract.py`) only validates base schema fields.

**Specific drift:**

| Table | Field | In schema.sql | In migration | In repo.py INSERT | In schema_contract |
|-------|-------|--------------|-------------|-------------------|-------------------|
| documents | document_type | No | 004 line 6 | Yes (line 13) | **No** |
| documents | classification_label | No | 004 line 7 | Yes (line 13) | **No** |
| documents | updated_at | No | 004 line 11 | Yes (line 260) | **No** |
| chunks | document_type | No | 004 line 14 | Yes (line 168) | **No** |
| chunks | classification_label | No | 004 line 15 | Yes (line 169) | **No** |
| chunks | heading_path | Yes (line 36) | — | Yes | **No** |
| chunks | section_id | Yes (line 37) | — | Yes | **No** |

### 3.2 ON CONFLICT Constraint Verification

| Table | ON CONFLICT target | Unique constraint exists? | Status |
|-------|-------------------|--------------------------|--------|
| documents | `(doc_id)` | PK on doc_id | **OK** |
| pages | `(doc_id, page_number)` | PK on (doc_id, page_number) | **OK** |
| chunks | `(doc_id, macro_id, child_id)` | UNIQUE added in migration 003 | **OK** |
| document_facts | `(doc_id, fact_name)` | PK on (doc_id, fact_name) | **OK** |
| classification_embeddings | `(embedding_id)` | PK on embedding_id (migration 006) | **OK** |

**sha256 UNIQUE constraint:** CLAUDE.md specifies `sha256 (unique)` but `schema.sql` line 7 defines `sha256 TEXT NOT NULL` without a UNIQUE constraint. This allows duplicate document ingestion. **Must be fixed.**

### 3.3 Schema Contract Checker Gaps

`schema_contract.py` validates only 4 tables (documents, pages, chunks, document_facts) and only base-schema fields. It does not validate:
- Enterprise fields (document_type, classification_label, updated_at)
- Lineage fields (heading_path, section_id) — critical for the platform's core design
- Any enterprise tables from migrations 004-009
- Classification tables from migrations 005-006
- Feedback tables from migrations 007, 009

### 3.4 Remediation Actions

1. **Add `heading_path` and `section_id` to chunks validation** in schema_contract.py
2. **Add `document_type`, `classification_label` to both documents and chunks validation**
3. **Add UNIQUE constraint on sha256** in schema.sql
4. **Add enterprise table validation** for tables actively used by repo.py

---

## SECTION 4 — FEEDBACK LOOP / RETRAINING FINDINGS

### 4.1 Old Agent-Level Path

**Files:** `agents/feedback_agent.py` (282 lines), `agents/retraining_agent.py` (299 lines)

**Capabilities:**
- Feedback collection with in-memory TTL store
- Message-bus routing to classifier/preprocessor/retriever
- Threshold-based retraining triggers
- SGD partial_fit incremental classifier training
- Pattern pruning for low-accuracy classification patterns

**Limitations:**
- In-memory feedback store (TTLCache, max 50K entries) — data lost on restart
- No boundary isolation — global per-instance
- No multi-layer attribution — fixed routing to 2-3 targets
- No training row generation — only message routing
- No model lifecycle management

### 4.2 New Feedback Loop Subsystem

**Files:** 9 files in `feedback_loop/` (~2,600 lines total)

**Capabilities:**
- 9 abstract service interfaces (pluggable backends)
- 6 deterministic attribution rules mapping feedback → impacted layers
- 5 layer-specific training row builders with mathematical reward signals
- Boundary-safe training isolation: B = (client, division, jurisdiction)
- Cross-boundary sharing governance with approval pairs and sanitization
- Model lifecycle: shadow → canary → approved → rejected → rollback_ready
- 14 DB tables for full persistence (migration 009)

**Limitations:**
- Pipeline currently wired to `InMemory*` service implementations
- BoundaryKey allows empty division/jurisdiction
- Synthetic trace fallback on missing prediction traces

### 4.3 Canonical Path Decision

**The `feedback_loop/` subsystem is the canonical feedback/retraining architecture.**

Rationale:
- Categorically stronger in every dimension (attribution, lineage, boundary safety, training rows, model lifecycle)
- Designed with enterprise requirements (multi-tenant isolation, governance)
- Properly typed with Pydantic v2 models
- Has comprehensive test coverage in `tests/test_feedback_loop.py`

### 4.4 Deprecation Plan for Old Path

1. **Deprecate** `agents/feedback_agent.py` and `agents/retraining_agent.py`
2. **Retain** migration 007 tables (`feedback_entries`, `retraining_events`, `chunking_outcomes`) — they serve different granularity than migration 009 tables and can coexist for backward compatibility
3. **Add deprecation markers** to old agent files directing developers to `feedback_loop/`
4. **Do not delete** the old files yet — mark as deprecated with clear migration guidance

### 4.5 Hardening Actions

1. **BoundaryKey validation:** Require non-empty `client` field. Log warnings on empty `division`/`jurisdiction`.
2. **Synthetic trace quarantine:** Replace silent synthetic trace creation with explicit quarantine semantics — store feedback, mark as `non_trainable`, create remediation event.
3. **Pipeline interface typing:** Change `FeedbackLoopPipeline.__init__` to accept abstract interfaces, not concrete `InMemory*` types.

---

## SECTION 5 — TEST / PACKAGING / OPERABILITY FINDINGS

### 5.1 Dependency Declaration

**Missing from requirements.txt:**

| Package | Used In | Type |
|---------|---------|------|
| `langchain-core` | `agents/classifier_agent.py:46` | Runtime import |
| `langgraph` | `agents/classifier_agent.py:47` | Runtime import |
| `scikit-learn` | `agents/classifier_agent.py:48` | Runtime import |

**Optional dependencies not declared:**

| Package | Used In | Type |
|---------|---------|------|
| `docling` | `ingestion/docling_parser.py` | Optional parser backend |
| `python-docx` | `ingestion/multi_format_parser.py` | Optional format support |
| `openpyxl` | `ingestion/multi_format_parser.py` | Optional format support |
| `beautifulsoup4` | `ingestion/multi_format_parser.py` | Optional format support |
| `fastapi` | `feedback_loop/api.py` | Optional API support |
| `uvicorn` | `feedback_loop/api.py` | Optional API server |

### 5.2 Test Reproducibility

- 47 test files, majority pass in clean environment
- 3 files use `@pytest.mark.skipif` for environment-dependent tests
- No formal separation between unit tests (no external deps) and integration tests (DB, LLM)
- No `pytest.ini` or `pyproject.toml` test configuration with markers

### 5.3 Remediation Actions

1. **Add missing runtime dependencies** to requirements.txt
2. **Create `requirements-optional.txt`** for optional backends (docling, multi-format, feedback API)
3. **Add pytest markers** in `pyproject.toml`: `unit`, `integration`, `external`
4. **Document test execution** in README with marker-based filtering

---

## SECTION 6 — SECURITY / GOVERNANCE FINDINGS

### 6.1 Positive

- Audit log is IMMUTABLE and APPEND-ONLY (design intent)
- LLM calls logged with full prompt/response via Model Gateway
- Boundary-safe learning design in feedback_loop subsystem
- Lineage tracking from chunk to synthesis

### 6.2 Gaps

1. **Tenant isolation:** RBAC tables exist (migration 004: `users`, `document_access`) but no enforcement middleware. Document-level access control is not checked at query time.
2. **PII treatment:** No PII detection or redaction implemented. Compliance agent is planned (Phase 7) but not started.
3. **Retention controls:** No TTL or retention policy on audit logs, feedback events, or training rows.
4. **Secret handling:** API keys loaded from environment variables — acceptable for development, but no vault integration.
5. **Boundary enforcement in old path:** `agents/feedback_agent.py` has zero boundary isolation — all feedback is global.

### 6.3 Recommendations

These are Phase 7+ concerns. For now:
1. Ensure the canonical feedback_loop path has boundary enforcement (it does)
2. Add explicit `non_trainable` quarantine for feedback without valid traces
3. Document security assumptions and gaps in this review

---

## SECTION 7 — P0 / P1 / P2 REMEDIATION PLAN

### P0 — Must Fix Immediately

| # | Action | Files | Impact |
|---|--------|-------|--------|
| P0.1 | Add `heading_path`, `section_id`, `document_type`, `classification_label` to schema_contract.py chunks validation | `storage/schema_contract.py` | Prevents contract validation from silently passing with missing lineage fields |
| P0.2 | Add `document_type`, `classification_label`, `updated_at` to schema_contract.py documents validation | `storage/schema_contract.py` | Aligns contract checker with actual repo.py usage |
| P0.3 | Add UNIQUE constraint on `documents.sha256` in schema.sql | `storage/schema.sql` | Prevents duplicate document ingestion per CLAUDE.md spec |
| P0.4 | Add `langchain-core`, `langgraph`, `scikit-learn` to requirements.txt | `requirements.txt` | Fixes import failures in classifier_agent.py |
| P0.5 | Deprecate `agents/feedback_agent.py` and `agents/retraining_agent.py` with clear markers | `agents/feedback_agent.py`, `agents/retraining_agent.py` | Establishes canonical feedback architecture |
| P0.6 | Document `feedback_loop/` as canonical in CLAUDE.md | `CLAUDE.md` | Single source of truth for developers |

### P1 — Fix Soon

| # | Action | Files | Impact |
|---|--------|-------|--------|
| P1.1 | Harden BoundaryKey: require non-empty `client`, warn on empty division/jurisdiction | `feedback_loop/models.py` | Prevents permissive boundary keys |
| P1.2 | Replace synthetic trace fallback with quarantine semantics | `feedback_loop/pipeline.py` | Preserves auditability |
| P1.3 | Update README.md to reflect enterprise platform status | `README.md` | Accurate project framing |
| P1.4 | Update GAP_ANALYSIS.md to mark Extractor/Transformer as implemented | `docs/GAP_ANALYSIS.md` | Eliminates documentation drift |
| P1.5 | Create `requirements-optional.txt` for optional backends | New file | Clean dependency management |
| P1.6 | Type pipeline constructor to use abstract interfaces | `feedback_loop/pipeline.py` | Enables DB-backed service injection |

### P2 — Fix When Capacity Allows

| # | Action | Files | Impact | Status |
|---|--------|-------|--------|--------|
| P2.1 | Implement DB-backed feedback services replacing InMemory* | `feedback_loop/services.py` | Production persistence | **DONE** |
| P2.2 | Define canonical DocumentLifecycle and QueryState enums | `core/contracts.py` | Formal state model | **DONE** |
| P2.3 | Add pytest markers (unit/integration/external) and pyproject.toml config | `pyproject.toml`, test files | Reproducible test execution | **DONE** |
| P2.4 | Migrate audit.py to use repo.py abstraction | `agents/audit.py`, `storage/repo.py` | Separation of concerns | **DONE** |
| P2.5 | Add enterprise table validation to schema_contract.py | `storage/schema_contract.py` | Complete contract coverage | **DONE** |

---

## SECTION 8 — CANONICAL TARGET ARCHITECTURE

### 8.1 Feedback Loop / Retraining Model

```
Canonical path: feedback_loop/

User/System Feedback
    → FeedbackEvent (Pydantic v2, with BoundaryKey)
    → FeedbackLoopPipeline.process()
        1. Ingest (validate + persist to feedback_events table)
        2. Join (link with prediction_traces table)
        3. Normalize (derive structured ReasonCodes)
        4. Attribute (6-rule engine → ImpactedLayers)
        5. Build (layer-specific TrainingRows)
        6. Guard (boundary validation + sanitization)
        7. Submit (queue for retraining via RetrainingOrchestrator)
    → Model lifecycle: shadow → canary → approved

Deprecated path: agents/feedback_agent.py, agents/retraining_agent.py
    → Retained for backward compatibility during migration
    → Marked with deprecation warnings
    → Will be removed in a future phase
```

### 8.2 Contract Structure

```
core/contracts.py        — Storage records, retrieval results, triage (10 types)
agents/contracts.py      — Agent communication, all agent I/O contracts (50 types)
feedback_loop/models.py  — Feedback, attribution, training, model lifecycle (25 types)

Bridge: core.contracts.RetrievedChunk ↔ agents.contracts.RankedEvidence
Deprecated: agents.contracts.FeedbackEntry/FeedbackResult (use feedback_loop.models)
```

### 8.3 Persistence Expectations

| Layer | InMemory (tests) | DB-backed (production) | Status |
|-------|-----------------|----------------------|--------|
| Feedback ingestion | `InMemoryFeedbackIngestionService` | `PostgresFeedbackIngestionService` | **DONE** |
| Trace join | `InMemoryTraceJoinService` | `PostgresTraceJoinService` | **DONE** |
| Retraining orchestration | `InMemoryRetrainingOrchestrator` | `PostgresRetrainingOrchestrator` | **DONE** |
| Training row accumulation | In-memory (within orchestrator) | In-memory + DB job metadata | **Partial** |
| Model promotion | `InMemoryModelPromotionController` | In-memory (DB promotion planned) | **Planned** |
| Boundary approvals | In-memory set | In-memory (DB approvals planned) | **Planned** |

---

## SECTION 9 — REFACTOR / MIGRATION PLAN

### 9.1 Migration Steps

1. **Phase A (P0 — this PR):**
   - Fix schema_contract.py
   - Fix schema.sql sha256 constraint
   - Fix requirements.txt
   - Add deprecation markers to old feedback agents
   - Update CLAUDE.md with canonical feedback path
   - Harden BoundaryKey
   - Replace synthetic trace with quarantine

2. **Phase B (P1 — next sprint):**
   - Update README.md and GAP_ANALYSIS.md
   - Create requirements-optional.txt
   - Type pipeline to use interfaces
   - Add pytest markers

3. **Phase C (P2 — subsequent sprint):**
   - Implement DB-backed feedback services
   - Add enterprise table validation
   - Migrate audit.py to repo.py abstraction
   - Define canonical state enums

### 9.2 Deprecation Strategy

```python
# In agents/feedback_agent.py and agents/retraining_agent.py:
import warnings

warnings.warn(
    "This module is deprecated. Use feedback_loop/ subsystem instead. "
    "See docs/ENGINEERING_REVIEW.md Section 4 for migration guidance.",
    DeprecationWarning,
    stacklevel=2,
)
```

### 9.3 Compatibility Strategy

- Old migration 007 tables are retained (they store different granularity data)
- Old agents remain importable but emit deprecation warnings
- New `feedback_loop/` subsystem does not depend on old agents
- No breaking changes to existing API surface

### 9.4 Documentation Updates

| Document | Change |
|----------|--------|
| CLAUDE.md | Add feedback_loop as canonical path, update architecture section |
| README.md | Rewrite to reflect enterprise platform status |
| docs/GAP_ANALYSIS.md | Mark Extractor/Transformer as IMPLEMENTED |
| docs/decisions.md | Record canonical feedback architecture decision |
| docs/traceability.md | Add feedback_loop traceability entries |

---

## SECTION 10 — IMPLEMENTATION RECOMMENDATIONS

### 10.1 schema_contract.py

Add missing fields to REQUIRED_SCHEMA:
- chunks: `heading_path`, `section_id`, `document_type`, `classification_label`
- documents: `document_type`, `classification_label`

### 10.2 schema.sql

Add after line 7: `UNIQUE` constraint on sha256 or create a separate unique index.

### 10.3 requirements.txt

Add:
```
langchain-core>=0.3.0
langgraph>=0.2.0
scikit-learn>=1.3.0
```

### 10.4 feedback_loop/models.py

Add `@field_validator("client")` to BoundaryKey requiring non-empty string.

### 10.5 feedback_loop/pipeline.py

Replace synthetic trace creation (lines 89-97) with:
- Store feedback as `non_trainable`
- Return PipelineResult with quarantine flag
- Skip attribution/training row generation

### 10.6 feedback_loop/pipeline.py constructor

Change type hints from `InMemory*` to abstract interfaces from `feedback_loop/interfaces.py`.

### 10.7 agents/feedback_agent.py, agents/retraining_agent.py

Add module-level deprecation warnings.

---

## SECTION 11 — RISKS IF NOT FIXED

1. **Schema drift → runtime failures.** If schema_contract.py passes but repo.py inserts fields that don't exist, silent data loss or insert failures occur in production.
2. **Duplicate feedback paths → developer confusion.** New engineers will not know which path to use, extend, or test. Bug fixes may be applied to the wrong subsystem.
3. **Missing dependencies → broken imports.** `classifier_agent.py` will fail at import time in any clean environment where langchain-core is not transitively installed.
4. **Permissive boundary keys → data contamination.** Training rows with empty boundaries can mix across tenants, violating the stated isolation guarantee.
5. **Synthetic traces → audit failure.** If a synthetic trace enters the training pipeline, the causal link between feedback and training data is broken. Regulators or auditors cannot verify that training data originated from real system behavior.
6. **Documentation drift → stakeholder misalignment.** GAP_ANALYSIS claiming features are unimplemented when they exist wastes planning time and creates false urgency.

---

## SECTION 12 — FINAL VERDICT

### Overall Engineering Assessment

The platform demonstrates genuine architectural ambition backed by substantive implementation. The agent decomposition, lineage tracking, chunking strategies, and feedback loop design are all above-average for an enterprise IDP system. The primary issue is **consistency drift** — the codebase has evolved faster than its validation, documentation, and dependency declarations have been updated.

The path to enterprise-grade is clear and achievable: converge on the canonical feedback architecture, fix schema/contract drift, declare dependencies correctly, and harden boundary enforcement. No major redesign is needed.

### Ratings

| Dimension | Rating |
|-----------|--------|
| As PoC / advanced internal platform | **8/10** |
| As coherent enterprise-grade implementation today | **6/10** |
| As enterprise-grade foundation after hardening | **8.5/10** |

### Bottom Line

The next step is **convergence and hardening**, not feature expansion. The changes recommended in this review are concrete, scoped, and directly actionable by a coding agent.
