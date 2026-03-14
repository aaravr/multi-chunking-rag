# Implementation Status Matrix

**Last updated:** 2026-03-14

This document provides a single-glance view of every major subsystem's implementation status, service tier, and test coverage.

## Status Labels

| Label | Meaning |
|-------|---------|
| **Production-ready** | Fully implemented, tested, and safe for production use |
| **Production-backed (explicit wiring)** | DB-backed implementation exists but requires explicit factory/config wiring |
| **In-memory / test utility** | Functional but ephemeral; suitable only for tests and local dev |
| **Transitional** | Implemented but not yet wired into production pipeline |
| **Deprecated** | Retained for backward compatibility; scheduled for removal |
| **Planned** | Not yet implemented; tracked in MASTER_PROMPT phase plan |

---

## Core Subsystems

| Subsystem | Status | Tests | Notes |
|-----------|--------|-------|-------|
| **Ingestion Pipeline** | Production-ready | `test_ingestion_smoke` (integration) | PyMUPDF default, Docling optional |
| **Late Chunking** | Production-ready | `test_chunking_strategies` | Macro/child spans, 8192-token macro |
| **Embedding (ModernBERT)** | Production-ready | `test_wo010_model_singleton` | Singleton, 768-dim, CPU |
| **Hybrid Retrieval** | Production-ready | `test_bm25_index_manager`, `test_rrf_merge`, `test_rerank` | BM25 + vector + reranking |
| **Router Agent** | Production-ready | `test_router_agent`, `test_intent_router`, `test_router_intent_subtypes` | Deterministic + SLM fallback |
| **Synthesiser Agent** | Production-ready | `test_coverage_modes`, `test_coverage_query` | Evidence-only, citation-backed |
| **Verifier Agent** | Production-ready | `test_coverage_modes` | Per-claim verification |
| **Orchestrator Agent** | Production-ready | — | ReAct loop, sub-task delegation |
| **Extractor Agent** | Production-ready | `test_extractor_agent` (9 tests) | Schema-driven, regex + LLM |
| **Transformer Agent** | Production-ready | `test_transformer_agent` (37 tests) | MCP reference data normalization |
| **Classifier Agent** | Production-ready | `test_classifier_agent` | 4-tier LangGraph classification |
| **Preprocessor Agent** | Production-ready | `test_preprocessor_agent` (124 tests) | Quality-driven strategy selection |
| **Model Gateway** | Production-ready | `test_model_gateway` (16 tests) | Circuit breaker, audit logging |
| **Message Bus** | Production-ready | `test_message_bus` (8 tests) | Typed messages, audit trail |
| **Prompt Registry** | Production-ready | — | Content-hashed versioning |
| **Audit Logging** | Production-ready | — | Immutable, append-only, via repo.py |
| **Parser Abstraction** | Production-ready | `test_parser_abstraction` | PyMUPDF, Docling, multi-format |
| **Azure OpenAI** | Production-ready | `test_azure_openai` | Dual-provider support |
| **Multi-Format Ingestion** | Production-ready | — | DOCX, Excel, CSV, JSON, HTML |
| **Schema Contract** | Production-ready | `test_schema_contract` (integration) | Columns + unique + FK + indexes |
| **DB Migrations** | Production-ready | `test_migrations_and_contract` (integration) | 9 migrations (001–009) |

## Feedback / Retraining

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| **feedback_loop/ pipeline** | Production-backed (explicit wiring) | `test_feedback_loop` (45 tests) | Use `create_production()` or `create_test()` |
| **Attribution Engine** | Production-ready | `test_feedback_loop` | 6-rule deterministic |
| **Training Row Builders** | Production-ready | `test_feedback_loop` | 5 layer-specific builders |
| **Boundary Policy Guard** | Production-ready | `test_feedback_loop` | Strict: requires full granularity or policy approval |
| **InMemory Services** | In-memory / test utility | `test_feedback_loop` | Ingestion, trace join, orchestration |
| **Postgres Services** | Production-backed (explicit wiring) | — (integration) | DB-backed ingestion, trace join, orchestration |
| **Pipeline Factories** | Production-ready | — | `create_production(get_conn)`, `create_test()` |
| **Model Evaluator** | In-memory / test utility | `test_feedback_loop` | Placeholder metrics; needs real model runner |
| **Model Promotion** | In-memory / test utility | `test_feedback_loop` | State machine lifecycle; DB promotion planned |
| **agents/feedback_agent** | Deprecated | `test_feedback_and_retraining` (legacy) | Removal: Phase 8 |
| **agents/retraining_agent** | Deprecated | `test_feedback_and_retraining` (legacy) | Removal: Phase 8 |

## State Model

| Enum | Location | Status |
|------|----------|--------|
| `DocumentLifecycle` | `core/contracts.py` | Transitional — defined, not yet wired to ingestion |
| `QueryState` | `core/contracts.py` | Transitional — defined, not yet wired to orchestrator |
| `FeedbackState` | `core/contracts.py` | Transitional — mirrors feedback_loop pipeline stages |
| `BoundaryGranularity` | `feedback_loop/models.py` | Production-ready — used by boundary guard |

## Planned (Not Yet Implemented)

| Subsystem | Phase | Blocker |
|-----------|-------|---------|
| Conversational + Episodic Memory | Phase 6 | — |
| PII Detection / Compliance Agent | Phase 7 | — |
| 4-Level Explainability Reports | Phase 7 | — |
| RBAC Enforcement Middleware | Phase 8 | Users/document_access tables exist |
| JWT Authentication | Phase 8 | — |
| Cross-Corpus Entity Graph | Phase 8 | Neo4j tables exist |
| Vision / Chart Understanding | Phase 9 | — |
| Domain-Specific Embeddings | Phase 9 | — |
| Confidence Calibration Training | Phase 9 | CalibrationTrainingRow model exists |

## Deprecated Module Removal Schedule

| Module | Current State | Expected Removal |
|--------|--------------|-----------------|
| `agents/feedback_agent.py` | Deprecated, emits DeprecationWarning | Phase 8 |
| `agents/retraining_agent.py` | Deprecated, emits DeprecationWarning | Phase 8 |
| Migration 007 tables | Retained for backward compatibility | Phase 8 |
| `tests/test_feedback_and_retraining.py` | Legacy marker, auto-skipped by default CI | Phase 8 |

## Test Stratification

| Marker | Count (approx) | Description |
|--------|---------------|-------------|
| `unit` | ~275 | No external dependencies; auto-applied by conftest.py |
| `integration` | ~5 | Require PostgreSQL via TEST_DATABASE_URL |
| `external` | ~2 | Require external PDF files or services |
| `legacy` | ~32 | Deprecated module tests; auto-tagged by conftest.py |
| `slow` | 0 | Reserved for tests >5s |

Run commands:
```bash
pytest tests/ -v -m unit                   # unit tests only (CI default)
pytest tests/ -v -m integration            # DB tests only
pytest tests/ -v -m "not external"         # skip external-dependency tests
pytest tests/ -v -m "not legacy"           # skip deprecated module tests
pytest tests/ -v -m legacy                 # deprecated module tests only
pytest tests/ -v                           # all tests
```
