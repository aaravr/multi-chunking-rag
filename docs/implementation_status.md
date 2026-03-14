# Implementation Status Matrix

**Last updated:** 2026-03-14

This document provides a single-glance view of every major subsystem's implementation status, service tier, and test coverage.

---

## Subsystem Status

| Subsystem | Status | Service Tier | Tests | Notes |
|-----------|--------|-------------|-------|-------|
| **Ingestion Pipeline** | Complete | Production | `test_ingestion_smoke` (integration) | PyMUPDF default, Docling optional |
| **Late Chunking** | Complete | Production | `test_chunking_strategies` | Macro/child spans, 8192-token macro |
| **Embedding (ModernBERT)** | Complete | Production | `test_wo010_model_singleton` | Singleton, 768-dim, CPU |
| **Hybrid Retrieval** | Complete | Production | `test_bm25_index_manager`, `test_rrf_merge`, `test_rerank` | BM25 + vector + reranking |
| **Router Agent** | Complete | Production | `test_router_agent`, `test_intent_router`, `test_router_intent_subtypes` | Deterministic + SLM fallback |
| **Synthesiser Agent** | Complete | Production | `test_coverage_modes`, `test_coverage_query` | Evidence-only, citation-backed |
| **Verifier Agent** | Complete | Production | `test_coverage_modes` | Per-claim verification |
| **Orchestrator Agent** | Complete | Production | — | ReAct loop, sub-task delegation |
| **Extractor Agent** | Complete | Production | `test_extractor_agent` (9 tests) | Schema-driven, regex + LLM |
| **Transformer Agent** | Complete | Production | `test_transformer_agent` (37 tests) | MCP reference data normalization |
| **Classifier Agent** | Complete | Production | `test_classifier_agent` | 4-tier LangGraph classification |
| **Preprocessor Agent** | Complete | Production | `test_preprocessor_agent` (124 tests) | Quality-driven strategy selection |
| **Model Gateway** | Complete | Production | `test_model_gateway` (16 tests) | Circuit breaker, audit logging |
| **Message Bus** | Complete | Production | `test_message_bus` (8 tests) | Typed messages, audit trail |
| **Prompt Registry** | Complete | Production | — | Content-hashed versioning |
| **Audit Logging** | Complete | Production | — | Immutable, append-only, via repo.py |
| **Parser Abstraction** | Complete | Production | `test_parser_abstraction` | PyMUPDF, Docling, multi-format |
| **Azure OpenAI** | Complete | Production | `test_azure_openai` | Dual-provider support |
| **Multi-Format Ingestion** | Complete | Production | — | DOCX, Excel, CSV, JSON, HTML |
| **Schema Contract** | Complete | Production | `test_schema_contract` (integration) | Columns + unique constraints + indexes |
| **DB Migrations** | Complete | Production | `test_migrations_and_contract` (integration) | 9 migrations (001–009) |

## Feedback / Retraining

| Component | Status | Service Tier | Tests | Notes |
|-----------|--------|-------------|-------|-------|
| **feedback_loop/ pipeline** | Complete | Production | `test_feedback_loop` (45 tests) | Canonical path |
| **Attribution Engine** | Complete | Production | `test_feedback_loop` | 6-rule deterministic |
| **Training Row Builders** | Complete | Production | `test_feedback_loop` | 5 layer-specific builders |
| **Boundary Policy Guard** | Complete | Production | `test_feedback_loop` | Strict client validation |
| **InMemory Services** | Complete | Test/Dev | `test_feedback_loop` | Ingestion, trace join, orchestration |
| **Postgres Services** | Complete | Production | — (integration) | DB-backed ingestion, trace join, orchestration |
| **Pipeline Factories** | Complete | Production | — | `create_production()`, `create_test()` |
| **Model Evaluator** | Complete | In-memory | `test_feedback_loop` | Placeholder metrics |
| **Model Promotion** | Complete | In-memory | `test_feedback_loop` | State machine lifecycle |
| **agents/feedback_agent** | Deprecated | — | `test_feedback_and_retraining` | Use feedback_loop/ instead |
| **agents/retraining_agent** | Deprecated | — | `test_feedback_and_retraining` | Use feedback_loop/ instead |

## State Model

| Enum | Location | Status |
|------|----------|--------|
| `DocumentLifecycle` | `core/contracts.py` | Defined, not yet wired to ingestion |
| `QueryState` | `core/contracts.py` | Defined, not yet wired to orchestrator |
| `FeedbackState` | `core/contracts.py` | Defined, mirrors feedback_loop pipeline stages |

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

## Test Stratification

| Marker | Count (approx) | Description |
|--------|---------------|-------------|
| `unit` | ~275 | No external dependencies; auto-applied by conftest.py |
| `integration` | ~5 | Require PostgreSQL via TEST_DATABASE_URL |
| `external` | ~2 | Require external PDF files or services |
| `slow` | 0 | Reserved for tests >5s |

Run commands:
```bash
pytest tests/ -v -m unit               # unit tests only (default for CI)
pytest tests/ -v -m integration        # DB tests only
pytest tests/ -v -m "not external"     # skip external-dependency tests
pytest tests/ -v                       # all tests
```
