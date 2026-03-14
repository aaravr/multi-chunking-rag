# CLAUDE.md — AI Assistant Guide for multi-chunking-rag

## Project Overview

Enterprise-grade Agentic Intelligent Document Processing (IDP) RAG platform, evolved from a validated PoC. Processes financial PDFs (annual reports, 10-K/10-Q, Basel Pillar 3, contracts, regulatory filings) with evidence-grounded, citation-backed answers and full provenance to exact page coordinates.

## Spec Governance

**MASTER_PROMPT.md is the enterprise constitution.** SPEC.md and SPEC_ADDENDUM.md govern PoC-inherited behaviour. If code and spec disagree, the spec wins. All changes must be traceable to spec sections.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit + streamlit-pdf-viewer |
| LLM (Tier-1) | GPT-4o / Claude Opus — complex synthesis, orchestration |
| LLM (Tier-2) | GPT-4o-mini / Claude Sonnet — standard synthesis, explainability |
| Embeddings | nomic-ai/modernbert-embed-base (768-dim, CPU) |
| Database | PostgreSQL 14 + pgvector (HNSW index) |
| DB Access | psycopg2 + ThreadedConnectionPool + execute_values |
| PDF Parsing | PyMUPDF (fitz) + Docling (optional) |
| Document Parsing | Parser abstraction: PyMUPDF, Docling, DOCX, Excel, CSV, JSON, HTML |
| Document Intelligence | Azure Document Intelligence SDK |
| LLM Provider | OpenAI (default) or Azure OpenAI (via LLM_PROVIDER config) |
| Lexical Search | rank-bm25 (BM25Okapi) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 (optional) |
| Extraction | Schema-driven field extraction with MCP reference data normalization |
| Testing | pytest |

## Repository Structure

```
agents/           Enterprise agent framework (MASTER_PROMPT §4-§5)
  contracts.py      Typed agent contracts (25+ frozen dataclasses)
  message_bus.py    In-process message bus with audit trail (§5)
  model_gateway.py  Central LLM gateway with circuit breaker (§7.3)
  prompt_registry.py Versioned prompt templates (§4.4)
  audit.py          Audit log DB writer (§2.4)
  base.py           Abstract base agent class
  orchestrator_agent.py  ReAct reasoning loop (§4.1)
  router_agent.py   Intent classification + plan building (§4.2)
  retriever_agent.py Evidence finder (§4.3)
  synthesiser_agent.py Answer generation (§4.4)
  verifier_agent.py  Claim verification (§4.5)
  extractor_agent.py Schema-driven field extraction (§10.1)
  transformer_agent.py Value normalization via MCP (§10.2)
  mcp_reference_server.py MCP reference data server (§10.3)
app/              Streamlit UI (poc_app.py)
core/             Config (config.py), contracts (contracts.py), logging
embedding/        Late chunking (late_chunking.py), ModernBERT embedder, model registry singleton
grounding/        PDF highlight annotations from chunk polygons
ingestion/        Multi-format ingestion pipeline, page triage, canonicalization, Azure DI client, document facts
  parser_base.py    Parser abstraction layer (§10.4)
  pymupdf_parser.py PyMUPDF parser backend (default)
  docling_parser.py Docling parser backend (optional)
  multi_format_parser.py DOCX, Excel, CSV, JSON, HTML parsers
feedback_loop/    CANONICAL feedback/retraining subsystem (see §Feedback Architecture below)
  models.py         Pydantic v2 typed domain models (BoundaryKey, PredictionTrace, TrainingRows)
  pipeline.py       End-to-end orchestration: Ingest → Join → Normalize → Attribute → Build → Guard → Submit
  attribution.py    6-rule attribution engine mapping feedback → impacted decision layers
  training_rows.py  Layer-specific training row builders (planner, classifier, chunking, extraction, calibration)
  boundary.py       Boundary-safe training isolation: B = (client, division, jurisdiction)
  interfaces.py     Abstract service interfaces (pluggable backends)
  services.py       In-memory service implementations (to be replaced with DB-backed)
  api.py            FastAPI route skeletons for feedback API
retrieval/        Vector search, BM25 hybrid, intent-based query routing, metadata queries, reranking
storage/          DB pool, schema, migrations, repository CRUD, schema contract validation
synthesis/        LLM answer synthesis, coverage extraction, verifier, prompts
scripts/          Demo/integration scripts
tests/            Test suite (contracts, bus, gateway, router, agents, schema, routing, coverage, regression)
docs/             Decision log, traceability matrix, knowledge hardening, work orders, engineering review
```

## Key Commands

```bash
# Setup database (Postgres 14 + pgvector)
docker-compose up -d
python storage/setup_db.py

# Run tests
export TEST_DATABASE_URL=postgresql://user:pass@localhost/test_db
pytest tests/ -v

# Run the app
streamlit run app/poc_app.py
```

## Environment Variables

Copy `.env.example` to `.env`. Key variables:

- `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` / `_KEY` — Azure DI credentials
- `OPENAI_API_KEY` — LLM for synthesis
- `DATABASE_URL` — PostgreSQL connection string
- `DISABLE_DI` — Skip Azure DI (default: false)
- `ENABLE_HYBRID_RETRIEVAL` — BM25 + vector fusion (default: false)
- `ENABLE_VERIFIER` — LLM claim verification (default: false)
- `ENABLE_RERANKER` — Cross-encoder reranking (default: false)
- `ENABLE_DOCUMENT_FACTS` — Cache metadata facts (default: false)
- `COVERAGE_MODE` — `deterministic` | `llm_fallback` | `llm_always` (default: llm_fallback)
- `ENABLE_EXTRACTOR` — Schema-driven field extraction (default: false)
- `ENABLE_TRANSFORMER` — Value normalization via MCP (default: false)
- `MCP_REFERENCE_SERVER_URL` — MCP reference data server (default: http://localhost:8100)
- `LLM_PROVIDER` — `openai` | `azure_openai` (default: openai)
- `AZURE_OPENAI_ENDPOINT` — Azure OpenAI resource endpoint
- `AZURE_OPENAI_API_KEY` — Azure OpenAI API key
- `AZURE_OPENAI_API_VERSION` — Azure API version (default: 2024-06-01)
- `AZURE_OPENAI_DEPLOYMENT_ID` — Azure deployment name (overrides model_id)
- `ENABLE_MULTI_FORMAT` — DOCX, Excel, CSV, JSON, HTML ingestion (default: false)
- `PARSER_BACKEND` — `pymupdf` | `docling` (default: pymupdf)

## Absolute Invariants (MASTER_PROMPT §2 — MUST NEVER BREAK)

1. **Deterministic Lineage (§2.1):** Every chunk MUST have doc_id, page_numbers, char_start/end, polygons, heading_path, section_id, embedding_model/dim, document_type, classification_label. Lineage MUST be carried through every agent hop.
2. **Late Chunking (§2.2):** Macro chunks ≤8192 tokens, ONE forward pass per macro, child spans ~256 tokens via tokeniser offsets. NEVER use early/naive chunking.
3. **Evidence-Only Synthesis (§2.3):** Every LLM synthesis call MUST use ONLY retrieved evidence. Explicit citations mapping claims to chunk_ids. NEVER hallucinate.
4. **Auditability (§2.4):** Every LLM call logged with call_id, model_id, full_prompt, full_response, tokens, temperature (MUST be 0 for synthesis), cost. Audit logs are IMMUTABLE and APPEND-ONLY.
5. **Separation of Concerns (§2.5):** UI → API Gateway → Orchestrator → Agents → Tools → Storage → Audit. No cross-layer imports except through defined interface contracts.

## Agent Architecture (MASTER_PROMPT §4)

| Agent | Role | Model Tier |
|-------|------|-----------|
| **Orchestrator** | ReAct loop: plan → retrieve → synthesise → verify | Tier-1 (GPT-4o/Opus) |
| **Router** | Intent classification + query decomposition | Deterministic + SLM fallback |
| **Retriever** | Locate → Expand → Select → Rerank pipeline | None (deterministic) |
| **Synthesiser** | Evidence-grounded answer generation | Tier-2 (GPT-4o-mini/Sonnet) |
| **Verifier** | Per-claim citation verification | Deterministic + SLM |
| **Extractor** | Schema-driven field extraction (§10.1) | Tier-2 (GPT-4o-mini/Sonnet) |
| **Transformer** | Value normalization via MCP reference data (§10.2) | None (deterministic + MCP) |
| **Compliance** | PII detection, access control, policy rules | Deterministic + SLM |
| **Explainability** | 4-level audit report generation | Tier-2 |

### Agent Communication (§5)
- Agents communicate via **typed messages** on the in-process message bus
- Every message is logged to the audit trail
- Orchestrator is the ONLY agent that can initiate sub-tasks
- All LLM calls go through the **Model Gateway** (§7.3)

### Query Intent Types
- **PoC intents:** `location`, `coverage` (list/attribute/numeric_list/pointer), `semantic`
- **Enterprise intents:** `comparison`, `multi_hop`, `aggregation`, `metadata`

### Feedback / Retraining Architecture

**Canonical path: `feedback_loop/` subsystem** (migration 009).

The canonical feedback/retraining pipeline is:
1. **Ingest** — validate + persist feedback event with BoundaryKey
2. **Join** — link with runtime prediction trace (quarantine if missing)
3. **Normalize** — derive structured ReasonCodes
4. **Attribute** — 6-rule engine → impacted DecisionLayers
5. **Build** — layer-specific TrainingRows (planner, classifier, chunking, extraction, calibration)
6. **Guard** — boundary validation + cross-boundary sanitization
7. **Submit** — queue for retraining via RetrainingOrchestrator

**Deprecated:** `agents/feedback_agent.py` and `agents/retraining_agent.py` are deprecated.
They remain importable for backward compatibility but emit `DeprecationWarning` on import.
All new feedback/retraining work must use the `feedback_loop/` subsystem.

## Key Patterns

- **Model Gateway:** All LLM calls route through `agents/model_gateway.py` — validates model registration, enforces circuit breaker (3 failures in 60s), logs complete prompt+response, tracks cost
- **Prompt Registry:** All prompts from `agents/prompt_registry.py` — versioned by content hash, deterministic template selection by (intent, coverage_subtype, status_filter)
- **Message Bus:** `agents/message_bus.py` — typed AgentMessage envelope, full audit trail, handler registration
- **Singleton Embedder:** `embedding/model_registry.py` — prevents 440MB model reload per query
- **Connection Pooling:** `storage/db_pool.py` — ThreadedConnectionPool with context manager
- **Bulk Inserts:** `storage/repo.py` — uses `execute_values()` for batch performance
- **BM25 Caching:** JSON-based cache (never pickle) in `retrieval/bm25_index.py`
- **Feature Flags:** All optional features controlled via `core/config.py` Settings dataclass
- **Schema-Driven Extraction:** Declarative field schemas in `agents/extractor_agent.py` — regex-first with LLM fallback, per-field confidence
- **MCP Reference Data:** Team-extensible reference data via `agents/mcp_reference_server.py` — standard contract (MCPLookupRequest/Response), reference implementation ships with platform
- **Parser Abstraction:** Pluggable parsers via `ingestion/parser_base.py` — PyMUPDF default, Docling optional, multi-format (DOCX, Excel, CSV, JSON, HTML)
- **Azure OpenAI:** Dual-provider support via `LLM_PROVIDER` config — vanilla OpenAI or Azure OpenAI with deployment ID override

## Naming Conventions

- **Chunk types:** `narrative`, `table`, `heading`, `boilerplate`
- **Source types:** `native` (PyMUPDF), `di` (Azure DI)
- **Triage decisions:** `native_only`, `di_required`
- **Fact status:** `found`, `not_found`, `ambiguous`
- **Query intents:** `location`, `coverage`, `semantic`, `comparison`, `multi_hop`, `aggregation`, `metadata`
- **Agent names:** `orchestrator`, `router`, `retriever`, `synthesiser`, `verifier`, `compliance`, `explainability`
- **Model tiers:** `tier-1-api`, `tier-2-api`, `local-slm`, `local-embedding`, `local-reranker`

## Knowledge Hardening (Mandatory)

When discovering non-obvious behavior, edge cases, or design trade-offs, you MUST harden knowledge into:

1. **MASTER_PROMPT.md / SPEC.md / SPEC_ADDENDUM.md** — if it changes required behavior
2. **docs/decisions.md** — Context → Decision → Consequences → Alternatives
3. **docs/traceability.md** — Requirement → Spec ref → Files → Tests → Status
4. **Regression tests** — if discovered through a bug or debugging
5. **Code comments** — at branch points where "simplification" would reintroduce bugs

See `hardening.md` and `KNOWLEDGE_HARDENING.md` for full rules.

## Development Workflow

1. Read MASTER_PROMPT.md, SPEC.md, and SPEC_ADDENDUM.md before making changes
2. Check docs/decisions.md for prior architectural context
3. Make changes traceable to spec sections (cite §numbers)
4. All LLM calls MUST go through Model Gateway — never call OpenAI directly from agents
5. All agent communication MUST use typed contracts from `agents/contracts.py`
6. Add/update tests — especially regression tests for bug fixes
7. Update docs/traceability.md for new or changed requirements
8. Record architectural decisions in docs/decisions.md
9. All components must run on CPU (Mac local development)

## Database Schema

### Core Tables (PoC — 4 tables)
- **documents** — doc_id (UUID PK), filename, sha256 (unique), page_count, document_type, classification_label, entity_name, reporting_period, jurisdiction
- **pages** — (doc_id, page_number) PK, triage_metrics (JSONB), triage_decision, reason_codes
- **chunks** — chunk_id (UUID PK), doc_id, page_numbers[], text_content, char_start/end, polygons (JSONB), source_type, embedding (vector(768)), chunk_type, heading_path, section_id, macro_id, child_id, document_type, classification_label; UNIQUE(doc_id, macro_id, child_id)
- **document_facts** — (doc_id, fact_name) PK, value, status, confidence, source_chunk_id, evidence_excerpt

### Enterprise Tables (Migration 004 — 8 new tables)
- **users** — user_id, email, role, clearance_level
- **document_access** — doc_id, user_id/role, permission_level (RBAC + document-level)
- **audit_log** — IMMUTABLE, APPEND-ONLY; logs every LLM call with full prompt/response
- **query_history** — query_id, user_id, intent_type, answer_summary, confidence, verification_verdict
- **entities** — entity_id, entity_type, canonical_name, aliases (knowledge graph)
- **entity_mentions** — mention_id, entity_id, chunk_id, doc_id, mention_text, confidence
- **episodic_memory** — per-user-per-document memory (query history, fact cache, annotations)
- **prompt_templates** — template_id, intent_type, version, template_text

### Extraction & Transform Tables (Migration 008 — 5 new tables)
- **extraction_schemas** — schema_id PK, document_type, classification_label, version, fields_json (JSONB)
- **extracted_fields** — extraction_id PK, doc_id, schema_id, field_name, raw_value, normalized_value, confidence, source_chunk_ids[], extraction_method
- **extraction_results** — result_id PK, doc_id, schema_id, overall_confidence, extraction_model
- **mcp_lookup_log** — lookup_id PK, doc_id, field_name, lookup_key/value, canonical_value, confidence, matched
- **transformation_rules** — rule_id PK, schema_id, field_name, transform_type, mcp_server_url

Migrations in `storage/migrations/` (001-008, applied by `setup_db.py`).

## Implementation Phases (MASTER_PROMPT §12)

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Stability Hardening | Complete | execute_values, singleton, pooling, JSON cache, idempotency |
| 2. Typed Contracts & Message Bus | Complete | Agent contracts, message bus, Model Gateway, Prompt Registry, audit log |
| 3. Core Agents | Complete | Retriever, Synthesiser, Verifier agents wrapped in contracts |
| 4. Router Agent (Extended) | Complete | ComparisonQuery, MultiHopQuery, AggregationQuery, MetadataQuery |
| 5. Orchestrator Agent | Complete | ReAct loop, sub-task delegation, token budget tracking |
| 5a. Extraction Pipeline | Complete | Extractor Agent, Transformer Agent, MCP reference data, parser abstraction, Azure OpenAI, multi-format |
| 6. Memory System | Planned | Conversational + episodic memory, coreference resolution |
| 7. Compliance & Explainability | Planned | PII detection, access control, 4-level explainability reports |
| 8. Multi-Document & Security | Planned | Cross-corpus search, RBAC, JWT auth, entity graph |
| 9. Vision & Advanced Models | Planned | Chart understanding, domain embeddings, confidence calibration |
