# CLAUDE.md — AI Assistant Guide for multi-chunking-rag

## Project Overview

Selective Azure DI + Late Chunking RAG Proof-of-Concept for Intelligent Document Processing (IDP) on financial PDFs (e.g., annual reports, 600+ pages). The system extracts evidence-grounded answers with clickable PDF highlights and full lineage from answer → chunk → page → coordinates.

## Spec Governance

**SPEC.md is the single source of truth.** If code behavior and SPEC.md disagree, SPEC.md wins. SPEC_ADDENDUM.md extends it (SPEC.md takes precedence on conflicts). All changes must be traceable to the spec.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit + streamlit-pdf-viewer |
| LLM | OpenAI API (GPT-4o-mini) |
| Embeddings | nomic-ai/modernbert-embed-base (768-dim, CPU) |
| Database | PostgreSQL 14 + pgvector (HNSW index) |
| DB Access | psycopg2 + ThreadedConnectionPool |
| PDF Parsing | PyMUPDF (fitz) |
| Document Intelligence | Azure Document Intelligence SDK |
| Lexical Search | rank-bm25 (BM25Okapi) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 (optional) |
| Testing | pytest |

## Repository Structure

```
app/              Streamlit UI (poc_app.py)
core/             Config (config.py), contracts (contracts.py), logging
embedding/        Late chunking (late_chunking.py), ModernBERT embedder, model registry singleton
grounding/        PDF highlight annotations from chunk polygons
ingestion/        PDF ingestion pipeline, page triage, canonicalization, Azure DI client, document facts
retrieval/        Vector search, BM25 hybrid, intent-based query routing, metadata queries, reranking
storage/          DB pool, schema, migrations, repository CRUD, schema contract validation
synthesis/        LLM answer synthesis, coverage extraction, verifier, prompts
scripts/          Demo/integration scripts
tests/            ~58 test files (schema, routing, coverage, regression, WO-010 safety)
docs/             Decision log, traceability matrix, knowledge hardening, work orders
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

## Critical Invariants (Do Not Break)

1. **Deterministic Lineage (SPEC §4.2):** Every chunk MUST have doc_id, page_numbers, char_start/end, polygons, heading_path, section_id, embedding_model/dim
2. **Late Chunking (SPEC §6):** Macro chunks ≤8192 tokens, child spans ~256 tokens, one ModernBERT forward pass per macro, then pool child embeddings. No early chunking
3. **Table Atomicity (ADDENDUM A3):** Tables are single atomic chunks, never split by sentence chunking
4. **Selective DI Policy (SPEC §5):** Page triage computes 4 metrics (text_length, text_density, image_coverage_ratio, layout_complexity_score), persists decision + reason_codes
5. **DB Contract (SPEC §7):** All required columns in chunks table per spec
6. **Engineering Quality (SPEC §13, WO-010):** No model reload per request (singleton), connection pooling, no pickle, idempotent writes, method complexity ≤10, method length ≤40 lines, nesting ≤3

## Key Patterns

- **Query Intent Routing:** Patterns classify queries as `location` (page-specific), `coverage` (exhaustive lists), or `semantic` (standard QA). Coverage subtypes: `list`, `attribute`, `numeric_list`, `pointer`
- **Singleton Embedder:** `embedding/model_registry.py` prevents 440MB model reload per query
- **Connection Pooling:** `storage/db_pool.py` with context manager
- **BM25 Caching:** JSON-based cache (never pickle) in `retrieval/bm25_index.py`
- **Feature Flags:** All optional features controlled via `core/config.py` Settings dataclass

## Naming Conventions

- **Chunk types:** `narrative`, `table`, `heading`, `boilerplate`
- **Source types:** `native` (PyMUPDF), `di` (Azure DI)
- **Triage decisions:** `native_only`, `di_required`
- **Fact status:** `found`, `not_found`, `ambiguous`
- **Query intents:** `location`, `coverage`, `semantic`

## Knowledge Hardening (Mandatory)

When discovering non-obvious behavior, edge cases, or design trade-offs, you MUST harden knowledge into:

1. **SPEC.md / SPEC_ADDENDUM.md** — if it changes required behavior
2. **docs/decisions.md** — Context → Decision → Consequences → Alternatives
3. **docs/traceability.md** — Requirement → Spec ref → Files → Tests → Status
4. **Regression tests** — if discovered through a bug or debugging
5. **Code comments** — at branch points where "simplification" would reintroduce bugs

See `hardening.md` and `KNOWLEDGE_HARDENING.md` for full rules.

## Development Workflow

1. Read SPEC.md and SPEC_ADDENDUM.md before making changes
2. Check docs/decisions.md for prior architectural context
3. Make changes traceable to spec sections
4. Add/update tests — especially regression tests for bug fixes
5. Update docs/traceability.md for new or changed requirements
6. Record architectural decisions in docs/decisions.md
7. All components must run on CPU (Mac local development)

## Database Schema (4 tables)

- **documents** — doc_id (UUID PK), filename, sha256 (unique), page_count
- **pages** — (doc_id, page_number) PK, triage_metrics (JSONB), triage_decision, reason_codes
- **chunks** — chunk_id (UUID PK), doc_id, page_numbers[], text_content, char_start/end, polygons (JSONB), source_type, embedding (vector(768)), chunk_type, heading_path, section_id, macro_id, child_id; UNIQUE(doc_id, macro_id, child_id)
- **document_facts** — (doc_id, fact_name) PK, value, status, confidence, source_chunk_id, evidence_excerpt

Migrations in `storage/migrations/` (applied by `setup_db.py`).
