# Multi-Chunking RAG — Enterprise Intelligent Document Processing Platform

Enterprise-grade Agentic Intelligent Document Processing (IDP) RAG platform. Processes financial PDFs (annual reports, 10-K/10-Q, Basel Pillar 3, contracts, regulatory filings) with evidence-grounded, citation-backed answers and full provenance to exact page coordinates.

## Architecture

- **Agent Framework:** Orchestrator → Router → Retriever → Synthesiser → Verifier → Extractor → Transformer
- **Chunking:** Late chunking with macro/child spans (8192/256 tokens), BM25 hybrid retrieval, cross-encoder reranking
- **Lineage:** Every chunk carries doc_id, page_numbers, char_start/end, polygons, heading_path, section_id
- **Feedback:** Multi-layer attribution with boundary-safe training isolation (`feedback_loop/` subsystem)
- **LLM:** OpenAI or Azure OpenAI via Model Gateway with circuit breaker and full audit logging

See `CLAUDE.md` for the full architecture reference and `MASTER_PROMPT.md` for the enterprise constitution.

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 14 with pgvector extension
- API keys (see `.env.example`)

### Install

```bash
pip install -r requirements.txt

# Optional backends (Docling parser, multi-format, feedback API):
pip install -r requirements-optional.txt
```

### Database

```bash
# Option A: Local Postgres
python storage/setup_db.py

# Option B: Docker
docker-compose up -d
python storage/setup_db.py
```

### Run

```bash
streamlit run app/poc_app.py
```

### Test

```bash
export TEST_DATABASE_URL=postgresql://user:pass@localhost/test_db
pytest tests/ -v
```

## Configuration

Copy `.env.example` to `.env`. Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | — |
| `OPENAI_API_KEY` | LLM for synthesis | — |
| `LLM_PROVIDER` | `openai` or `azure_openai` | `openai` |
| `ENABLE_HYBRID_RETRIEVAL` | BM25 + vector fusion | `false` |
| `ENABLE_VERIFIER` | LLM claim verification | `false` |
| `ENABLE_RERANKER` | Cross-encoder reranking | `false` |
| `ENABLE_EXTRACTOR` | Schema-driven field extraction | `false` |
| `COVERAGE_MODE` | `deterministic`, `llm_fallback`, `llm_always` | `llm_fallback` |
| `PARSER_BACKEND` | `pymupdf` or `docling` | `pymupdf` |

See `CLAUDE.md` for the full environment variable reference.
