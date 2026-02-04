# RAG PoC
Skeleton project for selective Azure DI + Late Chunking RAG.

## Local Postgres 14 + pgvector (preferred)

1. Install pgvector for Postgres 14:
   - macOS (Homebrew): `brew install pgvector`
2. Ensure Postgres 14 is running locally.
3. Set `DATABASE_URL` in your environment (see `.env.example`).
4. Run schema setup:
   - `python storage/setup_db.py`

## Optional Docker Postgres

If you want a container instead of local Postgres:
- `docker-compose up -d`

## CoverageQuery Mode

Configure `COVERAGE_MODE` to control CoverageQuery behavior:
- `deterministic`: only deterministic list extraction
- `llm_fallback`: deterministic first, then LLM if too few items (default)
- `llm_always`: always use LLM list extraction on expanded scope
