# WO-010 — Runtime Stability, Security & Code Quality Hardening

Status: Approved  
Owner: Sovereign Coding Agent  
Priority: Critical  
SPEC Impact: §4, §6, §7, §11  
Non-Goal Check: No production auth/HA/distributed systems introduced. Fully PoC-compliant.

---

## 1. Context

Enterprise review identified five runtime-level show-stoppers:

1. ModernBERT reloaded per query (440MB disk hit per request)
2. No database connection pooling
3. Pickle-based BM25 cache (RCE vulnerability)
4. Broken regex patterns in document_facts extraction
5. Non-idempotent chunk insertion (duplicate risk on retry)

Additionally, code review indicates:
- Excessive conditional branching
- Deeply nested logic
- High cyclomatic complexity
- Unclear naming patterns
- Lack of defensive programming

This Work Order addresses runtime stability, security, and enforceable code quality standards.

---

## 2. Objectives

The system MUST:

- Load embedding models exactly once per process
- Use deterministic DB connection pooling
- Eliminate unsafe pickle deserialization
- Correct all regex extraction patterns and add regression tests
- Guarantee idempotent chunk insertion
- Reduce method complexity and conditional branching
- Enforce meaningful naming and modular design

---

## 3. Scope of Changes

### 3.1 Model Singleton Enforcement

ModernBERT MUST be loaded once at process start.

- Introduce model_registry.py
- All embedding calls MUST use get_embedding_model()
- No model loading allowed inside request handlers

---

### 3.2 Database Connection Pooling

All DB access MUST use a shared connection pool.

- Introduce db_pool.py
- Replace direct psycopg2.connect() calls
- Pool size configurable via config

---

### 3.3 Secure BM25 Cache

Pickle-based caching is forbidden.

- Replace with JSON or SQLite-backed cache
- If cache file corrupted → safe fallback
- No dynamic code execution permitted

---

### 3.4 Regex Extraction Correction

All document_facts regex patterns MUST:

- Use raw string literals (r"\s+")
- Be unit-tested with positive and negative cases
- Fail explicitly if extraction confidence below threshold

---

### 3.5 Idempotent Chunk Insertion

Chunk table MUST enforce uniqueness.

Options:
- Unique constraint (doc_id, macro_id, child_id)
- Or hash-based dedupe
- Inserts MUST use ON CONFLICT DO NOTHING

Ingestion retries MUST NOT create duplicates.

---

## 4. Code Quality Standards (Mandatory)

The following standards are now enforced:

1. Maximum method length: 40 lines
2. Maximum cyclomatic complexity: 10
3. No nested conditionals beyond depth 3
4. Meaningful variable names required
5. No generic names (data, obj, temp, result)
6. No logic embedded inside large if/else trees
7. Prefer early returns over deep nesting
8. All non-trivial logic must have unit tests

These standards apply to all modified files.

---

## 5. Tests Required

### 5.1 Model Loading Test
Assert model loads once per process.

### 5.2 Pooling Test
Simulate concurrent queries; ensure no repeated connect() calls.

### 5.3 Regex Extraction Tests
- Positive case extraction
- Negative case rejection
- Edge whitespace tests

### 5.4 Idempotency Test
- Simulate ingestion crash
- Retry ingestion
- Assert chunk count stable

---

## 6. Acceptance Criteria

WO-010 is complete only if:

- No model reload per query
- No direct psycopg2.connect() outside pool module
- No pickle usage in codebase
- Regex tests pass
- Duplicate chunk insertion impossible
- Code complexity thresholds respected
- docs/traceability.md updated

---

## 7. Documentation Updates

- Update docs/decisions.md
- Update docs/traceability.md
- Add section to SPEC.md: “Engineering Quality & Runtime Safety”

---

Failure to comply invalidates PR.