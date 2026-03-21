# Enterprise IDP Platform — Gap Analysis & Architecture Blueprint

## Current State vs. Target Spec

This document maps the current `multi-chunking-rag` codebase against the enterprise IDP
platform requirements. For each section, **HAVE** = implemented, **PARTIAL** = scaffolded
but incomplete, **GAP** = not implemented.

---

## SECTION 1 — Executive Summary

### Current Platform

The codebase is an **evidence-grounded RAG platform** for financial PDFs with:
- Late-chunking with full lineage (doc\_id → page → char offsets → polygons)
- Agentic architecture (Orchestrator, Router, Retriever, Synthesiser, Verifier,
  Classifier, Preprocessor, Feedback, Retraining — 9 agents)
- Hybrid retrieval (pgvector HNSW + BM25 + optional cross-encoder reranking)
- Citation-backed synthesis with claim verification
- Immutable audit trail for every LLM call
- Enterprise messaging (in-process bus + Kafka A2A option)

### Gap to Enterprise IDP Spec

The platform is strong on **retrieval, synthesis, lineage, and auditability** but
has significant gaps in **structured extraction, transformation, Azure-native
integration, multi-format support, human-in-the-loop review, and deployment
infrastructure**. The six required agents map as follows:

| Required Agent       | Status      | Current Mapping                        |
|---------------------|-------------|----------------------------------------|
| Orchestrator        | **HAVE**    | `agents/orchestrator_agent.py` — ReAct loop |
| Classifier          | **HAVE**    | `agents/classifier_agent.py` — 4-tier classification |
| Data Extractor      | **HAVE**    | `agents/extractor_agent.py` — schema-driven field extraction (§10.1) |
| Transformer         | **HAVE**    | `agents/transformer_agent.py` — MCP reference-data normalization (§10.2) |
| Feedback Loop       | **HAVE**    | `feedback_loop/` subsystem — canonical multi-layer feedback pipeline (agents/feedback_agent.py is deprecated) |
| Retraining          | **HAVE**    | `feedback_loop/` subsystem — boundary-safe retraining orchestration (agents/retraining_agent.py is deprecated) |

### Business Rationale for Closing Gaps

Closing the gaps enables cross-divisional scale:
- **Data Extractor + Transformer** make the platform usable for structured workflows
  (insurance claims, trade confirmations, regulatory filings) beyond Q&A
- **Azure-native integration** satisfies enterprise procurement and compliance
- **Multi-format + Docling** removes the PDF-only limitation
- **Human review UI** enables regulated-industry deployment
- **Container + AKS deployment** enables production operations at scale

---

## SECTION 2 — Architecture Principles

### HAVE (Implemented)

| Principle | Evidence |
|-----------|----------|
| **Modular** | 9 independent agents, each with typed contracts (`agents/contracts.py`) |
| **Agentic** | ReAct orchestration, message bus, per-agent autonomy |
| **Evidence-grounded** | Every synthesis uses only retrieved chunks; citations map claims→chunk\_ids |
| **Governed** | Immutable audit log (§2.4), prompt versioning, model registry |
| **Retrainable** | FeedbackAgent + RetrainingAgent + SGDClassifier incremental learning |

### PARTIAL

| Principle | Status | Gap |
|-----------|--------|-----|
| **Schema-driven** | Contracts are schema-typed but no *extraction* schema | Need declarative field definitions with types, validation, dependencies |
| **Cloud-portable** | Runs on CPU/local; docker-compose for deps only | No Dockerfile, no Helm, no Azure deployment templates |

### GAP

| Principle | Gap |
|-----------|-----|
| **Azure-native** | No Azure OpenAI, no Azure AI Search, no Azure Blob, no Managed Identity |
| **Multi-format** | PDF only — no DOCX, XLSX, images, scanned docs via Docling |

---

## SECTION 3 — End-to-End Architecture

### Current Pipeline (HAVE)

```
PDF Upload → Page Triage → PyMUPDF/Azure DI Parse → Late Chunking → Embed
  → pgvector Store → Query → Route → Hybrid Retrieve → Rerank → Synthesise
  → Verify Claims → Return Citations + Highlights
```

### Required Pipeline (TARGET)

```
Multi-Format Upload → Classify → Parse (Azure DI / Docling) → Layout Analysis
  → Context-Aware Chunking → Embed → Vector + Lexical Index
  → Schema-Driven Extract → Transform/Normalize → Validate → Human Review
  → Feedback → Retrain → Output Serve → Monitor
```

### Gap Analysis by Stage

| Stage | Status | Gap |
|-------|--------|-----|
| Ingestion | **HAVE** (PDF) | Multi-format (DOCX, XLSX, images), Azure Blob trigger |
| Preprocessing | **HAVE** (page triage) | OCR for scanned docs, language detection |
| OCR/Parsing | **PARTIAL** (PyMUPDF + Azure DI) | Docling alternative, pluggable parser abstraction |
| Layout understanding | **HAVE** (polygons, heading\_path, section\_id) | — |
| Chunking | **HAVE** (late chunking, multi-strategy) | — |
| Retrieval | **HAVE** (hybrid vector+BM25+reranker) | Azure AI Search option |
| Extraction | **GAP** | Schema-driven Data Extractor Agent needed |
| Transformation | **GAP** | Transformer Agent for reference-data normalization |
| Validation | **PARTIAL** (verifier checks claims) | Schema-level field validation needed |
| Feedback | **HAVE** (FeedbackAgent) | Human review UI integration |
| Retraining | **HAVE** (RetrainingAgent) | Fine-tuning, prompt A/B, parser tuning |
| Output serving | **PARTIAL** (Streamlit UI) | API output endpoint, webhook delivery |
| Monitoring | **PARTIAL** (OTel instrumentation, audit log) | Drift detection, live dashboards |

---

## SECTION 4 — Agent Design

### A. Orchestrator Agent — **HAVE**

| Aspect | Implementation |
|--------|---------------|
| File | `agents/orchestrator_agent.py` (430 lines) |
| Responsibilities | ReAct loop: THINK→ACT→OBSERVE→ASSEMBLE |
| Inputs | `OrchestratorInput` (query, doc scope, permissions, token budget) |
| Outputs | `OrchestratorOutput` (answer, citations, confidence, execution trace) |
| Tools | Delegates to Router, Retriever, Synthesiser, Verifier via MessageBus |
| Decision logic | Token budget tracking, max 5 sequential LLM calls, 2 retries per sub-task |
| Failure modes | Budget exhaustion, circuit breaker on LLM, sub-agent timeout |
| Observability | OTel spans, execution trace, decision chain, working memory |

**Gap**: Does not yet orchestrate extraction/transformation workflows, only Q&A.

### B. Classifier Agent — **HAVE**

| Aspect | Implementation |
|--------|---------------|
| File | `agents/classifier_agent.py` (1351 lines) |
| Responsibilities | 4-tier classification: deterministic→pgvector→SGD→LLM |
| Inputs | Filename, page count, front-matter text |
| Outputs | `ClassificationResult` (document\_type, label, confidence, method) |
| Tools | Regex patterns, pgvector HNSW, sklearn SGDClassifier, LLM fallback |
| Learning | `ClassificationMemory` with PatternStore + EmbeddingStore + SGD |
| Observability | Pattern accuracy tracking, classification method logging |

**Gap**: Does not detect scanned vs digital, multilingual, or template format.

### C. Data Extractor Agent — **GAP** (Not Implemented)

**Required responsibilities:**
- Schema-driven field extraction from narrative, tables, and forms
- Multi-pass extraction where fields depend on one another
- LLM + retrieval context for each extraction
- Preserve evidence and source grounding per field
- Field-level confidence scores

**What exists that can be extended:**
- `SynthesiserAgent` does evidence-grounded LLM calls (can be adapted)
- `RetrieverAgent` does targeted retrieval (can feed extraction context)
- `PromptRegistry` has versioned templates (can add extraction templates)
- `contracts.py` has `Citation` with chunk lineage (can link to extracted fields)

### D. Transformer Agent — **GAP** (Not Implemented)

**Required responsibilities:**
- Map extracted raw values to enterprise reference data
- Normalize to target schema (codes, enums, aliases, canonical forms)
- Resolve ambiguous mappings
- Produce final structured output with transformation history

**What exists that can be extended:**
- `storage/knowledge_graph.py` has entity resolution (Neo4j graph)
- `contracts.py` has entity types and canonical names
- `ingestion/document_facts.py` has fact normalization patterns

### E. Feedback Loop Agent — **HAVE**

| Aspect | Implementation |
|--------|---------------|
| File | `agents/feedback_agent.py` |
| Responsibilities | Collect feedback, validate, store, route to classifier/preprocessor/retriever |
| Inputs | `FeedbackEntry` (rating, comment, correction, cited chunks) |
| Outputs | `FeedbackResult` (routed\_to, actions\_taken) |
| Store | TTLCache-backed `FeedbackStore` (bounded, thread-safe) |
| Observability | Stats tracking, eval recording |

**Gap**: Not yet connected to a human review UI. No recurring error pattern detection.

### F. Retraining Agent — **HAVE**

| Aspect | Implementation |
|--------|---------------|
| File | `agents/retraining_agent.py` |
| Responsibilities | Evaluate feedback → retrain classifier SGD → prune low-accuracy patterns |
| Inputs | `RetrainingRequest` (trigger, targets, thresholds) |
| Outputs | `RetrainingResult` (components retrained, metrics before/after) |
| Triggers | scheduled, threshold, manual |
| Observability | Watermark tracking, eval recording |

**Gap**: Does not yet support prompt improvement, parser tuning, retrieval tuning,
reference mapping updates, or supervised fine-tuning decision logic.

---

## SECTION 5 — Azure + Open Source Component Mapping

| Capability | Azure-Native Option | Open-Source Option | Current Implementation | Recommended Default |
|-----------|--------------------|--------------------|----------------------|-------------------|
| **LLM Inference** | Azure OpenAI (GPT-4o/4o-mini) | OpenAI API / local LLMs | OpenAI API only | Azure OpenAI (enterprise) |
| **Document Parsing** | Azure Document Intelligence | Docling / PyMUPDF | PyMUPDF + Azure DI | Pluggable: Azure DI default, Docling fallback |
| **Vector Search** | Azure AI Search | pgvector HNSW | pgvector | pgvector (cost), Azure AI Search (scale) |
| **Lexical Search** | Azure AI Search (keyword) | BM25Okapi (rank\_bm25) | rank\_bm25 | rank\_bm25 (simplicity) |
| **Blob Storage** | Azure Blob Storage | Local filesystem | Local filesystem | Azure Blob (production) |
| **Queue/Messaging** | Azure Service Bus | Kafka (kafka-python) | Kafka + in-process bus | Azure Service Bus (managed) or Kafka (portable) |
| **Container Runtime** | Azure Container Apps / AKS | Docker Compose / K8s | Docker Compose (dev only) | AKS with Helm charts |
| **Embeddings** | Azure OpenAI embeddings | ModernBERT (nomic-ai) | ModernBERT local CPU | ModernBERT (cost + privacy) |
| **Reranking** | — | ms-marco-MiniLM cross-encoder | cross-encoder local | Keep local (latency) |
| **Graph DB** | Azure Cosmos DB (Gremlin) | Neo4j Community | Neo4j | Neo4j (richer query language) |
| **Monitoring** | Azure Monitor / App Insights | OpenTelemetry + Grafana | OTel (partial) | OTel → Azure Monitor exporter |
| **Auth/RBAC** | Azure AD / Entra ID | Custom JWT | Schema only (migration 004) | Azure AD integration |
| **Secrets** | Azure Key Vault | .env file | .env file | Azure Key Vault (production) |

### Azure Document Intelligence vs Docling Comparison

| Criteria | Azure DI | Docling |
|----------|----------|---------|
| **Table preservation** | Excellent — cell-level extraction with spans | Good — requires post-processing for complex merges |
| **Structured JSON output** | Native AnalyzeResult with pages/tables/paragraphs | Markdown + JSON export |
| **Layout fidelity** | High — reading order, bounding boxes, roles | High — deep learning layout model |
| **Lineage retention** | Page + polygon per element | Page + bounding box per element |
| **Extensibility** | API-based, limited customization | Fully open-source, custom models possible |
| **Cost** | Pay-per-page ($0.01–$1.50/page depending on model) | Free (compute cost only) |
| **Portability** | Azure-locked | Runs anywhere |
| **Regulated workloads** | Azure compliance certifications (SOC2, HIPAA) | Self-hosted, full data control |
| **OCR quality** | Excellent (built-in) | Good (Tesseract/EasyOCR backend) |
| **Scanned PDFs** | Native support | Supported via OCR pipeline |

**Recommendation**: Build a `DocumentParser` abstraction layer. Default to Azure DI for
production (compliance + quality), fall back to Docling for cost-sensitive or
air-gapped deployments.

---

## SECTION 6 — LLM + RAG Design

### HAVE (Implemented)

| Capability | Implementation |
|-----------|---------------|
| **Hybrid search** | pgvector cosine + BM25Okapi with RRF fusion (`retrieval/hybrid.py`) |
| **Cross-encoder reranking** | ms-marco-MiniLM-L-6-v2 (`retrieval/rerank.py`) |
| **Chunking strategy** | Late chunking: macro ≤8192 tokens, child ~256 tokens via tokenizer offsets |
| **Table-aware chunking** | Multi-chunking: page-level content classification → per-section strategy |
| **Grounding + citations** | Every synthesis maps [C#] tags → chunk\_ids with page/polygon lineage |
| **Lineage preservation** | Chunk carries doc\_id, page\_numbers, char\_start/end, polygons, heading\_path, section\_id, macro\_id, child\_id, source\_type |

### GAP

| Capability | Gap |
|-----------|-----|
| **Schema-aware context assembly** | No field-aware retrieval; retrieves for Q&A, not structured extraction |
| **Table retrieval strategy** | Tables are chunked but not queried structurally (no row/cell retrieval) |
| **Azure AI Search** | No integration; all retrieval is pgvector + BM25 |
| **RAG for extraction vs Q&A** | RAG only used for Q&A; extraction would need field-focused retrieval |

### When RAG Is Used (Current)

- **Always for Q&A**: Query → hybrid retrieve → rerank → synthesise with evidence
- **Never for classification**: Deterministic rules → pgvector similarity → SGD → LLM
- **Never for chunking**: Strategy selection is deterministic/learned, not LLM-based

### When RAG Should Also Be Used (Target)

- **Schema extraction**: Retrieve field-relevant chunks → LLM extracts typed values
- **Reference data resolution**: Retrieve from reference knowledge base → LLM maps values
- **Multi-hop extraction**: Field A depends on Field B → sequential retrieval

---

## SECTION 7 — Data Model and Lineage Model

### HAVE (Current Schema — 13 tables across 7 migrations)

| Table | Purpose | Migration |
|-------|---------|-----------|
| `documents` | doc\_id, filename, sha256, page\_count, document\_type, classification\_label | Core |
| `pages` | (doc\_id, page\_number), triage\_metrics, triage\_decision | Core |
| `chunks` | chunk\_id, doc\_id, page\_numbers[], text, char\_start/end, polygons, embedding, heading\_path, section\_id, macro\_id, child\_id | Core |
| `document_facts` | (doc\_id, fact\_name), value, status, confidence, source\_chunk\_id | 002 |
| `users` | user\_id, email, role, clearance\_level | 004 |
| `document_access` | doc\_id, user\_id/role, permission\_level | 004 |
| `audit_log` | Immutable LLM call log | 004 |
| `query_history` | query\_id, intent, answer, confidence, verdict | 004 |
| `entities` | entity\_id, type, canonical\_name, aliases | 004 |
| `entity_mentions` | entity\_id, chunk\_id, mention\_text, confidence | 004 |
| `episodic_memory` | Per-user-per-document memory | 004 |
| `prompt_templates` | Versioned prompts | 004 |
| `classification_memory` | Pattern accuracy tracking | 005 |
| `classification_embeddings` | pgvector HNSW for classifier | 006 |
| `feedback_entries` | User feedback with rating, correction | 007 |
| `retraining_events` | Retraining audit trail | 007 |
| `chunking_outcomes` | Strategy quality scores | 007 |

### GAP (Missing Tables)

| Required Table | Purpose |
|---------------|---------|
| `extraction_schemas` | Declarative field definitions, types, validation rules, dependencies |
| `extracted_fields` | Per-field values with confidence, source\_chunk\_id, evidence text |
| `normalized_fields` | Transformed values with reference mapping, transformation history |
| `review_tasks` | Human review queue with status, assignee, deadline |
| `review_decisions` | Reviewer corrections with before/after values |
| `reference_data` | Enterprise master data lookups (codes, enums, canonical values) |
| `model_versions` | Parser/model/prompt version tracking for reproducibility |
| `drift_metrics` | Time-series quality metrics for drift detection |

---

## SECTION 8 — Processing Flow

### Current Flow (Q&A Only)

```
1. User uploads PDF
2. Ingestion: PyMUPDF extract → page triage → Azure DI (if needed) → canonicalize
3. Chunking: Preprocessor decides strategy → late chunking → embed → store in pgvector
4. Query: User asks question
5. Route: Router classifies intent (deterministic → patterns → SLM)
6. Retrieve: Hybrid vector+BM25 → optional rerank
7. Synthesise: LLM generates evidence-grounded answer with [C#] citations
8. Verify: Per-claim verification against source chunks
9. Display: Streamlit UI with PDF highlights on cited pages
```

### Target Flow (Full IDP)

```
1.  Document arrives (PDF, DOCX, XLSX, image) via API/Blob trigger
2.  Orchestrator creates workflow instance
3.  Classifier: type, template, format, quality assessment
4.  Parser: Azure DI or Docling (pluggable) → structured layout
5.  Chunking: Context-aware, layout-aware, table-preserving
6.  Embed + Index: pgvector + BM25 (or Azure AI Search)
7.  Schema Load: Fetch extraction schema for document type
8.  Extract: Data Extractor retrieves field-relevant chunks → LLM extracts
9.  Transform: Transformer normalizes to reference data → validates
10. Validate: Schema validation, confidence checks, dependency resolution
11. Review Queue: Low-confidence/ambiguous fields → human reviewer
12. Reviewer: See value + source + page → correct/approve/reject
13. Feedback: Corrections → FeedbackAgent → learning signals
14. Retrain: RetrainingAgent evaluates and acts on feedback
15. Output: Structured JSON to downstream systems via API/webhook
16. Monitor: Drift detection, quality dashboards, cost tracking
```

**Steps 7–12, 15–16 are not yet implemented.**

---

## SECTION 9 — Human Review and Feedback Loop

### HAVE

- `FeedbackAgent` collects structured feedback (positive/negative/correction)
- `FeedbackStore` persists entries with TTL, doc-level grouping, stats
- Routing to classifier (pattern accuracy), preprocessor (quality scores), retriever (relevance)
- `RetrainingAgent` consumes feedback for retraining decisions

### GAP

| Requirement | Status |
|------------|--------|
| Review queue for low-confidence fields | **GAP** — no review task table or assignment logic |
| Reviewer sees extracted value + source evidence + page location | **PARTIAL** — Streamlit shows citations but not in review context |
| Reviewer can correct values | **GAP** — UI is read-only |
| Structured feedback capture (field-level) | **GAP** — feedback is answer-level, not field-level |
| Review triggers (confidence threshold, missing mandatory) | **GAP** — no trigger rules |
| Feedback → measurable improvement tracking | **PARTIAL** — RetrainingAgent tracks metrics\_before/after |

---

## SECTION 10 — Retraining / Continuous Improvement Strategy

### HAVE

| Strategy | Implementation |
|----------|---------------|
| SGD incremental training | `classifier_agent.py` SGDClassifierWrapper.partial\_fit() |
| Pattern accuracy pruning | `RetrainingAgent._prune_low_accuracy_patterns()` |
| Quality score adjustment | `RetrainingAgent._retrain_preprocessor()` |
| Watermark tracking | Prevents redundant retraining cycles |
| Three trigger modes | scheduled, threshold, manual |

### GAP

| Strategy | Status |
|----------|--------|
| **Prompt improvement** | Not implemented — need A/B testing of prompt versions |
| **Parser tuning** | Not implemented — need per-document-type parser config |
| **Retrieval tuning** | Not implemented — need query-level relevance feedback → index tuning |
| **Rules adjustment** | Not implemented — need automated rule generation from patterns |
| **Reference mapping updates** | Not implemented — need Transformer Agent first |
| **Supervised fine-tuning** | Not implemented — need training data pipeline |
| **Evaluation before promotion** | Not implemented — need holdout eval set |
| **Safe rollout/rollback** | Not implemented — need model versioning + canary deployment |

---

## SECTION 11 — Non-Functional Requirements

| Requirement | Status | Evidence |
|------------|--------|----------|
| **Scalability** | PARTIAL | Kafka A2A for distributed agents; but no container orchestration |
| **Latency** | HAVE | Token budget tracking, circuit breakers, connection pooling |
| **Resilience** | HAVE | Circuit breakers (model\_gateway, kafka), retry with tenacity, DLQ |
| **Cost control** | HAVE | Per-call cost tracking, token budget per query |
| **Security** | PARTIAL | RBAC schema defined (migration 004); not enforced in app layer |
| **Compliance** | PARTIAL | Immutable audit log; no PII detection or access enforcement |
| **Explainability** | PARTIAL | Execution trace, decision chain; no 4-level report yet |
| **Portability** | PARTIAL | CPU-only, no GPU required; but no containerization |
| **Maintainability** | HAVE | Typed contracts, message bus, SOLID principles, 547+ tests |

---

## SECTION 12 — Risks and Tradeoffs

| Tradeoff | Current Position | Risk |
|----------|-----------------|------|
| **Azure-native vs open-source** | Fully open-source (no Azure LLM/search) | Cannot deploy in Azure-mandated enterprises |
| **Parser fidelity vs cost** | Azure DI for complex, PyMUPDF for simple | No Docling option for air-gapped deployments |
| **Latency vs accuracy** | Deterministic-first classification, LLM fallback | Good balance for Q&A; unknown for extraction |
| **Agent autonomy vs control** | Orchestrator controls all sub-tasks | May need human-in-the-loop checkpoints for extraction |
| **Fine-tuning vs prompt engineering** | Prompt-only (no fine-tuning) | May hit ceiling on extraction accuracy |
| **Table preservation vs throughput** | Late chunking preserves tables | Multi-strategy adds complexity per document |
| **Q&A vs extraction** | Platform optimized for Q&A retrieval | Extraction needs different retrieval + output patterns |

---

## SECTION 13 — Recommended Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE GOVERNANCE LAYER                       │
│  Azure AD / RBAC │ Audit Log (immutable) │ Key Vault │ Compliance   │
└───────────┬──────────────────────────────────────────┬───────────────┘
            │                                          │
┌───────────▼──────────────────────────────────────────▼───────────────┐
│                      ORCHESTRATION LAYER                             │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              ORCHESTRATOR AGENT (ReAct Loop)                 │    │
│  │  Plan → Route → Retrieve → Extract → Transform → Verify     │    │
│  └──────┬──────┬──────┬──────┬──────┬──────┬──────┬────────────┘    │
│         │      │      │      │      │      │      │                  │
│  ┌──────▼─┐ ┌─▼────┐ │  ┌───▼───┐ ┌▼─────┐│  ┌──▼──────┐          │
│  │Classif.│ │Router│ │  │Extract│ │Trans- ││  │Verifier │          │
│  │ Agent  │ │Agent │ │  │ Agent │ │former ││  │ Agent   │          │
│  │(4-tier)│ │(det.)│ │  │(NEW)  │ │(NEW)  ││  │(claims) │          │
│  └────────┘ └──────┘ │  └───────┘ └───────┘│  └─────────┘          │
│              ┌───────▼──┐          ┌────────▼──────────┐            │
│              │Retriever │          │Feedback│Retraining│            │
│              │  Agent   │          │ Agent  │  Agent   │            │
│              │(hybrid)  │          │(§4.10) │ (§4.11)  │            │
│              └──────────┘          └────────┴──────────┘            │
└──────────┬───────────────────────────────────────┬───────────────────┘
           │                                       │
┌──────────▼───────────────────────────────────────▼───────────────────┐
│                        PARSING LAYER                                 │
│                                                                      │
│  ┌──────────────────┐    ┌─────────────────┐    ┌────────────────┐  │
│  │  Azure Document   │    │    Docling       │    │   PyMUPDF      │  │
│  │  Intelligence     │◄──►│  (open-source)   │◄──►│  (native PDF)  │  │
│  │  (tables, OCR)    │    │  (tables, OCR)   │    │  (fast, local) │  │
│  └──────────────────┘    └─────────────────┘    └────────────────┘  │
│              ▲                                                        │
│              │  Pluggable DocumentParser Abstraction                  │
└──────────────┼───────────────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────────────┐
│                      RETRIEVAL LAYER                                 │
│                                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │  pgvector    │  │  BM25Okapi   │  │ Cross-Encoder│               │
│  │  HNSW Index  │  │  Lexical     │  │  Reranker    │               │
│  │  (768-dim)   │  │  (JSON cache)│  │  (MiniLM)    │               │
│  └─────────────┘  └──────────────┘  └──────────────┘               │
│         ▲                                                            │
│         │  Optional: Azure AI Search (managed, scaled)               │
└─────────┼────────────────────────────────────────────────────────────┘
          │
┌─────────▼────────────────────────────────────────────────────────────┐
│              EXTRACTION & TRANSFORMATION LAYER (NEW)                 │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ Extraction Schema │  │ Reference Data   │  │ Validation Rules │  │
│  │ (field defs,      │  │ (codes, enums,   │  │ (required fields,│  │
│  │  types, deps)     │  │  canonical maps) │  │  thresholds)     │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
          │
┌─────────▼────────────────────────────────────────────────────────────┐
│                    HUMAN REVIEW LAYER (NEW)                          │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ Review Queue      │  │ Review UI        │  │ Structured       │  │
│  │ (low-confidence,  │  │ (value + source  │  │ Feedback Capture │  │
│  │  ambiguous, failed│  │  + page + correct│  │ (field-level     │  │
│  │  ref mapping)     │  │  + approve/reject│  │  corrections)    │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────┐
│                  FEEDBACK / RETRAINING LOOP                          │
│                                                                      │
│  FeedbackAgent ──► FeedbackStore ──► RetrainingAgent                │
│       │                                    │                         │
│       ├─► Classifier (pattern accuracy)    ├─► SGD retrain           │
│       ├─► Preprocessor (quality scores)    ├─► Pattern pruning       │
│       └─► Retriever (relevance signals)    ├─► Prompt A/B (NEW)     │
│                                            ├─► Parser tuning (NEW)   │
│                                            └─► Fine-tune eval (NEW)  │
└──────────────────────────────────────────────────────────────────────┘
          │
┌─────────▼────────────────────────────────────────────────────────────┐
│               DOWNSTREAM PLATFORM OUTPUTS                            │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ REST API  │  │ Webhook  │  │ Event    │  │ Structured JSON  │   │
│  │ (extract) │  │ Delivery │  │ Stream   │  │ (schema-valid)   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
          │
┌─────────▼────────────────────────────────────────────────────────────┐
│                     OBSERVABILITY                                    │
│                                                                      │
│  OpenTelemetry ──► Azure Monitor / Grafana                          │
│  Audit Log (immutable) │ Cost Tracking │ Drift Detection (NEW)      │
│  Field Confidence │ Model Versions │ Prompt Versions                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Priority Implementation Roadmap

Based on the gap analysis, the recommended implementation order:

| Priority | Component | Effort | Business Value |
|----------|-----------|--------|----------------|
| **P0** | Data Extractor Agent | High | Unlocks structured extraction use cases |
| **P0** | Extraction Schema model + migration | Medium | Foundation for schema-driven extraction |
| **P1** | Transformer Agent | Medium | Enables reference-data normalization |
| **P1** | Azure OpenAI support in ModelGateway | Low | Enterprise deployment requirement |
| **P1** | Docling parser integration + abstraction layer | Medium | Vendor portability |
| **P2** | Human Review UI + review task queue | High | Regulated-industry requirement |
| **P2** | Dockerfile + Helm charts | Medium | Production deployment |
| **P3** | Azure AI Search integration | Medium | Scale alternative to pgvector |
| **P3** | Drift monitoring + dashboards | Medium | Operational excellence |
| **P3** | Prompt A/B testing in RetrainingAgent | Medium | Continuous improvement |
| **P4** | Multi-format ingestion (DOCX, XLSX) | Medium | Broader document coverage |
| **P4** | Fine-tuning pipeline | High | Accuracy ceiling breaker |
