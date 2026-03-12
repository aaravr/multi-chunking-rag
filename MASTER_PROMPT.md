# MASTER_PROMPT.md — Enterprise Agentic IDP RAG Platform

This document is the single source of truth (the constitution) for the enterprise
platform evolution. See the full spec in the task description and docs/decisions.md
for architectural decision rationale.

## Governance

- MASTER_PROMPT.md governs the enterprise agent architecture
- SPEC.md governs PoC-inherited behaviour (deterministic lineage, late chunking, etc.)
- SPEC_ADDENDUM.md extends SPEC.md (SPEC.md takes precedence on conflicts)
- If code and any spec disagree, the spec wins
- Any deviation requires explicit spec update with rationale in docs/decisions.md

## Implementation Status

Phases 1-5 of §12 are implemented. See CLAUDE.md for current status.

## Key References

- §2: Absolute Invariants (lineage, late chunking, evidence-only synthesis, auditability, SoC)
- §4: Agent Topology (Orchestrator, Router, Retriever, Synthesiser, Verifier, Compliance, Explainability)
- §5: Agent Communication Protocol (typed messages, message bus, flow control)
- §6: Memory System (working, conversational, episodic, semantic, procedural)
- §7: Model Registry & Gateway (model tiers, circuit breaker, audit logging)
- §8: Database Contract (core + enterprise tables, RLS, connection management)
- §12: Implementation Phases (build order)
- §13: Change Management (knowledge hardening, PR requirements)
