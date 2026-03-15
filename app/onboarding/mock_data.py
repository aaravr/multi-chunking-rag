"""Centralized deterministic mock data for the Enterprise IDP Onboarding Studio.

All values are seeded and stable for screenshot generation.
No random values — every render produces identical output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional


# ── Workspace ────────────────────────────────────────────────────────

WORKSPACE = {
    "name": "Commercial Lending — Annual Review Extraction",
    "department": "Commercial Banking Operations",
    "business_unit": "Global Corporate & Investment Banking",
    "jurisdiction": "United States — OCC Regulated",
    "region": "North America",
    "sensitivity": "Confidential — PII Present",
    "regulatory_profile": "SOX / OCC / FDIC",
    "outcome_type": "Field Extraction + Validation",
    "owner": "Sarah Chen — VP, Lending Operations",
    "reviewers": ["James Okoro — Sr. Analyst, Credit Risk", "Maria Santos — Compliance Officer"],
    "approver": "David Park — SVP, Operations Technology",
    "created_at": "2026-03-10",
    "workspace_id": "WS-2026-0147",
    "onboarding_mode": "guided",
}


# ── Document Intake ──────────────────────────────────────────────────

INTAKE_SOURCES = [
    {"name": "Annual Review — FY2024 Portfolio", "type": "repository", "path": "/gcib/lending/annual-reviews/2024/", "count": 342, "status": "connected"},
    {"name": "Annual Review — FY2023 Portfolio", "type": "repository", "path": "/gcib/lending/annual-reviews/2023/", "count": 318, "status": "connected"},
    {"name": "Supplemental Uploads", "type": "upload", "path": "", "count": 24, "status": "uploaded"},
]

INTAKE_SUMMARY = {
    "total_documents": 684,
    "total_pages": 18_426,
    "languages": {"English": 672, "Spanish": 8, "French": 4},
    "avg_pages": 26.9,
    "file_types": {"PDF": 658, "DOCX": 18, "XLSX": 8},
    "quality_scores": {"High (>0.9)": 614, "Medium (0.7–0.9)": 58, "Low (<0.7)": 12},
    "duplicates_detected": 3,
    "sample_strategy": "Stratified by document type and quality tier",
    "sample_size": 50,
    "estimated_processing_time": "~4.2 hours",
}


# ── Task & Schema ────────────────────────────────────────────────────

TASK_MODE = "extract_and_validate"

SCHEMA_FIELDS = [
    {"name": "borrower_name", "type": "Text", "required": True, "criticality": "Critical", "validation": "Entity lookup", "reference_resolver": True, "description": "Legal name of the borrowing entity"},
    {"name": "facility_amount", "type": "Currency", "required": True, "criticality": "Critical", "validation": "Range: $1M–$5B", "reference_resolver": False, "description": "Total committed facility amount"},
    {"name": "maturity_date", "type": "Date", "required": True, "criticality": "Critical", "validation": "ISO 8601, future date", "reference_resolver": False, "description": "Facility maturity or expiration date"},
    {"name": "interest_rate_type", "type": "Enum", "required": True, "criticality": "High", "validation": "Fixed | Floating | Mixed", "reference_resolver": False, "description": "Interest rate structure classification"},
    {"name": "spread_bps", "type": "Numeric", "required": False, "criticality": "High", "validation": "Range: 0–1000 bps", "reference_resolver": False, "description": "Credit spread in basis points over reference rate"},
    {"name": "collateral_type", "type": "Enum", "required": True, "criticality": "High", "validation": "Real Estate | Equipment | Receivables | Unsecured | Other", "reference_resolver": True, "description": "Primary collateral classification"},
    {"name": "risk_rating", "type": "Text", "required": True, "criticality": "Critical", "validation": "Internal rating scale (1–10)", "reference_resolver": True, "description": "Internal credit risk rating at review date"},
    {"name": "financial_covenant_breach", "type": "Boolean", "required": False, "criticality": "Medium", "validation": "True / False / Not Applicable", "reference_resolver": False, "description": "Whether any financial covenant was breached"},
    {"name": "review_date", "type": "Date", "required": True, "criticality": "High", "validation": "ISO 8601, past date", "reference_resolver": False, "description": "Date of the annual credit review"},
    {"name": "analyst_name", "type": "Text", "required": False, "criticality": "Low", "validation": "Free text", "reference_resolver": False, "description": "Name of the reviewing credit analyst"},
    {"name": "next_review_date", "type": "Date", "required": False, "criticality": "Medium", "validation": "ISO 8601, future date", "reference_resolver": False, "description": "Scheduled date for next annual review"},
    {"name": "total_exposure", "type": "Currency", "required": True, "criticality": "Critical", "validation": "Range: $0–$10B", "reference_resolver": False, "description": "Total credit exposure including unfunded commitments"},
]


# ── Ground Truth ─────────────────────────────────────────────────────

GROUND_TRUTH = {
    "mode": "partial",
    "total_documents": 684,
    "labeled_documents": 142,
    "unlabeled_documents": 542,
    "label_coverage_pct": 20.8,
    "field_coverage": {
        "borrower_name": {"labeled": 142, "total": 684, "pct": 100.0},
        "facility_amount": {"labeled": 142, "total": 684, "pct": 100.0},
        "maturity_date": {"labeled": 138, "total": 684, "pct": 97.2},
        "interest_rate_type": {"labeled": 135, "total": 684, "pct": 95.1},
        "spread_bps": {"labeled": 98, "total": 684, "pct": 69.0},
        "collateral_type": {"labeled": 142, "total": 684, "pct": 100.0},
        "risk_rating": {"labeled": 140, "total": 684, "pct": 98.6},
        "financial_covenant_breach": {"labeled": 72, "total": 684, "pct": 50.7},
        "review_date": {"labeled": 142, "total": 684, "pct": 100.0},
        "analyst_name": {"labeled": 45, "total": 684, "pct": 31.7},
        "next_review_date": {"labeled": 110, "total": 684, "pct": 77.5},
        "total_exposure": {"labeled": 139, "total": 684, "pct": 97.9},
    },
    "source": "Extracted from internal Credit Review Management System (CRMS) export",
    "format": "CSV with document_id key mapping",
    "validation_status": "Schema-validated — 3 warnings (date format inconsistencies)",
}


# ── Pipeline Settings ────────────────────────────────────────────────

PIPELINE_SETTINGS = {
    "profile": "balanced",
    "profiles": {
        "recommended": {
            "label": "Recommended",
            "description": "Optimized for this document type and schema complexity. Uses late chunking with semantic strategy, cross-encoder reranking, and regex-first extraction with LLM fallback.",
            "chunking": "Late Chunking — Semantic",
            "extraction": "Regex-first + LLM fallback",
            "reranking": "Cross-encoder (ms-marco-MiniLM-L-6-v2)",
            "retrieval": "Hybrid (BM25 + Vector, RRF fusion)",
            "verification": "Enabled — per-claim citation check",
            "temperature": 0.0,
            "confidence_threshold": 0.85,
        },
        "balanced": {
            "label": "Balanced",
            "description": "Good balance of accuracy and throughput. Suitable for most use cases with moderate schema complexity.",
            "chunking": "Late Chunking — Sliding Window",
            "extraction": "Regex-first + LLM fallback",
            "reranking": "Disabled",
            "retrieval": "Hybrid (BM25 + Vector, RRF fusion)",
            "verification": "Enabled — per-claim citation check",
            "temperature": 0.0,
            "confidence_threshold": 0.80,
        },
        "high_accuracy": {
            "label": "High Accuracy",
            "description": "Maximum accuracy configuration. Higher latency and cost. Recommended for critical regulatory extractions.",
            "chunking": "Late Chunking — Semantic + Clause-Aware",
            "extraction": "LLM-only (GPT-4o)",
            "reranking": "Cross-encoder (ms-marco-MiniLM-L-6-v2)",
            "retrieval": "Hybrid (BM25 + Vector, RRF fusion)",
            "verification": "Enabled — per-claim + cross-field consistency",
            "temperature": 0.0,
            "confidence_threshold": 0.90,
        },
    },
}


# ── Evaluation Plan ──────────────────────────────────────────────────

EVALUATION_PLAN = {
    "target_mode": "shadow",
    "modes": {
        "sandbox": {"label": "Sandbox", "icon": "🧪", "description": "Internal testing only. No production data flow.", "min_accuracy": 0.0},
        "pilot": {"label": "Pilot", "icon": "🔬", "description": "Limited scope with selected reviewers. 5–10% of volume.", "min_accuracy": 0.80},
        "shadow": {"label": "Shadow", "icon": "👁️", "description": "Full volume, parallel to manual process. No downstream impact.", "min_accuracy": 0.85},
        "canary": {"label": "Canary", "icon": "🐤", "description": "Small fraction of live traffic. Monitored with auto-rollback.", "min_accuracy": 0.90},
        "production": {"label": "Production", "icon": "🚀", "description": "Full production deployment. All quality gates must pass.", "min_accuracy": 0.95},
    },
    "quality_gates": [
        {"name": "Overall Extraction Accuracy", "threshold": 0.88, "current": 0.91, "status": "pass"},
        {"name": "Critical Field Accuracy", "threshold": 0.95, "current": 0.96, "status": "pass"},
        {"name": "Entity Resolution Rate", "threshold": 0.90, "current": 0.87, "status": "warn"},
        {"name": "Confidence Calibration (ECE)", "threshold": 0.05, "current": 0.038, "status": "pass"},
        {"name": "Zero Hallucination Rate", "threshold": 1.00, "current": 1.00, "status": "pass"},
        {"name": "Reviewer Agreement (κ)", "threshold": 0.80, "current": 0.83, "status": "pass"},
    ],
    "segments": [
        {"dimension": "Document Type", "slices": ["Annual Review", "Interim Review", "Covenant Compliance"], "coverage": "100%"},
        {"dimension": "Facility Size", "slices": ["<$50M", "$50M–$250M", "$250M–$1B", ">$1B"], "coverage": "98%"},
        {"dimension": "Risk Rating", "slices": ["1–3 (Pass)", "4–5 (Watch)", "6–8 (Substandard)", "9–10 (Doubtful/Loss)"], "coverage": "94%"},
    ],
    "readiness_score": 87,
}


# ── Review & Readiness ───────────────────────────────────────────────

READINESS = {
    "overall_accuracy": 0.912,
    "critical_field_accuracy": 0.964,
    "readiness_score": 87,
    "readiness_label": "Ready for Shadow Deployment",
    "mode_recommendation": "shadow",
    "ground_truth_summary": {
        "mode": "Partial Ground Truth",
        "labeled": 142,
        "total": 684,
        "coverage": "20.8%",
    },
    "blockers": [],
    "risks": [
        {"severity": "Medium", "description": "Entity resolution rate (87%) below target (90%) for 'borrower_name'. May require MCP reference data enrichment.", "mitigation": "Add CRMS entity lookup to MCP reference server."},
        {"severity": "Low", "description": "Financial covenant breach field has only 50.7% ground-truth coverage. Review-based evaluation will supplement.", "mitigation": "Prioritize labeling for covenant breach field in next review cycle."},
    ],
    "field_results": [
        {"field": "borrower_name", "accuracy": 0.96, "confidence": 0.93, "status": "pass", "reviewed": 142, "total": 684},
        {"field": "facility_amount", "accuracy": 0.98, "confidence": 0.95, "status": "pass", "reviewed": 142, "total": 684},
        {"field": "maturity_date", "accuracy": 0.94, "confidence": 0.91, "status": "pass", "reviewed": 138, "total": 684},
        {"field": "interest_rate_type", "accuracy": 0.97, "confidence": 0.94, "status": "pass", "reviewed": 135, "total": 684},
        {"field": "spread_bps", "accuracy": 0.89, "confidence": 0.85, "status": "pass", "reviewed": 98, "total": 684},
        {"field": "collateral_type", "accuracy": 0.93, "confidence": 0.90, "status": "pass", "reviewed": 142, "total": 684},
        {"field": "risk_rating", "accuracy": 0.95, "confidence": 0.92, "status": "pass", "reviewed": 140, "total": 684},
        {"field": "financial_covenant_breach", "accuracy": 0.88, "confidence": 0.82, "status": "warn", "reviewed": 72, "total": 684},
        {"field": "review_date", "accuracy": 0.99, "confidence": 0.97, "status": "pass", "reviewed": 142, "total": 684},
        {"field": "analyst_name", "accuracy": 0.84, "confidence": 0.78, "status": "warn", "reviewed": 45, "total": 684},
        {"field": "next_review_date", "accuracy": 0.92, "confidence": 0.88, "status": "pass", "reviewed": 110, "total": 684},
        {"field": "total_exposure", "accuracy": 0.97, "confidence": 0.94, "status": "pass", "reviewed": 139, "total": 684},
    ],
    "sign_off": {
        "operations": {"name": "Sarah Chen", "status": "approved", "date": "2026-03-12"},
        "compliance": {"name": "Maria Santos", "status": "pending", "date": None},
        "technology": {"name": "David Park", "status": "pending", "date": None},
    },
}


# ── Production Enablement ────────────────────────────────────────────

ENABLEMENT = {
    "target_mode": "shadow",
    "checklist": [
        {"item": "Schema validation passed", "status": "complete", "category": "Technical"},
        {"item": "Pipeline configuration reviewed", "status": "complete", "category": "Technical"},
        {"item": "Evaluation thresholds met", "status": "complete", "category": "Quality"},
        {"item": "Critical field accuracy ≥ 95%", "status": "complete", "category": "Quality"},
        {"item": "Boundary isolation confirmed", "status": "complete", "category": "Governance"},
        {"item": "Data sensitivity classification applied", "status": "complete", "category": "Governance"},
        {"item": "Operations owner sign-off", "status": "complete", "category": "Approval"},
        {"item": "Compliance review sign-off", "status": "pending", "category": "Approval"},
        {"item": "Technology owner sign-off", "status": "pending", "category": "Approval"},
        {"item": "Rollback plan documented", "status": "complete", "category": "Operations"},
        {"item": "Monitoring alerts configured", "status": "complete", "category": "Operations"},
        {"item": "Escalation contacts defined", "status": "complete", "category": "Operations"},
    ],
    "governance": {
        "boundary_key": "gcib|lending_ops|us_occ",
        "data_classification": "Confidential — PII Present",
        "audit_trail": "Enabled — all LLM calls logged with full prompt/response",
        "retention_policy": "7 years per OCC guidance",
    },
}


# ── Post-Go-Live Monitoring ──────────────────────────────────────────

MONITORING = {
    "status": "healthy",
    "uptime": "99.7%",
    "deployment_mode": "Shadow",
    "model_version": "v1.2.0-shadow",
    "run_version": "run-20260314-001",
    "last_updated": "2026-03-15 08:42 UTC",
    "volume": {
        "today": 47,
        "week": 312,
        "month": 1_284,
        "trend": [28, 35, 42, 38, 51, 44, 47, 39, 55, 48, 43, 47, 52, 41],
    },
    "accuracy": {
        "current": 0.912,
        "target": 0.88,
        "trend": [0.89, 0.90, 0.91, 0.90, 0.92, 0.91, 0.91, 0.92, 0.91, 0.90, 0.91, 0.92, 0.91, 0.912],
        "critical_field": 0.964,
    },
    "field_health": [
        {"field": "borrower_name", "accuracy": 0.96, "volume": 1284, "drift": "stable", "status": "healthy"},
        {"field": "facility_amount", "accuracy": 0.98, "volume": 1284, "drift": "stable", "status": "healthy"},
        {"field": "maturity_date", "accuracy": 0.94, "volume": 1284, "drift": "stable", "status": "healthy"},
        {"field": "risk_rating", "accuracy": 0.95, "volume": 1284, "drift": "improving", "status": "healthy"},
        {"field": "collateral_type", "accuracy": 0.91, "volume": 1284, "drift": "declining", "status": "warning"},
        {"field": "total_exposure", "accuracy": 0.97, "volume": 1284, "drift": "stable", "status": "healthy"},
        {"field": "spread_bps", "accuracy": 0.87, "volume": 1284, "drift": "declining", "status": "warning"},
        {"field": "financial_covenant_breach", "accuracy": 0.86, "volume": 1284, "drift": "stable", "status": "attention"},
    ],
    "review_burden": {
        "total_reviews": 312,
        "auto_accepted": 247,
        "manual_reviews": 65,
        "escalations": 4,
        "auto_accept_rate": 0.792,
        "avg_review_time_sec": 42,
    },
    "alerts": [
        {"severity": "warning", "message": "Collateral type accuracy dropped 2.1% over last 7 days", "timestamp": "2026-03-14 14:22 UTC", "acknowledged": False},
        {"severity": "info", "message": "Spread BPS extraction confidence trending below 85% for facilities > $500M", "timestamp": "2026-03-13 09:15 UTC", "acknowledged": True},
        {"severity": "info", "message": "New document subtype detected: 'Covenant Waiver Letter' (3 instances)", "timestamp": "2026-03-12 16:40 UTC", "acknowledged": True},
    ],
    "failure_queue": [
        {"doc_id": "doc-20260314-0023", "error": "Extraction timeout — document exceeds 85 pages", "timestamp": "2026-03-14 11:02 UTC", "status": "pending"},
        {"doc_id": "doc-20260314-0031", "error": "Low confidence on all critical fields (<0.6)", "timestamp": "2026-03-14 13:45 UTC", "status": "pending"},
        {"doc_id": "doc-20260313-0089", "error": "Scanned PDF — OCR quality below threshold", "timestamp": "2026-03-13 15:20 UTC", "status": "resolved"},
        {"doc_id": "doc-20260312-0156", "error": "Entity resolution failed — borrower not in reference data", "timestamp": "2026-03-12 10:05 UTC", "status": "resolved"},
    ],
    "maintenance_actions": [
        {"action": "Retrain extraction model", "status": "available", "description": "142 new labeled samples since last training. Threshold: 100."},
        {"action": "Re-baseline calibration", "status": "available", "description": "ECE drift detected. Re-calibrate confidence scores."},
        {"action": "Update MCP reference data", "status": "recommended", "description": "12 new borrower entities added to CRMS since last sync."},
        {"action": "Rollback to v1.1.0", "status": "available", "description": "Previous stable version available for rollback if needed."},
    ],
}


# ── Wizard Navigation ────────────────────────────────────────────────

WIZARD_STEPS = [
    {"number": 1, "title": "Workspace", "subtitle": "Use Case Setup", "key": "workspace"},
    {"number": 2, "title": "Documents", "subtitle": "Document Intake", "key": "intake"},
    {"number": 3, "title": "Schema", "subtitle": "Task & Schema", "key": "schema"},
    {"number": 4, "title": "Ground Truth", "subtitle": "Labels & Evaluation Mode", "key": "ground_truth"},
    {"number": 5, "title": "Pipeline", "subtitle": "Pipeline Settings", "key": "pipeline"},
    {"number": 6, "title": "Evaluation", "subtitle": "Evaluation Plan", "key": "evaluation"},
    {"number": 7, "title": "Readiness", "subtitle": "Review & Readiness", "key": "readiness"},
    {"number": 8, "title": "Enablement", "subtitle": "Production Enablement", "key": "enablement"},
]
