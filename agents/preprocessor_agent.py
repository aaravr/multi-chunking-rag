"""Preprocessor Agent — owns document processing strategy AND ingestion (MASTER_PROMPT §4.9).

The PreprocessorAgent is the single owner of every document's lifecycle from
PDF upload through to stored, searchable chunks.  It decides HOW to process
a document AND executes the processing pipeline.

Responsibilities:
1. **Strategy decision** — what processing level and chunking algorithm to use.
2. **Multi-chunking** — classify page sections by content type and route
   different parts of the same document to different chunking strategies.
3. **Ingestion orchestration** — classify, canonicalize, chunk, embed, store.
4. **Outcome recording** — feed back results for self-learning.

Processing levels (lightest → heaviest):
- ``skip``          — Empty/corrupt document, nothing to do.
- ``metadata_only`` — Extract text, store as document facts.
- ``single_chunk``  — Embed the whole document as a single chunk.
- ``page_level``    — One chunk per page, independent embeddings.
- ``late_chunking`` — Full macro → child late-chunking pipeline.

Multi-chunking:
  A single document can have different chunking strategies per section.
  For example, a 10-K filing might use ``semantic`` for narrative risk
  factors, ``table_aware`` for financial statements, and ``clause_aware``
  for legal exhibits.  The agent classifies each page's content type
  using triage signals and span metadata, then groups contiguous pages
  of the same type and assigns the optimal strategy per group.

Strategy selection (3-tier):
1. **Deterministic rules**: Document type → known optimal strategy.
2. **Learned outcomes**: Consult past chunking outcomes for same doc type.
3. **Heuristic fallback**: Assess complexity from page count and triage.

Self-learning loop:
- After processing completes, ``record_outcome()`` stores chunk statistics.
- Future documents of the same type consult these outcomes.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from agents.base import BaseAgent
from agents.contracts import (
    AgentMessage,
    ChunkingOutcome,
    ChunkingStrategy,
    PreprocessorInput,
    PreprocessorResult,
    SectionStrategy,
    new_id,
)
from agents.message_bus import MessageBus
from agents.model_gateway import ModelGateway
from core.config import settings

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────
MIN_PAGES_FOR_STRATEGY_SELECTION = 3
MIN_OUTCOMES_FOR_LEARNING = 2
QUALITY_THRESHOLD = 0.5
MIN_PAGES_FOR_MULTI_CHUNKING = 10  # Documents shorter than this use a single strategy

# ── Valid processing levels (ordered lightest → heaviest) ─────────────────
PROCESSING_LEVELS = ("skip", "metadata_only", "single_chunk", "page_level", "late_chunking")

# ── Page-count boundaries for heuristic fallback ──────────────────────────
_SINGLE_CHUNK_MAX_PAGES = 3
_PAGE_LEVEL_MAX_PAGES = 15

# ── Content type → strategy mapping for multi-chunking ───────────────────
#  When multi-chunking is active, each page section is classified by content
#  type and routed to the optimal strategy for that content.

_CONTENT_TYPE_STRATEGIES: Dict[str, ChunkingStrategy] = {
    "narrative": ChunkingStrategy(
        strategy_name="semantic",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=256,
        child_target_tokens=256,
        table_extraction="span",
        heading_aware=True,
        rationale="Narrative sections: semantic splitting at topic boundaries.",
    ),
    "tabular": ChunkingStrategy(
        strategy_name="table_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=192,
        table_extraction="full_page",
        heading_aware=True,
        rationale="Tabular sections: table-aware chunking preserves table boundaries.",
    ),
    "legal": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=192,
        table_extraction="span",
        heading_aware=True,
        rationale="Legal sections: clause-aware splitting at Section/Article boundaries.",
    ),
    "mixed": ChunkingStrategy(
        strategy_name="sliding_window",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=256,
        child_target_tokens=256,
        table_extraction="span",
        heading_aware=True,
        rationale="Mixed content: sliding window with overlap for uniform coverage.",
    ),
}


import re as _re

# ── Clause marker patterns for page content classification ───────────────
_CLAUSE_MARKER_RE = _re.compile(
    r'(?:^|\n)\s*(?:'
    r'(?:SECTION|Section)\s+\d+'
    r'|(?:ARTICLE|Article)\s+[IVXLCDM\d]+'
    r'|\d+\.\d+\s+'
    r'|(?:SCHEDULE|Schedule|EXHIBIT|Exhibit|ANNEX|Annex)\s+[A-Z\d]'
    r'|(?:RECITALS?|WHEREAS|DEFINITIONS?)\b'
    r')',
    _re.MULTILINE | _re.IGNORECASE,
)


def _has_clause_markers(text: str) -> bool:
    """Check if text contains legal clause markers (Section, Article, etc.)."""
    if not text:
        return False
    matches = _CLAUSE_MARKER_RE.findall(text)
    # Need at least 2 markers to classify as legal
    return len(matches) >= 2


# ── Document type → chunking strategy rules (deterministic tier) ─────────
#
#  Covers the full breadth of documents an investment bank handles:
#  identity & KYC, proofs of existence, corporate docs, legal agreements,
#  trading agreements, credit facilities, capital markets, fund docs,
#  compliance certificates, regulatory filings, financial reports, and more.
#
#  Each entry maps a *document_type* OR *classification_label* to a named
#  strategy with tuned parameters based on domain knowledge.

_STRATEGY_RULES: Dict[str, ChunkingStrategy] = {
    # ──────────────────────────────────────────────────────────────────────
    # IDENTITY & KYC DOCUMENTS — metadata_only, no embedding needed
    # ──────────────────────────────────────────────────────────────────────
    "passport": ChunkingStrategy(
        strategy_name="identity_document",
        processing_level="metadata_only",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Passports are single-page identity documents; extract "
                  "metadata (name, DOB, nationality, expiry) only.",
    ),
    "driving_licence": ChunkingStrategy(
        strategy_name="identity_document",
        processing_level="metadata_only",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Driving licences are single-page identity documents; "
                  "extract metadata only.",
    ),
    "driving_license": ChunkingStrategy(
        strategy_name="identity_document",
        processing_level="metadata_only",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Driving licenses are single-page identity documents; "
                  "extract metadata only.",
    ),
    "national_id": ChunkingStrategy(
        strategy_name="identity_document",
        processing_level="metadata_only",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="National ID cards are identity documents; extract "
                  "metadata only.",
    ),
    "visa": ChunkingStrategy(
        strategy_name="identity_document",
        processing_level="metadata_only",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Visa pages are identity/travel documents; extract "
                  "metadata only.",
    ),
    "identity_document": ChunkingStrategy(
        strategy_name="identity_document",
        processing_level="metadata_only",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Generic identity document; extract metadata only.",
    ),
    "kyc_form": ChunkingStrategy(
        strategy_name="kyc_document",
        processing_level="metadata_only",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="KYC forms capture client data fields; store as "
                  "structured facts, no semantic search needed.",
    ),
    "beneficial_ownership": ChunkingStrategy(
        strategy_name="kyc_document",
        processing_level="metadata_only",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Beneficial ownership declarations are structured "
                  "data forms; metadata extraction only.",
    ),
    "sanctions_screening": ChunkingStrategy(
        strategy_name="kyc_document",
        processing_level="metadata_only",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Sanctions screening results are structured; "
                  "metadata extraction only.",
    ),
    "pep_declaration": ChunkingStrategy(
        strategy_name="kyc_document",
        processing_level="metadata_only",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="PEP declarations are structured data forms; "
                  "metadata extraction only.",
    ),

    # ──────────────────────────────────────────────────────────────────────
    # PROOF-OF-EXISTENCE & SIMPLE CERTIFICATES — metadata_only or single_chunk
    # ──────────────────────────────────────────────────────────────────────
    "utility_bill": ChunkingStrategy(
        strategy_name="proof_document",
        processing_level="metadata_only",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Utility bills are proof-of-address documents; "
                  "extract address and date metadata only.",
    ),
    "proof_of_address": ChunkingStrategy(
        strategy_name="proof_document",
        processing_level="metadata_only",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Proof-of-address documents; extract address metadata only.",
    ),
    "certificate_of_incorporation": ChunkingStrategy(
        strategy_name="proof_document",
        processing_level="single_chunk",
        macro_max_tokens=8192,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Certificate of incorporation is typically 1-2 pages; "
                  "embed as single chunk for entity verification queries.",
    ),
    "certificate_of_good_standing": ChunkingStrategy(
        strategy_name="proof_document",
        processing_level="single_chunk",
        macro_max_tokens=8192,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Good standing certificate is typically 1 page; "
                  "embed as single chunk.",
    ),
    "certificate_of_incumbency": ChunkingStrategy(
        strategy_name="proof_document",
        processing_level="single_chunk",
        macro_max_tokens=8192,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Certificate of incumbency is typically 1-3 pages; "
                  "embed as single chunk.",
    ),

    # ──────────────────────────────────────────────────────────────────────
    # CORPORATE GOVERNANCE — single_chunk or page_level
    # ──────────────────────────────────────────────────────────────────────
    "board_resolution": ChunkingStrategy(
        strategy_name="corporate_governance",
        processing_level="single_chunk",
        macro_max_tokens=8192,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Board resolutions are typically 1-3 pages authorising "
                  "specific actions; embed as single chunk.",
    ),
    "power_of_attorney": ChunkingStrategy(
        strategy_name="corporate_governance",
        processing_level="single_chunk",
        macro_max_tokens=8192,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Power of attorney is typically 1-3 pages; embed as "
                  "single chunk for authority verification.",
    ),
    "signing_authority": ChunkingStrategy(
        strategy_name="corporate_governance",
        processing_level="single_chunk",
        macro_max_tokens=8192,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Signing authority lists are short documents; "
                  "embed as single chunk.",
    ),
    "corporate_structure_chart": ChunkingStrategy(
        strategy_name="corporate_governance",
        processing_level="metadata_only",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Corporate structure charts are primarily visual; "
                  "extract entity names as metadata.",
    ),
    "shareholder_register": ChunkingStrategy(
        strategy_name="corporate_governance",
        processing_level="page_level",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="full_page",
        heading_aware=False,
        rationale="Shareholder registers are tabular data; "
                  "page-level extraction with table awareness.",
    ),
    "articles_of_association": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=128,
        child_target_tokens=192,
        table_extraction="none",
        heading_aware=True,
        rationale="Articles of association are clause-structured legal "
                  "documents; clause-aware splitting at article boundaries.",
    ),

    # ──────────────────────────────────────────────────────────────────────
    # COMPLIANCE CERTIFICATES — single_chunk
    # ──────────────────────────────────────────────────────────────────────
    "officer_certificate": ChunkingStrategy(
        strategy_name="compliance_certificate",
        processing_level="single_chunk",
        macro_max_tokens=8192,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Officer certificates are typically 1-3 pages; "
                  "embed as single chunk.",
    ),
    "compliance_certificate": ChunkingStrategy(
        strategy_name="compliance_certificate",
        processing_level="single_chunk",
        macro_max_tokens=8192,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Compliance certificates are short attestation "
                  "documents; embed as single chunk.",
    ),
    "legal_opinion": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=256,
        table_extraction="none",
        heading_aware=True,
        rationale="Legal opinions are multi-page analytical documents "
                  "with structured sections; use late chunking.",
    ),
    "auditor_report": ChunkingStrategy(
        strategy_name="semantic",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=256,
        table_extraction="span",
        heading_aware=True,
        rationale="Auditor reports have narrative opinion sections and "
                  "financial data; standard late chunking.",
    ),

    # ──────────────────────────────────────────────────────────────────────
    # BANK STATEMENTS & TAX DOCUMENTS — page_level (tabular, repetitive)
    # ──────────────────────────────────────────────────────────────────────
    "bank_statement": ChunkingStrategy(
        strategy_name="financial_statement_simple",
        processing_level="page_level",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="full_page",
        heading_aware=False,
        rationale="Bank statements are repetitive tabular pages; "
                  "page-level chunking with table extraction.",
    ),
    "tax_return": ChunkingStrategy(
        strategy_name="financial_statement_simple",
        processing_level="page_level",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="full_page",
        heading_aware=False,
        rationale="Tax returns are form-based tabular documents; "
                  "page-level chunking with table extraction.",
    ),
    "pay_slip": ChunkingStrategy(
        strategy_name="financial_statement_simple",
        processing_level="metadata_only",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Pay slips are single-page structured data; "
                  "extract salary and employer metadata only.",
    ),

    # ──────────────────────────────────────────────────────────────────────
    # INSURANCE — page_level for certificates, late_chunking for policies
    # ──────────────────────────────────────────────────────────────────────
    "insurance_certificate": ChunkingStrategy(
        strategy_name="insurance_document",
        processing_level="single_chunk",
        macro_max_tokens=8192,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Insurance certificates are typically 1-2 pages; "
                  "embed as single chunk.",
    ),
    "insurance_policy": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=128,
        child_target_tokens=192,
        table_extraction="span",
        heading_aware=True,
        rationale="Insurance policies are clause-structured with "
                  "schedules; clause-aware splitting at section boundaries.",
    ),

    # ──────────────────────────────────────────────────────────────────────
    # TRADING AGREEMENTS — late_chunking (complex, clause-heavy)
    # ──────────────────────────────────────────────────────────────────────
    "isda_master_agreement": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=192,
        table_extraction="span",
        heading_aware=True,
        rationale="ISDA Master Agreements are highly structured with "
                  "numbered sections, definitions, and schedules; "
                  "clause-aware splitting preserves clause boundaries.",
    ),
    "isda_schedule": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=192,
        table_extraction="span",
        heading_aware=True,
        rationale="ISDA Schedules amend the master agreement with "
                  "clause-by-clause modifications.",
    ),
    "credit_support_annex": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=192,
        table_extraction="span",
        heading_aware=True,
        rationale="CSAs govern collateral terms; clause-structured "
                  "with tables for thresholds and eligible collateral.",
    ),
    "gmra": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=192,
        table_extraction="span",
        heading_aware=True,
        rationale="Global Master Repurchase Agreements are clause-structured "
                  "trading agreements with defined terms.",
    ),
    "gmsla": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=192,
        table_extraction="span",
        heading_aware=True,
        rationale="Global Master Securities Lending Agreements follow "
                  "structured clause patterns similar to GMRA.",
    ),
    "prime_brokerage_agreement": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=256,
        table_extraction="span",
        heading_aware=True,
        rationale="Prime brokerage agreements are comprehensive trading "
                  "docs with multiple service schedules.",
    ),
    "futures_agreement": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=192,
        table_extraction="span",
        heading_aware=True,
        rationale="Futures/clearing agreements follow structured "
                  "clause patterns with product schedules.",
    ),
    "trading_agreement": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=192,
        table_extraction="span",
        heading_aware=True,
        rationale="Generic trading agreements: clause-oriented "
                  "with defined terms and schedules.",
    ),

    # ──────────────────────────────────────────────────────────────────────
    # CREDIT & LENDING — late_chunking (long, clause-heavy)
    # ──────────────────────────────────────────────────────────────────────
    "credit_agreement": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=256,
        table_extraction="span",
        heading_aware=True,
        rationale="Credit agreements are long-form legal documents "
                  "with covenants, conditions, and schedules; "
                  "clause-aware splitting at Section/Article boundaries.",
    ),
    "facility_agreement": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=256,
        table_extraction="span",
        heading_aware=True,
        rationale="Facility agreements follow credit agreement "
                  "structure; clause-aware splitting.",
    ),
    "term_sheet": ChunkingStrategy(
        strategy_name="deal_summary",
        processing_level="single_chunk",
        macro_max_tokens=8192,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Term sheets are typically 2-5 pages summarising "
                  "key deal terms; single chunk preserves context.",
    ),
    "commitment_letter": ChunkingStrategy(
        strategy_name="deal_summary",
        processing_level="single_chunk",
        macro_max_tokens=8192,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Commitment letters are short documents outlining "
                  "lending commitment; embed as single chunk.",
    ),
    "loan_agreement": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=128,
        child_target_tokens=192,
        table_extraction="none",
        heading_aware=True,
        rationale="Loan agreements follow contract clause patterns; "
                  "clause-aware splitting at section boundaries.",
    ),
    "security_agreement": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=256,
        table_extraction="span",
        heading_aware=True,
        rationale="Security agreements detail collateral pledged; "
                  "clause-aware splitting with schedules.",
    ),
    "guarantee": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=128,
        child_target_tokens=192,
        table_extraction="none",
        heading_aware=True,
        rationale="Guarantee agreements are clause-structured legal "
                  "documents; clause-aware splitting.",
    ),
    "intercreditor_agreement": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=256,
        table_extraction="none",
        heading_aware=True,
        rationale="Intercreditor agreements govern creditor priorities; "
                  "clause-aware splitting for waterfall provisions.",
    ),

    # ──────────────────────────────────────────────────────────────────────
    # CAPITAL MARKETS — late_chunking (large, complex documents)
    # ──────────────────────────────────────────────────────────────────────
    "prospectus": ChunkingStrategy(
        strategy_name="topic_segmentation",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=512,
        child_target_tokens=384,
        table_extraction="span",
        heading_aware=True,
        rationale="Prospectuses are 100+ page documents with distinct "
                  "topic sections (risk factors, financials, legal); "
                  "topic segmentation detects natural boundaries.",
    ),
    "offering_memorandum": ChunkingStrategy(
        strategy_name="topic_segmentation",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=512,
        child_target_tokens=384,
        table_extraction="span",
        heading_aware=True,
        rationale="Offering memoranda follow prospectus structure; "
                  "topic segmentation for distinct disclosure sections.",
    ),
    "pricing_supplement": ChunkingStrategy(
        strategy_name="capital_markets",
        processing_level="page_level",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="full_page",
        heading_aware=False,
        rationale="Pricing supplements are short, tabular documents "
                  "with key terms and pricing details.",
    ),
    "base_indenture": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=256,
        table_extraction="none",
        heading_aware=True,
        rationale="Base indentures are long-form legal agreements "
                  "governing bond terms; clause-aware splitting.",
    ),
    "supplemental_indenture": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=256,
        table_extraction="none",
        heading_aware=True,
        rationale="Supplemental indentures amend base indentures; "
                  "clause-aware splitting at article boundaries.",
    ),

    # ──────────────────────────────────────────────────────────────────────
    # FUND DOCUMENTS — varying complexity
    # ──────────────────────────────────────────────────────────────────────
    "fund_prospectus": ChunkingStrategy(
        strategy_name="topic_segmentation",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=512,
        child_target_tokens=384,
        table_extraction="span",
        heading_aware=True,
        rationale="Fund prospectuses are comprehensive disclosure "
                  "documents; topic segmentation for distinct sections.",
    ),
    "private_placement_memorandum": ChunkingStrategy(
        strategy_name="topic_segmentation",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=512,
        child_target_tokens=384,
        table_extraction="span",
        heading_aware=True,
        rationale="PPMs are detailed offering documents; topic "
                  "segmentation for distinct disclosure sections.",
    ),
    "subscription_agreement": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=128,
        child_target_tokens=192,
        table_extraction="none",
        heading_aware=True,
        rationale="Subscription agreements are clause-structured "
                  "investor commitment documents.",
    ),
    "side_letter": ChunkingStrategy(
        strategy_name="fund_document",
        processing_level="page_level",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=True,
        rationale="Side letters are typically 3-10 pages of specific "
                  "investor terms; page-level chunking sufficient.",
    ),
    "limited_partnership_agreement": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=256,
        table_extraction="none",
        heading_aware=True,
        rationale="LPAs are long-form legal agreements governing "
                  "fund operations; clause-aware splitting.",
    ),

    # ──────────────────────────────────────────────────────────────────────
    # VALUATION & ANALYSIS — late_chunking
    # ──────────────────────────────────────────────────────────────────────
    "valuation_report": ChunkingStrategy(
        strategy_name="semantic",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=256,
        child_target_tokens=320,
        table_extraction="span",
        heading_aware=True,
        rationale="Valuation reports mix narrative analysis with "
                  "financial tables; standard late chunking.",
    ),
    "appraisal": ChunkingStrategy(
        strategy_name="semantic",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=256,
        child_target_tokens=320,
        table_extraction="span",
        heading_aware=True,
        rationale="Appraisal reports are analytical with embedded "
                  "tables and comparables.",
    ),
    "fairness_opinion": ChunkingStrategy(
        strategy_name="semantic",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=256,
        table_extraction="span",
        heading_aware=True,
        rationale="Fairness opinions are moderate-length analytical "
                  "documents with financial exhibits.",
    ),
    "research_report": ChunkingStrategy(
        strategy_name="semantic",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=256,
        child_target_tokens=320,
        table_extraction="span",
        heading_aware=True,
        rationale="Research reports combine narrative analysis with "
                  "financial models and charts.",
    ),

    # ──────────────────────────────────────────────────────────────────────
    # SEC FILINGS — late_chunking (dense text, long sections)
    # ──────────────────────────────────────────────────────────────────────
    "10-K": ChunkingStrategy(
        strategy_name="parent_child",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=512,
        child_target_tokens=384,
        table_extraction="span",
        heading_aware=True,
        rationale="SEC 10-K filings have long narrative sections; "
                  "parent-child hierarchy enables both broad and "
                  "precise retrieval across Items.",
    ),
    "10-Q": ChunkingStrategy(
        strategy_name="parent_child",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=512,
        child_target_tokens=384,
        table_extraction="span",
        heading_aware=True,
        rationale="SEC 10-Q quarterly filings follow 10-K structure; "
                  "parent-child hierarchy for broad+precise retrieval.",
    ),
    "20-F": ChunkingStrategy(
        strategy_name="parent_child",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=512,
        child_target_tokens=384,
        table_extraction="span",
        heading_aware=True,
        rationale="SEC 20-F foreign filings follow 10-K structure; "
                  "parent-child hierarchy.",
    ),
    "sec_filing": ChunkingStrategy(
        strategy_name="parent_child",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=512,
        child_target_tokens=384,
        table_extraction="span",
        heading_aware=True,
        rationale="SEC filings benefit from parent-child hierarchy "
                  "for multi-granularity retrieval.",
    ),
    "proxy_statement": ChunkingStrategy(
        strategy_name="parent_child",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=512,
        child_target_tokens=384,
        table_extraction="span",
        heading_aware=True,
        rationale="Proxy statements (DEF 14A) are structured SEC "
                  "filings; parent-child for compensation tables "
                  "and proposal sections.",
    ),
    "8-K": ChunkingStrategy(
        strategy_name="sec_filing",
        processing_level="page_level",
        macro_max_tokens=0,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="span",
        heading_aware=True,
        rationale="8-K current reports are typically short event "
                  "disclosures; page-level chunking sufficient.",
    ),

    # ──────────────────────────────────────────────────────────────────────
    # FINANCIAL REPORTS — late_chunking (balanced narrative & tables)
    # ──────────────────────────────────────────────────────────────────────
    "annual_report": ChunkingStrategy(
        strategy_name="semantic",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=256,
        child_target_tokens=256,
        table_extraction="span",
        heading_aware=True,
        rationale="Annual reports are narrative-heavy with thematic "
                  "sections; semantic splitting at topic boundaries.",
    ),
    "financial_report": ChunkingStrategy(
        strategy_name="semantic",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=256,
        child_target_tokens=256,
        table_extraction="span",
        heading_aware=True,
        rationale="Financial reports: narrative-heavy; semantic "
                  "splitting preserves thematic coherence.",
    ),
    "interim_report": ChunkingStrategy(
        strategy_name="semantic",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=256,
        child_target_tokens=256,
        table_extraction="span",
        heading_aware=True,
        rationale="Interim reports follow annual report structure; "
                  "semantic splitting for narrative sections.",
    ),
    "financial_statement": ChunkingStrategy(
        strategy_name="table_aware",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=256,
        child_target_tokens=256,
        table_extraction="full_page",
        heading_aware=True,
        rationale="Financial statements are table-heavy; table-aware "
                  "chunking keeps financial tables as whole units.",
    ),

    # ──────────────────────────────────────────────────────────────────────
    # REGULATORY & BASEL — late_chunking (highly structured, table-heavy)
    # ──────────────────────────────────────────────────────────────────────
    "pillar3_disclosure": ChunkingStrategy(
        strategy_name="table_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=192,
        table_extraction="full_page",
        heading_aware=True,
        rationale="Pillar 3 disclosures are table-heavy; table-aware "
                  "chunking keeps regulatory tables as whole units.",
    ),
    "basel_regulatory": ChunkingStrategy(
        strategy_name="table_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=192,
        table_extraction="full_page",
        heading_aware=True,
        rationale="Basel regulatory docs are table-heavy; table-aware "
                  "chunking preserves table boundaries.",
    ),
    "regulatory_disclosure": ChunkingStrategy(
        strategy_name="table_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=192,
        table_extraction="full_page",
        heading_aware=True,
        rationale="Regulatory disclosures: table-aware chunking "
                  "for structured tables and disclosures.",
    ),
    "regulatory_filing": ChunkingStrategy(
        strategy_name="table_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=256,
        child_target_tokens=192,
        table_extraction="full_page",
        heading_aware=True,
        rationale="Regulatory filings: table-aware chunking for "
                  "mandatory tables and structured disclosures.",
    ),

    # ──────────────────────────────────────────────────────────────────────
    # LEGAL DOCUMENTS — late_chunking (clause-oriented)
    # ──────────────────────────────────────────────────────────────────────
    "contract": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=128,
        child_target_tokens=192,
        table_extraction="none",
        heading_aware=True,
        rationale="Contracts are clause-structured; clause-aware "
                  "splitting at Section/Article boundaries.",
    ),
    "legal_document": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=128,
        child_target_tokens=192,
        table_extraction="none",
        heading_aware=True,
        rationale="Legal documents: clause-aware splitting at "
                  "clause/section boundaries.",
    ),
    "indenture": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=128,
        child_target_tokens=192,
        table_extraction="none",
        heading_aware=True,
        rationale="Indentures: clause-aware splitting at article "
                  "and section boundaries.",
    ),
    "nda": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="single_chunk",
        macro_max_tokens=8192,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="NDAs are typically 2-5 pages; embed as single "
                  "chunk to preserve full context.",
    ),
    "engagement_letter": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="single_chunk",
        macro_max_tokens=8192,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Engagement letters are typically 3-8 pages; "
                  "single chunk preserves scope and terms.",
    ),
    "fee_letter": ChunkingStrategy(
        strategy_name="deal_summary",
        processing_level="single_chunk",
        macro_max_tokens=8192,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Fee letters are short documents detailing fee "
                  "arrangements; embed as single chunk.",
    ),

    # ──────────────────────────────────────────────────────────────────────
    # ESG / SUSTAINABILITY — late_chunking (narrative-heavy)
    # ──────────────────────────────────────────────────────────────────────
    "esg_report": ChunkingStrategy(
        strategy_name="semantic",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=256,
        child_target_tokens=320,
        table_extraction="span",
        heading_aware=True,
        rationale="ESG reports are narrative-heavy; semantic splitting "
                  "detects thematic boundaries between ESG pillars.",
    ),
    "sustainability": ChunkingStrategy(
        strategy_name="semantic",
        processing_level="late_chunking",
        macro_max_tokens=8192,
        macro_overlap_tokens=256,
        child_target_tokens=320,
        table_extraction="span",
        heading_aware=True,
        rationale="Sustainability reports: narrative-heavy; semantic "
                  "splitting at thematic boundaries.",
    ),

    # ──────────────────────────────────────────────────────────────────────
    # CORRESPONDENCE & NOTICES — single_chunk or page_level
    # ──────────────────────────────────────────────────────────────────────
    "notice": ChunkingStrategy(
        strategy_name="correspondence",
        processing_level="single_chunk",
        macro_max_tokens=8192,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Notices are typically 1-3 pages; embed as single chunk.",
    ),
    "waiver_letter": ChunkingStrategy(
        strategy_name="correspondence",
        processing_level="single_chunk",
        macro_max_tokens=8192,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Waiver letters are short documents; single chunk.",
    ),
    "consent_letter": ChunkingStrategy(
        strategy_name="correspondence",
        processing_level="single_chunk",
        macro_max_tokens=8192,
        macro_overlap_tokens=0,
        child_target_tokens=0,
        table_extraction="none",
        heading_aware=False,
        rationale="Consent letters are short documents; single chunk.",
    ),
    "amendment": ChunkingStrategy(
        strategy_name="clause_aware",
        processing_level="late_chunking",
        macro_max_tokens=4096,
        macro_overlap_tokens=128,
        child_target_tokens=192,
        table_extraction="none",
        heading_aware=True,
        rationale="Amendments modify existing agreements; "
                  "clause-structured with cross-references.",
    ),
}

# ── Default strategy for unknown document types ──────────────────────────
_DEFAULT_STRATEGY = ChunkingStrategy(
    strategy_name="late_chunking",
    processing_level="late_chunking",
    macro_max_tokens=8192,
    macro_overlap_tokens=256,
    child_target_tokens=256,
    table_extraction="span",
    heading_aware=True,
    rationale="Default late-chunking strategy for unclassified documents.",
)

# ── Skip strategy for very short documents ───────────────────────────────
_SKIP_STRATEGY = ChunkingStrategy(
    strategy_name="skip",
    processing_level="skip",
    macro_max_tokens=0,
    macro_overlap_tokens=0,
    child_target_tokens=0,
    table_extraction="none",
    heading_aware=False,
    rationale="Document is empty or has no extractable text content.",
)

# ── Metadata-only strategy for heuristic fallback ────────────────────────
_METADATA_ONLY_STRATEGY = ChunkingStrategy(
    strategy_name="metadata_only",
    processing_level="metadata_only",
    macro_max_tokens=0,
    macro_overlap_tokens=0,
    child_target_tokens=0,
    table_extraction="none",
    heading_aware=False,
    rationale="Document is very short with minimal text; extract "
              "metadata only.",
)

# ── Single-chunk strategy for heuristic fallback ─────────────────────────
_SINGLE_CHUNK_STRATEGY = ChunkingStrategy(
    strategy_name="single_chunk",
    processing_level="single_chunk",
    macro_max_tokens=8192,
    macro_overlap_tokens=0,
    child_target_tokens=0,
    table_extraction="none",
    heading_aware=False,
    rationale="Short document (≤3 pages); embed as single chunk.",
)

# ── Page-level strategy for heuristic fallback ───────────────────────────
_PAGE_LEVEL_STRATEGY = ChunkingStrategy(
    strategy_name="page_level",
    processing_level="page_level",
    macro_max_tokens=0,
    macro_overlap_tokens=0,
    child_target_tokens=0,
    table_extraction="span",
    heading_aware=False,
    rationale="Moderate-length document; page-level chunking.",
)


# ── Outcome Store: learns from past chunking results ─────────────────────

class OutcomeStore:
    """In-memory store for chunking outcomes, keyed by (document_type, classification_label).

    Persists outcomes to PostgreSQL ``chunking_outcomes`` table when available,
    falls back to in-memory storage for tests or when no DB is configured.
    """

    def __init__(self, use_db: bool = False) -> None:
        self._use_db = use_db
        self._outcomes: Dict[str, List[ChunkingOutcome]] = defaultdict(list)

    def record(self, outcome: ChunkingOutcome) -> None:
        """Store a chunking outcome for future strategy selection."""
        key = _outcome_key(outcome.document_type, outcome.classification_label)
        self._outcomes[key].append(outcome)
        if self._use_db:
            self._persist_to_db(outcome)

    def lookup(
        self,
        document_type: str,
        classification_label: str,
        min_outcomes: int = MIN_OUTCOMES_FOR_LEARNING,
        min_quality: float = QUALITY_THRESHOLD,
    ) -> Optional[ChunkingStrategy]:
        """Find the best strategy for a given doc type based on past outcomes.

        Returns None if insufficient data or no strategy meets quality threshold.
        """
        key = _outcome_key(document_type, classification_label)
        outcomes = self._outcomes.get(key, [])

        # Also try DB if in-memory is insufficient
        if len(outcomes) < min_outcomes and self._use_db:
            db_outcomes = self._fetch_from_db(document_type, classification_label)
            # Merge, dedup by doc_id
            seen = {o.doc_id for o in outcomes}
            for o in db_outcomes:
                if o.doc_id not in seen:
                    outcomes.append(o)
                    seen.add(o.doc_id)

        if len(outcomes) < min_outcomes:
            return None

        # Group by strategy_name and pick the one with best average quality
        strategy_scores: Dict[str, List[float]] = defaultdict(list)
        strategy_params: Dict[str, ChunkingOutcome] = {}
        for o in outcomes:
            strategy_scores[o.strategy_name].append(o.quality_score)
            strategy_params[o.strategy_name] = o  # Keep latest params

        best_strategy = None
        best_avg_quality = -1.0
        for name, scores in strategy_scores.items():
            avg = sum(scores) / len(scores)
            if avg >= min_quality and avg > best_avg_quality:
                best_avg_quality = avg
                best_strategy = name

        if best_strategy is None:
            return None

        # Reconstruct strategy from the deterministic rules if available,
        # otherwise use the outcome's parameters as-is
        if best_strategy in _STRATEGY_RULES:
            return _STRATEGY_RULES[best_strategy]

        # Build from learned outcome
        ref = strategy_params[best_strategy]
        return ChunkingStrategy(
            strategy_name=best_strategy,
            processing_level="late_chunking",
            macro_max_tokens=8192,
            macro_overlap_tokens=256,
            child_target_tokens=256,
            table_extraction="span" if ref.table_chunk_ratio > 0.1 else "none",
            heading_aware=ref.heading_chunk_ratio > 0.05,
            rationale=f"Learned from {len(strategy_scores[best_strategy])} prior "
                      f"documents (avg quality={best_avg_quality:.2f}).",
        )

    def get_doc_ids_for_type(
        self, document_type: str, classification_label: str
    ) -> List[str]:
        """Return doc_ids of prior documents that informed strategy for this type."""
        key = _outcome_key(document_type, classification_label)
        return [o.doc_id for o in self._outcomes.get(key, [])]

    @property
    def total_outcomes(self) -> int:
        return sum(len(v) for v in self._outcomes.values())

    def _persist_to_db(self, outcome: ChunkingOutcome) -> None:
        """Store outcome in PostgreSQL (best-effort)."""
        try:
            from storage.db_pool import get_connection
            from storage import repo
            with get_connection() as conn:
                repo.insert_chunking_outcome(conn, outcome)
                conn.commit()
        except Exception as exc:
            logger.debug("Failed to persist chunking outcome to DB: %s", exc)

    def _fetch_from_db(
        self, document_type: str, classification_label: str
    ) -> List[ChunkingOutcome]:
        """Fetch outcomes from PostgreSQL (best-effort)."""
        try:
            from storage.db_pool import get_connection
            from storage import repo
            with get_connection() as conn:
                return repo.fetch_chunking_outcomes(
                    conn, document_type, classification_label
                )
        except Exception as exc:
            logger.debug("Failed to fetch chunking outcomes from DB: %s", exc)
            return []


def _outcome_key(document_type: str, classification_label: str) -> str:
    return f"{document_type}::{classification_label}"


# ── Module-level singleton ───────────────────────────────────────────────

_outcome_store: Optional[OutcomeStore] = None


def get_outcome_store() -> OutcomeStore:
    """Get or create the module-level OutcomeStore singleton."""
    global _outcome_store
    if _outcome_store is None:
        _outcome_store = OutcomeStore(use_db=bool(settings.database_url))
    return _outcome_store


# ── Preprocessor Agent ───────────────────────────────────────────────────

class PreprocessorAgent(BaseAgent):
    """Determines the processing strategy for every document entering the platform.

    Responsible for the full spectrum of investment banking documents:
    from single-page identity documents (passport, driving licence) through
    to complex multi-hundred-page trading agreements (ISDA, CSA, GMRA) and
    regulatory filings (10-K, Pillar 3).

    Decision flow (3-tier):
    1. Check if document is too short/empty → skip
    2. Deterministic: document_type or classification_label → known strategy
    3. Learned: consult OutcomeStore for past outcomes on same doc type
    4. Heuristic: assess document complexity from page count and triage
       signals to pick the lightest sufficient processing level

    The agent also exposes ``record_outcome()`` for the ingestion pipeline
    to feed back processing results, enabling the learning loop.
    """

    agent_name = "preprocessor"

    def __init__(
        self,
        bus: MessageBus,
        gateway: Optional[ModelGateway] = None,
        outcome_store: Optional[OutcomeStore] = None,
    ) -> None:
        super().__init__(bus, gateway)
        self._outcome_store = outcome_store or get_outcome_store()

    def handle_message(self, message: AgentMessage) -> PreprocessorResult:
        """Handle an incoming preprocessor request via the message bus."""
        payload = message.payload
        inp = PreprocessorInput(
            doc_id=payload.get("doc_id", ""),
            filename=payload.get("filename", ""),
            page_count=payload.get("page_count", 0),
            document_type=payload.get("document_type"),
            classification_label=payload.get("classification_label"),
            classification_confidence=payload.get("classification_confidence", 0.0),
            triage_summary=payload.get("triage_summary", {}),
            front_matter_text=payload.get("front_matter_text", ""),
        )
        return self.determine_strategy(inp)

    def determine_strategy(self, inp: PreprocessorInput) -> PreprocessorResult:
        """Core decision logic: determine processing strategy for a document.

        Returns a PreprocessorResult with:
        - requires_chunking: whether to proceed with chunking/embedding
        - chunking_strategy: the selected ChunkingStrategy (incl. processing_level)
        - confidence: how confident the decision is
        - decision_method: which tier made the decision
        """
        start_ms = time.monotonic() * 1000
        warnings: List[str] = []

        # ── Tier 0: Skip check ───────────────────────────────────────
        if inp.page_count == 0:
            return PreprocessorResult(
                doc_id=inp.doc_id,
                requires_chunking=False,
                chunking_strategy=_SKIP_STRATEGY,
                confidence=1.0,
                decision_method="deterministic",
                warnings=["Document has 0 pages — skipping processing."],
            )

        # Check triage summary for empty documents
        triage = inp.triage_summary
        if triage:
            total_text = triage.get("total_text_length", -1)
            if total_text == 0:
                return PreprocessorResult(
                    doc_id=inp.doc_id,
                    requires_chunking=False,
                    chunking_strategy=_SKIP_STRATEGY,
                    confidence=1.0,
                    decision_method="deterministic",
                    warnings=["Document has no extractable text — skipping processing."],
                )

        # ── Tier 1: Deterministic rules ──────────────────────────────
        strategy = self._deterministic_lookup(
            inp.document_type, inp.classification_label
        )
        if strategy is not None:
            # Adjust strategy based on triage signals
            strategy = self._adjust_for_triage(strategy, triage, warnings)
            requires_chunking = strategy.processing_level not in ("skip", "metadata_only")
            elapsed = time.monotonic() * 1000 - start_ms
            logger.info(
                "Preprocessor: deterministic strategy '%s' (level=%s) for "
                "doc %s (type=%s, label=%s) in %.1fms",
                strategy.strategy_name, strategy.processing_level,
                inp.doc_id, inp.document_type, inp.classification_label,
                elapsed,
            )
            return PreprocessorResult(
                doc_id=inp.doc_id,
                requires_chunking=requires_chunking,
                chunking_strategy=strategy,
                confidence=0.95,
                decision_method="deterministic",
                warnings=warnings,
            )

        # ── Tier 2: Learned from past outcomes ───────────────────────
        if inp.document_type and inp.classification_label:
            learned = self._outcome_store.lookup(
                inp.document_type, inp.classification_label
            )
            if learned is not None:
                learned = self._adjust_for_triage(learned, triage, warnings)
                prior_ids = self._outcome_store.get_doc_ids_for_type(
                    inp.document_type, inp.classification_label
                )
                requires_chunking = learned.processing_level not in ("skip", "metadata_only")
                elapsed = time.monotonic() * 1000 - start_ms
                logger.info(
                    "Preprocessor: learned strategy '%s' (level=%s) for "
                    "doc %s from %d prior docs in %.1fms",
                    learned.strategy_name, learned.processing_level,
                    inp.doc_id, len(prior_ids), elapsed,
                )
                return PreprocessorResult(
                    doc_id=inp.doc_id,
                    requires_chunking=requires_chunking,
                    chunking_strategy=learned,
                    confidence=0.75,
                    decision_method="learned",
                    learned_from_doc_ids=prior_ids[:10],
                    warnings=warnings,
                )

        # ── Tier 3: Heuristic fallback ─────────────────────────────
        strategy = self._assess_complexity(inp, warnings)
        strategy = self._adjust_for_triage(strategy, triage, warnings)
        requires_chunking = strategy.processing_level not in ("skip", "metadata_only")
        elapsed = time.monotonic() * 1000 - start_ms
        logger.info(
            "Preprocessor: heuristic strategy '%s' (level=%s) for doc %s "
            "in %.1fms",
            strategy.strategy_name, strategy.processing_level,
            inp.doc_id, elapsed,
        )
        return PreprocessorResult(
            doc_id=inp.doc_id,
            requires_chunking=requires_chunking,
            chunking_strategy=strategy,
            confidence=0.5,
            decision_method="heuristic",
            warnings=warnings,
        )

    def record_outcome(self, outcome: ChunkingOutcome) -> None:
        """Record a processing outcome for future learning.

        Called by the ingestion pipeline after processing completes.
        The outcome's statistics (chunk count, type ratios, timing)
        are used to evaluate and improve strategy selection.
        """
        self._outcome_store.record(outcome)
        logger.info(
            "Preprocessor: recorded outcome for doc %s — strategy=%s, "
            "chunks=%d, quality=%.2f",
            outcome.doc_id, outcome.strategy_name,
            outcome.chunk_count, outcome.quality_score,
        )

    @property
    def outcome_store(self) -> OutcomeStore:
        return self._outcome_store

    # ── Internal helpers ─────────────────────────────────────────────

    def _deterministic_lookup(
        self,
        document_type: Optional[str],
        classification_label: Optional[str],
    ) -> Optional[ChunkingStrategy]:
        """Look up a known strategy by document_type or classification_label."""
        if document_type and document_type in _STRATEGY_RULES:
            return _STRATEGY_RULES[document_type]
        if classification_label and classification_label in _STRATEGY_RULES:
            return _STRATEGY_RULES[classification_label]
        return None

    def _assess_complexity(
        self,
        inp: PreprocessorInput,
        warnings: List[str],
    ) -> ChunkingStrategy:
        """Assess document complexity from page count and triage signals.

        Selects the lightest processing level that is appropriate:
        - 1 page with very little text → metadata_only
        - ≤3 pages → single_chunk
        - ≤15 pages → page_level
        - >15 pages → full late_chunking
        """
        triage = inp.triage_summary
        avg_text = triage.get("avg_text_length", 500) if triage else 500
        page_count = inp.page_count

        # Very short document with minimal text → metadata only
        if page_count == 1 and avg_text < 200:
            return _METADATA_ONLY_STRATEGY

        # Short documents → single chunk
        if page_count <= _SINGLE_CHUNK_MAX_PAGES:
            return _SINGLE_CHUNK_STRATEGY

        # Moderate documents → page level
        if page_count <= _PAGE_LEVEL_MAX_PAGES:
            return _PAGE_LEVEL_STRATEGY

        # Long/complex documents → full late chunking
        return _DEFAULT_STRATEGY

    def _adjust_for_triage(
        self,
        strategy: ChunkingStrategy,
        triage: Dict[str, Any],
        warnings: List[str],
    ) -> ChunkingStrategy:
        """Adjust strategy parameters based on page triage signals.

        If triage data shows the document is heavily image-based or has
        high layout complexity, we may want to tweak parameters.
        """
        if not triage:
            return strategy

        di_page_ratio = triage.get("di_page_ratio", 0.0)
        avg_image_coverage = triage.get("avg_image_coverage", 0.0)

        # If most pages needed DI (Azure Document Intelligence), the document
        # is likely image-heavy / scanned. Use full-page table extraction.
        if di_page_ratio > 0.5 and strategy.table_extraction == "span":
            warnings.append(
                f"High DI ratio ({di_page_ratio:.0%}) — upgrading table "
                "extraction to full_page."
            )
            return ChunkingStrategy(
                strategy_name=strategy.strategy_name,
                processing_level=strategy.processing_level,
                macro_max_tokens=strategy.macro_max_tokens,
                macro_overlap_tokens=strategy.macro_overlap_tokens,
                child_target_tokens=strategy.child_target_tokens,
                table_extraction="full_page",
                heading_aware=strategy.heading_aware,
                rationale=strategy.rationale + " [adjusted: full_page table extraction for DI-heavy doc]",
            )

        # If average image coverage is very high, warn about potential
        # low text yield
        if avg_image_coverage > 0.7:
            warnings.append(
                f"Very high image coverage ({avg_image_coverage:.0%}) — "
                "processing may produce limited text content."
            )

        return strategy

    # ── Multi-chunking: section-level strategy assignment ────────────

    def classify_page_sections(
        self,
        canonical_pages: list,
        default_strategy: ChunkingStrategy,
    ) -> List[SectionStrategy]:
        """Classify pages by content type and assign per-section strategies.

        Groups contiguous pages of the same content type into sections,
        then picks the optimal chunking strategy for each section.

        Content types detected:
        - ``tabular``: >50% of spans are tables → ``table_aware``
        - ``legal``: spans contain clause/section markers → ``clause_aware``
        - ``narrative``: mostly prose text → ``semantic``
        - ``boilerplate``: very short text, likely cover/TOC → skipped or merged
        - ``mixed``: no dominant type → uses document-level default

        Returns a list of SectionStrategy covering all pages.
        """
        if not canonical_pages or len(canonical_pages) < MIN_PAGES_FOR_MULTI_CHUNKING:
            return []

        # Step 1: classify each page's content type
        page_types: List[str] = []
        for page in canonical_pages:
            page_types.append(self._classify_page_content(page))

        # Step 2: group contiguous pages of the same type
        sections: List[SectionStrategy] = []
        group_start = 0
        for i in range(1, len(page_types)):
            if page_types[i] != page_types[group_start]:
                section = self._make_section_strategy(
                    page_start=canonical_pages[group_start].page_number,
                    page_end=canonical_pages[i - 1].page_number,
                    content_type=page_types[group_start],
                    default_strategy=default_strategy,
                )
                sections.append(section)
                group_start = i

        # Final group
        section = self._make_section_strategy(
            page_start=canonical_pages[group_start].page_number,
            page_end=canonical_pages[-1].page_number,
            content_type=page_types[group_start],
            default_strategy=default_strategy,
        )
        sections.append(section)

        # Step 3: merge tiny sections (<2 pages) into neighbours
        sections = self._merge_small_sections(sections, default_strategy)

        return sections

    def _classify_page_content(self, page) -> str:
        """Classify a single canonical page by its dominant content type."""
        if not page.text or len(page.text.strip()) < 50:
            return "boilerplate"

        spans = page.spans if hasattr(page, "spans") else []
        total_spans = len(spans) if spans else 0

        if total_spans == 0:
            return "narrative"

        table_spans = sum(1 for s in spans if s.is_table)
        table_ratio = table_spans / total_spans if total_spans > 0 else 0.0

        if table_ratio > 0.5:
            return "tabular"

        # Check for legal clause markers in text
        text = page.text
        if _has_clause_markers(text):
            return "legal"

        return "narrative"

    def _make_section_strategy(
        self,
        page_start: int,
        page_end: int,
        content_type: str,
        default_strategy: ChunkingStrategy,
    ) -> SectionStrategy:
        """Create a SectionStrategy with the optimal chunking algorithm for the content type."""
        strategy = _CONTENT_TYPE_STRATEGIES.get(content_type)
        if strategy is None:
            strategy = default_strategy

        return SectionStrategy(
            page_start=page_start,
            page_end=page_end,
            content_type=content_type,
            chunking_strategy=strategy,
            rationale=f"Pages {page_start}-{page_end}: {content_type} content → "
                      f"{strategy.strategy_name}",
        )

    def _merge_small_sections(
        self,
        sections: List[SectionStrategy],
        default_strategy: ChunkingStrategy,
    ) -> List[SectionStrategy]:
        """Merge very small sections (single page) into adjacent sections."""
        if len(sections) <= 1:
            return sections

        merged: List[SectionStrategy] = [sections[0]]
        for section in sections[1:]:
            prev = merged[-1]
            section_size = section.page_end - section.page_start + 1
            prev_size = prev.page_end - prev.page_start + 1

            # Merge single-page boilerplate sections into neighbours
            if section_size == 1 and section.content_type == "boilerplate":
                merged[-1] = SectionStrategy(
                    page_start=prev.page_start,
                    page_end=section.page_end,
                    content_type=prev.content_type,
                    chunking_strategy=prev.chunking_strategy,
                    rationale=prev.rationale + " [merged boilerplate page]",
                )
            # Merge single-page sections of same type
            elif section.content_type == prev.content_type:
                merged[-1] = SectionStrategy(
                    page_start=prev.page_start,
                    page_end=section.page_end,
                    content_type=prev.content_type,
                    chunking_strategy=prev.chunking_strategy,
                    rationale=prev.rationale,
                )
            else:
                merged.append(section)

        return merged

    # ── Ingestion pipeline orchestration ─────────────────────────────

    def process_document(
        self,
        doc_id: str,
        canonical_pages: list,
        classification=None,
        preprocess_result: Optional[PreprocessorResult] = None,
        progress_cb=None,
    ) -> List:
        """Execute the full chunking pipeline for a document.

        This is the main entry point that owns the entire chunking lifecycle:
        1. Determine document-level strategy (or use pre-computed result)
        2. Classify page sections for multi-chunking
        3. Dispatch chunking per section (or single strategy for simple docs)
        4. Return the list of ChunkRecord objects

        The caller (ingest_pipeline) handles PDF ingestion, triage, DB storage,
        and outcome recording.  This method handles canonicalized-pages → chunks.

        Parameters
        ----------
        doc_id : str
            Document UUID.
        canonical_pages : List[CanonicalPage]
            Canonicalized page objects with text, spans, and lineage.
        classification : ClassificationResult, optional
            Document classification for strategy lookup.
        preprocess_result : PreprocessorResult, optional
            Pre-computed strategy result from an earlier determine_strategy call.
            If provided, skips strategy determination (avoids double computation).
        progress_cb : callable, optional
            Progress callback(stage, current, total).

        Returns
        -------
        List[ChunkRecord]
            Chunked and embedded document content.
        """
        if not canonical_pages:
            return []

        if progress_cb:
            progress_cb("embed", 0, len(canonical_pages))

        # Step 1: Use pre-computed result or determine strategy
        if preprocess_result is not None:
            result = preprocess_result
        else:
            page_count = len(canonical_pages)
            triage_summary = self._build_triage_from_pages(canonical_pages)

            inp = PreprocessorInput(
                doc_id=doc_id,
                filename="",
                page_count=page_count,
                document_type=classification.document_type if classification else None,
                classification_label=classification.classification_label if classification else None,
                classification_confidence=classification.confidence if classification else 0.0,
                triage_summary=triage_summary,
            )
            result = self.determine_strategy(inp)

        strategy = result.chunking_strategy
        processing_level = strategy.processing_level

        # Step 2: Handle non-chunking processing levels
        if processing_level == "skip":
            return []

        if processing_level == "metadata_only":
            return []  # Caller handles metadata extraction

        if processing_level == "single_chunk":
            return self._build_single_chunk(doc_id, canonical_pages)

        if processing_level == "page_level":
            return self._build_page_level_chunks(doc_id, canonical_pages)

        # Step 3: late_chunking — check for multi-chunking
        section_strategies = self.classify_page_sections(
            canonical_pages, strategy
        )

        if section_strategies and len(section_strategies) > 1:
            # Multi-chunking: dispatch per section
            return self._dispatch_multi_chunk(
                doc_id=doc_id,
                canonical_pages=canonical_pages,
                section_strategies=section_strategies,
                progress_cb=progress_cb,
            )

        # Single strategy for entire document
        return self._dispatch_single_strategy(
            doc_id=doc_id,
            canonical_pages=canonical_pages,
            strategy=strategy,
            progress_cb=progress_cb,
        )

    def _dispatch_single_strategy(
        self,
        doc_id: str,
        canonical_pages: list,
        strategy: ChunkingStrategy,
        progress_cb=None,
    ) -> list:
        """Dispatch chunking using a single strategy for the entire document."""
        from embedding.chunking_strategies import get_strategy_dispatch
        from embedding.late_chunking import late_chunk_embeddings

        dispatch = get_strategy_dispatch()
        strategy_fn = dispatch.get(strategy.strategy_name)

        if strategy_fn is not None:
            kwargs: dict = {}
            if strategy.strategy_name in ("proposition", "summary_indexed"):
                kwargs["gateway"] = self._gateway
            return strategy_fn(doc_id, canonical_pages, **kwargs)

        # Fall through to standard late chunking
        return late_chunk_embeddings(
            canonical_pages,
            macro_max_tokens=strategy.macro_max_tokens,
            macro_overlap_tokens=strategy.macro_overlap_tokens,
            child_target_tokens=strategy.child_target_tokens,
            progress_cb=progress_cb,
        )

    def _dispatch_multi_chunk(
        self,
        doc_id: str,
        canonical_pages: list,
        section_strategies: List[SectionStrategy],
        progress_cb=None,
    ) -> list:
        """Dispatch chunking per section for multi-chunking.

        Each section gets its own strategy applied to its page range.
        Chunks from all sections are combined into a single list with
        unique macro_ids across the whole document.
        """
        from embedding.chunking_strategies import get_strategy_dispatch
        from embedding.late_chunking import late_chunk_embeddings
        from core.contracts import ChunkRecord

        dispatch = get_strategy_dispatch()
        all_chunks: list = []
        macro_offset = 0

        # Build page number → canonical page index mapping
        page_map = {p.page_number: i for i, p in enumerate(canonical_pages)}

        for section in section_strategies:
            # Extract pages for this section
            section_pages = [
                canonical_pages[page_map[pn]]
                for pn in range(section.page_start, section.page_end + 1)
                if pn in page_map
            ]

            if not section_pages:
                continue

            strategy = section.chunking_strategy
            strategy_fn = dispatch.get(strategy.strategy_name)

            if strategy_fn is not None:
                kwargs: dict = {}
                if strategy.strategy_name in ("proposition", "summary_indexed"):
                    kwargs["gateway"] = self._gateway
                section_chunks = strategy_fn(doc_id, section_pages, **kwargs)
            else:
                section_chunks = late_chunk_embeddings(
                    section_pages,
                    macro_max_tokens=strategy.macro_max_tokens,
                    macro_overlap_tokens=strategy.macro_overlap_tokens,
                    child_target_tokens=strategy.child_target_tokens,
                    progress_cb=progress_cb,
                )

            # Offset macro_ids to avoid collisions across sections
            for chunk in section_chunks:
                all_chunks.append(
                    ChunkRecord(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        page_numbers=chunk.page_numbers,
                        macro_id=chunk.macro_id + macro_offset,
                        child_id=chunk.child_id,
                        chunk_type=chunk.chunk_type,
                        text_content=chunk.text_content,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        polygons=chunk.polygons,
                        source_type=chunk.source_type,
                        embedding_model=chunk.embedding_model,
                        embedding_dim=chunk.embedding_dim,
                        embedding=chunk.embedding,
                        heading_path=chunk.heading_path,
                        section_id=chunk.section_id,
                    )
                )

            if section_chunks:
                max_macro = max(c.macro_id for c in section_chunks)
                macro_offset += max_macro + 1

            logger.info(
                "Multi-chunk section pages %d-%d (%s): %d chunks via %s",
                section.page_start, section.page_end,
                section.content_type, len(section_chunks),
                strategy.strategy_name,
            )

        return all_chunks

    def _build_single_chunk(self, doc_id: str, canonical_pages: list) -> list:
        """Build a single chunk from the entire document text."""
        from embedding.model_registry import get_embedding_model
        from core.contracts import ChunkRecord
        import uuid as _uuid

        all_text = "\n\n".join(p.text for p in canonical_pages if p.text)
        if not all_text.strip():
            return []

        embedder = get_embedding_model()
        embedding = embedder.embed_text(all_text[:8192 * 4])

        all_pages = sorted({p.page_number for p in canonical_pages})
        all_polygons: list = []
        source_type = "native"
        heading_path = ""
        section_id = ""
        for page in canonical_pages:
            for span in page.spans:
                all_polygons.extend(span.polygons)
                if span.source_type == "di":
                    source_type = "di"
                if span.heading_path and not heading_path:
                    heading_path = span.heading_path
                if span.section_id and not section_id:
                    section_id = span.section_id

        chunk = ChunkRecord(
            chunk_id=str(_uuid.uuid5(_uuid.NAMESPACE_URL, f"{doc_id}:single:0")),
            doc_id=doc_id,
            page_numbers=all_pages,
            macro_id=0,
            child_id=0,
            chunk_type="narrative",
            text_content=all_text,
            char_start=0,
            char_end=len(all_text),
            polygons=all_polygons[:100],
            source_type=source_type,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
            embedding=embedding,
            heading_path=heading_path,
            section_id=section_id,
        )
        return [chunk]

    def _build_page_level_chunks(self, doc_id: str, canonical_pages: list) -> list:
        """Build one chunk per page with independent embeddings."""
        from embedding.model_registry import get_embedding_model
        from core.contracts import ChunkRecord
        import uuid as _uuid

        embedder = get_embedding_model()
        chunks: list = []

        for page_idx, page in enumerate(canonical_pages):
            if not page.text or not page.text.strip():
                continue

            embedding = embedder.embed_text(page.text)

            page_polygons: list = []
            source_type = "native"
            heading_path = ""
            section_id = ""
            for span in page.spans:
                page_polygons.extend(span.polygons)
                if span.source_type == "di":
                    source_type = "di"
                if span.heading_path and not heading_path:
                    heading_path = span.heading_path
                if span.section_id and not section_id:
                    section_id = span.section_id

            chunk = ChunkRecord(
                chunk_id=str(_uuid.uuid5(_uuid.NAMESPACE_URL, f"{doc_id}:page:{page.page_number}")),
                doc_id=doc_id,
                page_numbers=[page.page_number],
                macro_id=page_idx,
                child_id=0,
                chunk_type="narrative",
                text_content=page.text,
                char_start=0,
                char_end=len(page.text),
                polygons=page_polygons[:50],
                source_type=source_type,
                embedding_model=settings.embedding_model,
                embedding_dim=settings.embedding_dim,
                embedding=embedding,
                heading_path=heading_path,
                section_id=section_id,
            )
            chunks.append(chunk)

        return chunks

    def _build_triage_from_pages(self, canonical_pages: list) -> dict:
        """Build a triage summary from canonical pages (when page records not available)."""
        if not canonical_pages:
            return {}

        total_text = sum(len(p.text) for p in canonical_pages if p.text)
        page_count = len(canonical_pages)

        table_pages = 0
        for page in canonical_pages:
            if hasattr(page, "spans") and page.spans:
                table_spans = sum(1 for s in page.spans if s.is_table)
                if table_spans > len(page.spans) / 2:
                    table_pages += 1

        return {
            "total_text_length": total_text,
            "avg_text_length": total_text / page_count if page_count else 0,
            "page_count": page_count,
            "di_page_ratio": 0.0,
            "avg_image_coverage": 0.0,
        }
