"""Versioned Prompt Registry (MASTER_PROMPT §4.4, §11).

Template selection is deterministic based on (intent_type, coverage_subtype, output_format).
Templates are versioned by content hash and logged with every synthesis call.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from synthesis import prompts as poc_prompts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptTemplate:
    """A versioned prompt template."""
    template_id: str
    intent_type: str
    system_prompt: str
    user_prompt: str
    version: str         # Content hash for reproducibility


def _hash_content(text: str) -> str:
    """Compute short hash of prompt content for versioning."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _make_template(template_id: str, intent_type: str, system: str, user: str) -> PromptTemplate:
    combined = system + user
    return PromptTemplate(
        template_id=template_id,
        intent_type=intent_type,
        system_prompt=system,
        user_prompt=user,
        version=_hash_content(combined),
    )


# ── Built-in templates (migrated from PoC synthesis/prompts.py) ──────

_REGISTRY: Dict[str, PromptTemplate] = {}


def _register(t: PromptTemplate) -> PromptTemplate:
    _REGISTRY[t.template_id] = t
    return t


# Standard QA
SEMANTIC_QA = _register(_make_template(
    "semantic_qa",
    "semantic",
    poc_prompts.SYSTEM,
    poc_prompts.USER,
))

# Coverage list extraction
COVERAGE_LIST = _register(_make_template(
    "coverage_list",
    "coverage",
    poc_prompts.COVERAGE_SYSTEM,
    poc_prompts.COVERAGE_USER,
))

# Coverage closed matters
COVERAGE_CLOSED = _register(_make_template(
    "coverage_closed",
    "coverage",
    poc_prompts.COVERAGE_CLOSED_SYSTEM,
    poc_prompts.COVERAGE_CLOSED_USER,
))

# Coverage attribute (numeric extraction)
COVERAGE_ATTRIBUTE = _register(_make_template(
    "coverage_attribute",
    "coverage",
    poc_prompts.COVERAGE_ATTRIBUTE_SYSTEM,
    poc_prompts.COVERAGE_ATTRIBUTE_USER,
))


# Document classification
CLASSIFICATION = _register(_make_template(
    "classification",
    "classification",
    (
        "You are a document classification expert. Your task is to determine the type "
        "and category of a document based on its filename and front-matter text.\n\n"
        "You MUST respond with a JSON object containing exactly these fields:\n"
        '- "document_type": A specific document type identifier\n'
        '- "classification_label": A broader category label\n'
        '- "confidence": A float between 0 and 1\n\n'
        "Respond with ONLY the JSON object, no other text."
    ),
    (
        "Classify this document:\n\n"
        "Filename: {filename}\n"
        "Page count: {page_count}\n\n"
        "Front-matter text (first pages):\n---\n{front_matter_text}\n---\n\n"
        "Return your classification as JSON."
    ),
))


# Lookup table for deterministic template selection (§4.4, OCP).
# Key: (intent_type, coverage_subtype, status_filter)
# Entries are checked in order; first match wins. None matches any value.
_TEMPLATE_LOOKUP: List[Tuple[Tuple[Optional[str], Optional[str], Optional[str]], PromptTemplate]] = [
    (("coverage", None, "closed"), COVERAGE_CLOSED),
    (("coverage", "attribute", None), COVERAGE_ATTRIBUTE),
    (("coverage", None, None), COVERAGE_LIST),
]


def get_template(
    intent_type: str,
    coverage_subtype: Optional[str] = None,
    status_filter: Optional[str] = None,
) -> PromptTemplate:
    """Select the correct prompt template based on query classification.

    Template selection is deterministic (§4.4).
    New templates can be added to _TEMPLATE_LOOKUP without modifying this function.
    """
    for (match_intent, match_subtype, match_status), template in _TEMPLATE_LOOKUP:
        if match_intent is not None and match_intent != intent_type:
            continue
        if match_subtype is not None and match_subtype != coverage_subtype:
            continue
        if match_status is not None and match_status != status_filter:
            continue
        return template

    # Default: semantic QA template (also used for location, comparison, etc.)
    return SEMANTIC_QA


def get_template_by_id(template_id: str) -> Optional[PromptTemplate]:
    """Look up a template by its ID."""
    return _REGISTRY.get(template_id)


def list_templates() -> Dict[str, Tuple[str, str]]:
    """Return {template_id: (intent_type, version)} for all registered templates."""
    return {
        tid: (t.intent_type, t.version)
        for tid, t in _REGISTRY.items()
    }


def register_template(template: PromptTemplate) -> None:
    """Register a custom prompt template."""
    _REGISTRY[template.template_id] = template
    logger.info("Registered prompt template: %s (v=%s)", template.template_id, template.version)
