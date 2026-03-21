"""Data Extractor Agent — schema-driven field extraction (§10.1).

Extracts structured fields from document chunks using declarative
ExtractionSchema definitions.  Supports deterministic (regex) extraction
with LLM fallback for complex fields.

All LLM calls go through the Model Gateway (§7.3).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from agents.contracts import (
    AgentMessage,
    ExtractionResult,
    ExtractionSchema,
    ExtractedField,
    FieldDefinition,
    new_id,
)
from agents.message_bus import MessageBus
from agents.model_gateway import ModelCallContext, ModelGateway
from core.enums import ExtractionMethod, SynthesisMode

logger = logging.getLogger(__name__)

# ── Schema Registry ──────────────────────────────────────────────────
# Maps (document_type, classification_label) → ExtractionSchema

_SCHEMA_REGISTRY: Dict[str, ExtractionSchema] = {}


def register_schema(schema: ExtractionSchema) -> None:
    """Register an extraction schema globally."""
    key = f"{schema.document_type}:{schema.classification_label}"
    _SCHEMA_REGISTRY[key] = schema
    logger.info("Registered extraction schema: %s (%s)", schema.schema_name, key)


def get_schema(document_type: str, classification_label: str) -> Optional[ExtractionSchema]:
    """Look up the schema for a document type."""
    key = f"{document_type}:{classification_label}"
    return _SCHEMA_REGISTRY.get(key)


def list_schemas() -> List[ExtractionSchema]:
    """Return all registered schemas."""
    return list(_SCHEMA_REGISTRY.values())


# ── Extraction Prompt ─────────────────────────────────────────────────

_EXTRACTION_SYSTEM = """You are a precise document field extractor.
Extract the requested fields from the provided evidence chunks.
Return ONLY a JSON object with field names as keys and extracted values as strings.
If a field cannot be found, return null for that field.
Do NOT hallucinate or infer values not present in the evidence."""

_EXTRACTION_USER = """Extract these fields from the document evidence:

Fields to extract:
{field_definitions}

Evidence chunks:
{evidence}

Return a JSON object with the field names as keys. Example:
{{"field_name": "extracted value", "other_field": "other value"}}"""


class ExtractorAgent(BaseAgent):
    """Schema-driven field extraction agent (§10.1).

    Flow:
    1. Receives extraction request with doc_id, chunks, and optional schema_id
    2. Looks up ExtractionSchema from registry (or uses provided fields)
    3. Attempts deterministic extraction (regex) first
    4. Falls back to LLM for fields that require semantic understanding
    5. Validates extracted values against field constraints
    6. Returns ExtractionResult with per-field confidence
    """

    agent_name = "extractor"

    def __init__(
        self,
        bus: MessageBus,
        gateway: Optional[ModelGateway] = None,
        model_id: str = "gpt-4o-mini",
    ) -> None:
        super().__init__(bus, gateway)
        self._model_id = model_id

    def handle_message(self, message: AgentMessage) -> ExtractionResult:
        """Handle an extraction request from the Orchestrator."""
        payload = message.payload
        doc_id = payload["doc_id"]
        chunks = payload.get("chunks", [])
        query_id = message.query_id
        schema_id = payload.get("schema_id", "")
        document_type = payload.get("document_type", "")
        classification_label = payload.get("classification_label", "")

        # Resolve schema
        schema = None
        if schema_id:
            schema = next((s for s in _SCHEMA_REGISTRY.values() if s.schema_id == schema_id), None)
        if schema is None and document_type:
            schema = get_schema(document_type, classification_label)

        if schema is None:
            return ExtractionResult(
                doc_id=doc_id,
                schema_id="",
                query_id=query_id,
                fields=[],
                warnings=["No extraction schema found for document type"],
            )

        return self.extract(doc_id, chunks, schema, query_id)

    def extract(
        self,
        doc_id: str,
        chunks: List[Any],
        schema: ExtractionSchema,
        query_id: str = "",
    ) -> ExtractionResult:
        """Run extraction pipeline: deterministic → LLM fallback → validate."""
        extracted: List[ExtractedField] = []
        llm_needed: List[FieldDefinition] = []

        # Phase 1: Deterministic extraction (regex-based)
        evidence_text = self._build_evidence_text(chunks)
        for field_def in schema.fields:
            result = self._try_deterministic_extraction(field_def, evidence_text, chunks)
            if result is not None:
                extracted.append(result)
            else:
                llm_needed.append(field_def)

        # Phase 2: LLM extraction for remaining fields
        input_tokens = 0
        output_tokens = 0
        if llm_needed and self.gateway:
            llm_fields, in_tok, out_tok = self._extract_via_llm(
                llm_needed, evidence_text, chunks, query_id
            )
            extracted.extend(llm_fields)
            input_tokens = in_tok
            output_tokens = out_tok

        # Phase 3: Validate all extracted fields
        validated = [self._validate_field(f, schema) for f in extracted]

        # Compute overall confidence
        confidences = [f.confidence for f in validated if f.confidence > 0]
        overall = sum(confidences) / len(confidences) if confidences else 0.0

        return ExtractionResult(
            doc_id=doc_id,
            schema_id=schema.schema_id,
            query_id=query_id,
            fields=validated,
            overall_confidence=round(overall, 4),
            extraction_model=self._model_id if llm_needed else SynthesisMode.DETERMINISTIC,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _try_deterministic_extraction(
        self,
        field_def: FieldDefinition,
        evidence_text: str,
        chunks: List[Any],
    ) -> Optional[ExtractedField]:
        """Try regex-based extraction if validation_regex is provided."""
        if not field_def.validation_regex:
            return None

        pattern = re.compile(field_def.validation_regex, re.IGNORECASE)
        match = pattern.search(evidence_text)
        if match:
            value = match.group(0)
            # Find which chunks contain this value
            source_chunks = []
            page_numbers = []
            for chunk in chunks:
                chunk_text = getattr(chunk, "text_content", str(chunk))
                if value in chunk_text:
                    cid = getattr(chunk, "chunk_id", "")
                    if cid:
                        source_chunks.append(cid)
                    pages = getattr(chunk, "page_numbers", [])
                    page_numbers.extend(pages)

            return ExtractedField(
                field_name=field_def.field_name,
                raw_value=value,
                confidence=0.9,
                source_chunk_ids=source_chunks,
                page_numbers=sorted(set(page_numbers)),
                extraction_method=ExtractionMethod.REGEX,
            )
        return None

    def _extract_via_llm(
        self,
        fields: List[FieldDefinition],
        evidence_text: str,
        chunks: List[Any],
        query_id: str,
    ) -> tuple:
        """Use LLM to extract fields that regex couldn't handle."""
        field_descriptions = "\n".join(
            f"- {f.field_name} ({f.field_type}): {f.description or f.display_name}"
            + (f" [Hint: {f.extraction_hint}]" if f.extraction_hint else "")
            for f in fields
        )

        messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM},
            {
                "role": "user",
                "content": _EXTRACTION_USER.format(
                    field_definitions=field_descriptions,
                    evidence=evidence_text[:12000],
                ),
            },
        ]

        result = self.gateway.call_model(
            model_id=self._model_id,
            messages=messages,
            temperature=0.0,
            ctx=ModelCallContext(query_id=query_id, agent_id=self.agent_name, step_id=new_id()),
        )

        parsed = _parse_llm_json(result.get("content", "{}"))
        extracted_fields = [
            self._build_llm_field(fdef, parsed.get(fdef.field_name), chunks)
            for fdef in fields
        ]
        return extracted_fields, result.get("input_tokens", 0), result.get("output_tokens", 0)

    def _build_llm_field(
        self, fdef: FieldDefinition, raw_value: Optional[Any], chunks: List[Any],
    ) -> ExtractedField:
        """Build an ExtractedField from a single LLM-parsed value."""
        if raw_value is None:
            return ExtractedField(
                field_name=fdef.field_name,
                raw_value="",
                confidence=0.0,
                extraction_method=ExtractionMethod.LLM,
                validation_passed=not fdef.required,
                validation_errors=["Field not found"] if fdef.required else [],
            )
        str_val = str(raw_value)
        source_chunks, page_numbers = _find_source_chunks(str_val, chunks)
        return ExtractedField(
            field_name=fdef.field_name,
            raw_value=str_val,
            confidence=0.7 if source_chunks else 0.4,
            source_chunk_ids=source_chunks,
            page_numbers=sorted(set(page_numbers)),
            extraction_method=ExtractionMethod.LLM,
        )

    def _validate_field(
        self, field: ExtractedField, schema: ExtractionSchema
    ) -> ExtractedField:
        """Validate an extracted field against its definition."""
        field_def = next(
            (f for f in schema.fields if f.field_name == field.field_name), None
        )
        if field_def is None:
            return field

        errors: List[str] = list(field.validation_errors)
        passed = field.validation_passed

        # Check required
        if field_def.required and not field.raw_value:
            errors.append(f"Required field '{field.field_name}' is empty")
            passed = False

        # Check allowed values
        if field_def.allowed_values and field.raw_value:
            if field.raw_value not in field_def.allowed_values:
                errors.append(
                    f"Value '{field.raw_value}' not in allowed values: {field_def.allowed_values}"
                )
                passed = False

        # Check regex validation
        if field_def.validation_regex and field.raw_value:
            if not re.match(field_def.validation_regex, field.raw_value):
                errors.append(f"Value fails regex validation: {field_def.validation_regex}")
                passed = False

        if errors != list(field.validation_errors) or passed != field.validation_passed:
            from dataclasses import replace
            return replace(field, validation_passed=passed, validation_errors=errors)
        return field

    def _build_evidence_text(self, chunks: List[Any]) -> str:
        """Build concatenated evidence text from chunks."""
        lines = []
        for i, chunk in enumerate(chunks):
            text = getattr(chunk, "text_content", str(chunk))
            cid = getattr(chunk, "chunk_id", f"chunk_{i}")
            pages = getattr(chunk, "page_numbers", [])
            page_str = ",".join(str(p) for p in pages)
            lines.append(f"[{cid}] (pages: {page_str})\n{text}")
        return "\n\n".join(lines)


# ── Module-level helpers ──────────────────────────────────────────────

def _parse_llm_json(content: str) -> dict:
    """Parse JSON from LLM response, stripping markdown code fences."""
    clean = content.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        clean = clean.rsplit("```", 1)[0]
    try:
        return json.loads(clean)
    except (json.JSONDecodeError, IndexError):
        return {}


def _find_source_chunks(value: str, chunks: List[Any]) -> tuple:
    """Find which chunks contain the given value. Returns (chunk_ids, page_numbers)."""
    source_chunks: List[str] = []
    page_numbers: List[int] = []
    lower_val = value.lower()
    for chunk in chunks:
        chunk_text = getattr(chunk, "text_content", str(chunk))
        if lower_val in chunk_text.lower():
            cid = getattr(chunk, "chunk_id", "")
            if cid:
                source_chunks.append(cid)
            page_numbers.extend(getattr(chunk, "page_numbers", []))
    return source_chunks, page_numbers
