"""Docling parser backend — enterprise document parser (§10.4).

Wraps the Docling library for advanced document understanding.
Falls back gracefully when Docling is not installed.

Install: pip install docling

Docling provides:
- Superior table extraction
- Document structure analysis
- Multi-format support (PDF, DOCX, PPTX, HTML)
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from typing import Any, Dict, List

from agents.contracts import ParsedDocument, ParsedPage, new_id
from ingestion.parser_base import BaseParser, register_parser

logger = logging.getLogger(__name__)

_DOCLING_AVAILABLE = False
try:
    from docling.document_converter import DocumentConverter
    _DOCLING_AVAILABLE = True
except ImportError:
    logger.debug("Docling not installed — docling parser unavailable")


class DoclingParser(BaseParser):
    """Docling-based document parser for advanced document understanding.

    Requires: pip install docling

    Provides superior table extraction and document structure analysis
    compared to PyMUPDF for complex documents.
    """

    parser_name = "docling"
    supported_formats = ["pdf", "docx", "pptx", "html"]

    def __init__(self) -> None:
        if not _DOCLING_AVAILABLE:
            logger.warning("Docling parser created but docling is not installed")

    def parse(self, file_path: str, doc_id: str = "") -> ParsedDocument:
        """Parse a document file using Docling."""
        if not _DOCLING_AVAILABLE:
            raise RuntimeError(
                "Docling is not installed. Install with: pip install docling"
            )

        if not doc_id:
            with open(file_path, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, sha))

        filename = os.path.basename(file_path)
        file_format = os.path.splitext(filename)[1].lstrip(".").lower()

        converter = DocumentConverter()
        result = converter.convert(file_path)
        doc = result.document

        pages: List[ParsedPage] = []
        # Docling provides a flat document model; we reconstruct pages
        page_texts: Dict[int, List[str]] = {}
        page_tables: Dict[int, List[Dict[str, Any]]] = {}
        page_headings: Dict[int, List[Dict[str, Any]]] = {}

        for element in doc.iterate_items():
            item = element
            # Get page number from provenance
            page_num = 1
            if hasattr(item, "prov") and item.prov:
                prov = item.prov[0] if isinstance(item.prov, list) else item.prov
                page_num = getattr(prov, "page_no", 1)

            page_texts.setdefault(page_num, [])
            page_tables.setdefault(page_num, [])
            page_headings.setdefault(page_num, [])

            item_type = type(item).__name__
            text = getattr(item, "text", "")

            if "Table" in item_type:
                md = getattr(item, "export_to_markdown", lambda: text)()
                page_tables[page_num].append({
                    "markdown": md,
                    "bbox": [],
                    "cells": [],
                })
            elif "Heading" in item_type or "Title" in item_type:
                level = getattr(item, "level", 1)
                page_headings[page_num].append({
                    "text": text,
                    "level": level,
                    "bbox": [],
                })
                page_texts[page_num].append(text)
            else:
                if text:
                    page_texts[page_num].append(text)

        # Build pages
        max_page = max(page_texts.keys()) if page_texts else 0
        for pn in range(1, max_page + 1):
            pages.append(ParsedPage(
                page_number=pn,
                text_content="\n".join(page_texts.get(pn, [])),
                tables=page_tables.get(pn, []),
                headings=page_headings.get(pn, []),
            ))

        return ParsedDocument(
            doc_id=doc_id,
            filename=filename,
            page_count=len(pages),
            pages=pages,
            parser_name=self.parser_name,
            file_format=file_format,
        )

    def parse_bytes(self, data: bytes, filename: str, doc_id: str = "") -> ParsedDocument:
        """Parse document bytes via temp file (Docling requires file path)."""
        import tempfile
        ext = os.path.splitext(filename)[1] or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            return self.parse(tmp_path, doc_id)
        finally:
            os.unlink(tmp_path)


# Auto-register only if docling is available
if _DOCLING_AVAILABLE:
    _docling_parser = DoclingParser()
    register_parser(_docling_parser)
