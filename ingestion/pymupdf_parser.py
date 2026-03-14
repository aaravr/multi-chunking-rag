"""PyMUPDF parser backend — default PDF parser (§10.4).

Wraps the existing PyMUPDF (fitz) extraction logic into the
parser abstraction layer.  This is the default parser for PDFs.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

from agents.contracts import ParsedDocument, ParsedPage, new_id
from ingestion.parser_base import BaseParser, register_parser

try:
    import fitz  # PyMUPDF
    _FITZ_AVAILABLE = True
except ImportError:
    fitz = None  # type: ignore[assignment]
    _FITZ_AVAILABLE = False
    logger.debug("PyMUPDF (fitz) not installed — pymupdf parser unavailable")


class PyMuPDFParser(BaseParser):
    """PyMUPDF-based PDF parser — the platform default."""

    parser_name = "pymupdf"
    supported_formats = ["pdf"]

    def parse(self, file_path: str, doc_id: str = "") -> ParsedDocument:
        """Parse a PDF file using PyMUPDF."""
        if not _FITZ_AVAILABLE:
            raise RuntimeError("PyMUPDF (fitz) not installed. Install with: pip install pymupdf")
        with open(file_path, "rb") as f:
            data = f.read()
        filename = os.path.basename(file_path)
        return self.parse_bytes(data, filename, doc_id)

    def parse_bytes(self, data: bytes, filename: str, doc_id: str = "") -> ParsedDocument:
        """Parse PDF bytes using PyMUPDF."""
        if not _FITZ_AVAILABLE:
            raise RuntimeError("PyMUPDF (fitz) not installed. Install with: pip install pymupdf")
        if not doc_id:
            sha = hashlib.sha256(data).hexdigest()
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, sha))

        pdf = fitz.open(stream=data, filetype="pdf")
        pages: List[ParsedPage] = []

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text("text") or ""

            # Extract tables
            tables = []
            try:
                for table in page.find_tables():
                    cells = table.extract()
                    if cells:
                        # Build markdown
                        md_lines = []
                        for i, row in enumerate(cells):
                            md_lines.append("| " + " | ".join(str(c) if c else "" for c in row) + " |")
                            if i == 0:
                                md_lines.append("| " + " | ".join("---" for _ in row) + " |")
                        tables.append({
                            "markdown": "\n".join(md_lines),
                            "bbox": list(table.bbox) if hasattr(table, "bbox") else [],
                            "cells": cells,
                        })
            except Exception:
                pass  # find_tables() may not be available in all fitz versions

            # Extract headings (heuristic-based)
            headings = []
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", [])
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                    line_text = line_text.strip()
                    if not line_text:
                        continue
                    # Check if this looks like a heading
                    avg_size = sum(s.get("size", 12) for s in line.get("spans", [])) / max(len(line.get("spans", [])), 1)
                    if avg_size > 14 or (line_text.isupper() and len(line_text) <= 80):
                        level = 1 if avg_size > 16 or line_text.isupper() else 2
                        headings.append({
                            "text": line_text,
                            "level": level,
                            "bbox": list(line.get("bbox", [])),
                        })

            pages.append(ParsedPage(
                page_number=page_num + 1,
                text_content=text,
                tables=tables,
                headings=headings,
            ))

        pdf.close()

        return ParsedDocument(
            doc_id=doc_id,
            filename=filename,
            page_count=len(pages),
            pages=pages,
            parser_name=self.parser_name,
            file_format="pdf",
        )


# Auto-register on import only if fitz is available
if _FITZ_AVAILABLE:
    _pymupdf_parser = PyMuPDFParser()
    register_parser(_pymupdf_parser)
