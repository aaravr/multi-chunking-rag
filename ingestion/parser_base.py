"""Parser Abstraction Layer — pluggable document parsers (§10.4).

Provides a common interface for document parsing backends:
- PyMUPDF (default, ships with platform)
- Docling (optional, enterprise)
- Azure Document Intelligence (for selective OCR)

New parsers extend BaseParser and register via register_parser().
The active parser is selected via PARSER_BACKEND config.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agents.contracts import ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """Abstract base class for all document parsers.

    Teams can implement custom parsers by extending this class.
    The parser MUST return ParsedDocument/ParsedPage contracts.
    """

    parser_name: str = "base"
    supported_formats: List[str] = []

    @abstractmethod
    def parse(self, file_path: str, doc_id: str = "") -> ParsedDocument:
        """Parse a document file and return structured output.

        Args:
            file_path: Path to the document file
            doc_id: Optional document ID (generated if empty)

        Returns:
            ParsedDocument with all pages, tables, headings
        """

    @abstractmethod
    def parse_bytes(self, data: bytes, filename: str, doc_id: str = "") -> ParsedDocument:
        """Parse document from bytes.

        Args:
            data: Raw file bytes
            filename: Original filename (used for format detection)
            doc_id: Optional document ID

        Returns:
            ParsedDocument with all pages, tables, headings
        """

    def supports_format(self, file_format: str) -> bool:
        """Check if this parser supports a file format."""
        return file_format.lower().lstrip(".") in self.supported_formats


# ── Parser Registry ──────────────────────────────────────────────────

_PARSER_REGISTRY: Dict[str, BaseParser] = {}


def register_parser(parser: BaseParser) -> None:
    """Register a parser backend."""
    _PARSER_REGISTRY[parser.parser_name] = parser
    logger.info("Registered parser: %s (formats: %s)", parser.parser_name, parser.supported_formats)


def get_parser(name: str) -> Optional[BaseParser]:
    """Get a registered parser by name."""
    return _PARSER_REGISTRY.get(name)


def get_parser_for_format(file_format: str) -> Optional[BaseParser]:
    """Find a parser that supports the given format."""
    fmt = file_format.lower().lstrip(".")
    for parser in _PARSER_REGISTRY.values():
        if parser.supports_format(fmt):
            return parser
    return None


def list_parsers() -> List[str]:
    """List all registered parser names."""
    return list(_PARSER_REGISTRY.keys())
