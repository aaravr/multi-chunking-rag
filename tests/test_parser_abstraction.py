"""Tests for the Parser Abstraction Layer (§10.4)."""

import os
import tempfile
import pytest

from agents.contracts import ParsedDocument, ParsedPage
from ingestion.parser_base import (
    BaseParser,
    register_parser,
    get_parser,
    get_parser_for_format,
    list_parsers,
    _PARSER_REGISTRY,
)


# ── Parser Base Tests ────────────────────────────────────────────────


class DummyParser(BaseParser):
    parser_name = "dummy"
    supported_formats = ["dummy", "test"]

    def parse(self, file_path, doc_id=""):
        return ParsedDocument(
            doc_id=doc_id or "dummy-id",
            filename=os.path.basename(file_path),
            page_count=1,
            pages=[ParsedPage(page_number=1, text_content="dummy content")],
            parser_name=self.parser_name,
            file_format="dummy",
        )

    def parse_bytes(self, data, filename, doc_id=""):
        return ParsedDocument(
            doc_id=doc_id or "dummy-id",
            filename=filename,
            page_count=1,
            pages=[ParsedPage(page_number=1, text_content=data.decode("utf-8"))],
            parser_name=self.parser_name,
            file_format="dummy",
        )


def test_register_and_get_parser():
    dummy = DummyParser()
    register_parser(dummy)
    assert get_parser("dummy") is dummy


def test_get_parser_for_format():
    dummy = DummyParser()
    register_parser(dummy)
    p = get_parser_for_format("dummy")
    assert p is not None
    assert p.parser_name == "dummy"


def test_get_parser_for_format_with_dot():
    dummy = DummyParser()
    register_parser(dummy)
    p = get_parser_for_format(".dummy")
    assert p is not None


def test_supports_format():
    dummy = DummyParser()
    assert dummy.supports_format("dummy") is True
    assert dummy.supports_format("pdf") is False


def test_list_parsers():
    register_parser(DummyParser())
    names = list_parsers()
    assert "dummy" in names


# ── PyMUPDF Parser Tests ─────────────────────────────────────────────


def test_pymupdf_parser_registered():
    """PyMUPDF parser should auto-register on import if fitz available."""
    try:
        import ingestion.pymupdf_parser  # noqa: F401
        if not ingestion.pymupdf_parser._FITZ_AVAILABLE:
            pytest.skip("PyMUPDF (fitz) not installed")
        p = get_parser("pymupdf")
        assert p is not None
        assert p.supports_format("pdf")
    except ImportError:
        pytest.skip("PyMUPDF (fitz) not installed")


def test_pymupdf_parse_bytes():
    """Test PyMUPDF can parse a minimal PDF from bytes."""
    try:
        import fitz
        import ingestion.pymupdf_parser  # noqa: F401
    except ImportError:
        pytest.skip("PyMUPDF (fitz) not installed")

    p = get_parser("pymupdf")
    if p is None:
        pytest.skip("PyMUPDF parser not available")

    # Create a minimal valid PDF
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello World")
    pdf_bytes = doc.tobytes()
    doc.close()

    result = p.parse_bytes(pdf_bytes, "test.pdf")
    assert isinstance(result, ParsedDocument)
    assert result.parser_name == "pymupdf"
    assert result.file_format == "pdf"
    assert result.page_count == 1
    assert "Hello World" in result.pages[0].text_content


# ── CSV Parser Tests ─────────────────────────────────────────────────


def test_csv_parser_registered():
    import ingestion.multi_format_parser  # noqa: F401
    p = get_parser("csv")
    assert p is not None
    assert p.supports_format("csv")


def test_csv_parse_bytes():
    import ingestion.multi_format_parser  # noqa: F401
    p = get_parser("csv")
    data = b"name,amount,date\nACME,1000,2024-01-01\nFoo Inc,2000,2024-06-15\n"
    result = p.parse_bytes(data, "test.csv")
    assert isinstance(result, ParsedDocument)
    assert result.parser_name == "csv"
    assert result.page_count == 1
    assert "ACME" in result.pages[0].text_content
    assert len(result.pages[0].tables) == 1


# ── JSON Parser Tests ────────────────────────────────────────────────


def test_json_parser_registered():
    import ingestion.multi_format_parser  # noqa: F401
    p = get_parser("json")
    assert p is not None


def test_json_parse_bytes():
    import ingestion.multi_format_parser  # noqa: F401
    p = get_parser("json")
    data = b'[{"name": "ACME", "revenue": 1000}, {"name": "Foo", "revenue": 2000}]'
    result = p.parse_bytes(data, "test.json")
    assert isinstance(result, ParsedDocument)
    assert result.parser_name == "json"
    assert "ACME" in result.pages[0].text_content
    assert len(result.pages[0].tables) == 1


# ── Contract Compliance Tests ────────────────────────────────────────


def test_parsed_document_contract():
    """Verify ParsedDocument is frozen and has required fields."""
    doc = ParsedDocument(
        doc_id="doc-001",
        filename="test.pdf",
        page_count=1,
        pages=[ParsedPage(page_number=1, text_content="text")],
        parser_name="test",
        file_format="pdf",
    )
    assert doc.doc_id == "doc-001"
    with pytest.raises(AttributeError):
        doc.doc_id = "changed"  # type: ignore[misc]


def test_parsed_page_contract():
    page = ParsedPage(
        page_number=1,
        text_content="Some text",
        tables=[{"markdown": "| a |", "bbox": [], "cells": [["a"]]}],
        headings=[{"text": "Title", "level": 1, "bbox": []}],
    )
    assert page.page_number == 1
    assert len(page.tables) == 1
    assert len(page.headings) == 1
