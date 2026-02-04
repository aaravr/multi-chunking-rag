import json
import os
import tempfile

import fitz

from core.contracts import PageRecord, TriageMetrics
from ingestion.canonicalize import canonicalize_document


def test_di_table_atomic_markdown_chunk():
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "test_doc.pdf")
        di_path = os.path.join(tmpdir, "page_0001_di.json")
        doc = fitz.open()
        doc.new_page()
        doc.save(pdf_path)
        doc.close()
        payload = {
            "pages": [{"pageNumber": 1, "lines": []}],
            "tables": [
                {
                    "cells": [
                        {"rowIndex": 0, "columnIndex": 0, "content": "Header A"},
                        {"rowIndex": 0, "columnIndex": 1, "content": "Header B"},
                        {"rowIndex": 1, "columnIndex": 0, "content": "1"},
                        {"rowIndex": 1, "columnIndex": 1, "content": "2"},
                    ],
                    "boundingRegions": [
                        {
                            "pageNumber": 1,
                            "polygon": [0, 0, 10, 0, 10, 5, 0, 5]
                        }
                    ],
                }
            ],
        }
        with open(di_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

        page_record = PageRecord(
            doc_id="doc-1",
            page_number=1,
            triage_metrics=TriageMetrics(
                text_length=0,
                text_density=0.0,
                image_coverage_ratio=0.0,
                layout_complexity_score=0.0,
            ),
            triage_decision="di_required",
            reason_codes=[],
            di_json_path=di_path,
        )

        canonical_pages = canonicalize_document(
            doc_id="doc-1", pdf_path=pdf_path, pages=[page_record]
        )
        spans = canonical_pages[0].spans
        table_spans = [span for span in spans if span.is_table]
        assert len(table_spans) == 1
        assert table_spans[0].text.startswith("[TABLE]")
        assert table_spans[0].polygons
