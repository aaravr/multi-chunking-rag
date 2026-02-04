import os
import tempfile

import fitz

from core.contracts import PageRecord, TriageMetrics
from ingestion.canonicalize import canonicalize_document


def test_heading_path_presence_in_canonicalization():
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "test_doc.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "SECTION ONE")
        page.insert_text((72, 120), "This is body text.")
        doc.save(pdf_path)
        doc.close()

        page_record = PageRecord(
            doc_id="doc-1",
            page_number=1,
            triage_metrics=TriageMetrics(
                text_length=10,
                text_density=0.1,
                image_coverage_ratio=0.0,
                layout_complexity_score=0.0,
            ),
            triage_decision="native_only",
            reason_codes=[],
            di_json_path=None,
        )

        canonical_pages = canonicalize_document(
            doc_id="doc-1", pdf_path=pdf_path, pages=[page_record]
        )
        assert canonical_pages
        spans = canonical_pages[0].spans
        assert spans
        heading_span = next((s for s in spans if s.text == "SECTION ONE"), None)
        assert heading_span is not None
        assert heading_span.heading_path.endswith("/SECTION ONE")
        assert heading_span.section_id == "SECTION ONE"
        for span in spans:
            assert span.heading_path
            assert span.section_id
