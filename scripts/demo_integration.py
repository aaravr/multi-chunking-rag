import os
import tempfile
from datetime import datetime

import fitz

from ingestion.ingest_pipeline import ingest_and_chunk
from retrieval.router import search_with_intent
from storage.db import get_connection
from storage import repo


def build_demo_pdf(path: str) -> None:
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "SECTION ONE")
    page1.insert_text((72, 100), "Overview of the report and purpose.")

    page2 = doc.new_page()
    page2.insert_text((72, 72), "SIGNIFICANT EVENTS")
    page2.insert_text((72, 100), "Event A: Acquisition completed.")
    page2.insert_text((72, 120), "Event B: Divestiture announced.")
    page2.insert_text((72, 160), "CET1 Ratio 12.3%")

    page3 = doc.new_page()
    page3.insert_text((72, 72), "FINANCIAL HIGHLIGHTS")
    page3.insert_text((72, 100), "Net Income $123M")
    doc.save(path)
    doc.close()


def run_demo(output_path: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "demo.pdf")
        build_demo_pdf(pdf_path)
        doc_id = ingest_and_chunk(
            pdf_path,
            filename="demo.pdf",
            force_reprocess=True,
        )

        with get_connection() as conn:
            pages = repo.fetch_pages(conn, doc_id)
            chunk_count = repo.count_chunks(conn, doc_id)

        queries = [
            "page 2 CET1 ratio",
            "list all significant events",
            "What is net income?",
        ]
        results = []
        for query in queries:
            hits = search_with_intent(doc_id, query, top_k=3)
            results.append(
                {
                    "query": query,
                    "hits": [
                        {
                            "chunk_id": hit.chunk_id,
                            "pages": hit.page_numbers,
                            "heading_path": hit.heading_path,
                            "section_id": hit.section_id,
                            "char_span": f"{hit.char_start}-{hit.char_end}",
                            "polygons": len(hit.polygons),
                        }
                        for hit in hits
                    ],
                }
            )

        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("# Integration Demo Evidence\n\n")
            handle.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")
            handle.write("## Document\n")
            handle.write(f"- doc_id: {doc_id}\n")
            handle.write(f"- pages: {len(pages)}\n")
            handle.write(f"- chunks: {chunk_count}\n\n")
            handle.write("## Queries\n")
            for entry in results:
                handle.write(f"### {entry['query']}\n")
                if not entry["hits"]:
                    handle.write("- no hits\n\n")
                    continue
                for hit in entry["hits"]:
                    handle.write(
                        "- "
                        + f"chunk_id={hit['chunk_id']} "
                        + f"pages={hit['pages']} "
                        + f"heading_path={hit['heading_path']} "
                        + f"section_id={hit['section_id']} "
                        + f"char_span={hit['char_span']} "
                        + f"polygons={hit['polygons']}\n"
                    )
                handle.write("\n")


if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "docs", "demo_evidence.md"
    )
    run_demo(output_file)
