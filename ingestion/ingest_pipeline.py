import hashlib
import json
import os
import uuid
from typing import List, Optional, Set

import fitz

from azure.core.exceptions import HttpResponseError

from core.config import settings
from core.contracts import DocumentRecord, PageRecord, TriageDecision
from embedding.late_chunking import late_chunk_embeddings
from ingestion.canonicalize import canonicalize_document
from core.logging import configure_logging
from ingestion.di_client import DIClient
from ingestion.pdf_analysis import analyze_page
from storage.db import get_connection
from storage import repo
from storage.schema_contract import check_schema_contract


def ingest_pdf(
    pdf_path: str,
    filename: Optional[str] = None,
    force_di_pages: Optional[List[int]] = None,
    progress_cb=None,
) -> str:
    configure_logging()
    check_schema_contract()
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    sha256 = _compute_sha256(pdf_path)
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, sha256))

    pdf = fitz.open(pdf_path)
    page_count = pdf.page_count

    di_client: Optional[DIClient] = None
    force_set: Set[int] = set(force_di_pages or [])
    page_buffer: List[PageRecord] = []

    try:
        with get_connection() as conn:
            existing = repo.fetch_document_by_sha(conn, sha256)
            if existing:
                doc_id = existing.doc_id
            else:
                doc_record = DocumentRecord(
                    doc_id=doc_id,
                    filename=filename or os.path.basename(pdf_path),
                    sha256=sha256,
                    page_count=page_count,
                )
                repo.insert_document(conn, doc_record)
                conn.commit()

            output_dir = os.path.join(settings.data_dir, doc_id)
            os.makedirs(output_dir, exist_ok=True)

            for page_index in range(page_count):
                if progress_cb:
                    progress_cb(
                        "triage",
                        page_index + 1,
                        page_count,
                    )
                page = pdf.load_page(page_index)
                triage = analyze_page(page)
                triage = _apply_force_di(triage, page_index + 1, force_set)
                di_json_path = None
                if triage.decision == "di_required":
                    if progress_cb:
                        progress_cb(
                            "di",
                            page_index + 1,
                            page_count,
                        )
                    if settings.disable_di:
                        triage = _apply_disable_di(triage)
                        if progress_cb:
                            progress_cb(
                                "di_skipped",
                                page_index + 1,
                                page_count,
                            )
                    else:
                        di_json_path = os.path.join(
                            output_dir, f"page_{page_index + 1:04d}_di.json"
                        )
                        if not os.path.exists(di_json_path):
                            if di_client is None:
                                di_client = DIClient()
                            page_bytes = _extract_single_page_pdf(pdf, page_index)
                            try:
                                di_result = di_client.analyze_page_bytes(page_bytes)
                            except HttpResponseError as exc:
                                if _is_invalid_content_length(exc):
                                    di_result = _analyze_with_image_fallback(
                                        di_client=di_client, page=page
                                    )
                                else:
                                    raise
                            _write_json(di_json_path, di_result.result)

                page_record = _build_page_record(
                    doc_id=doc_id,
                    page_number=page_index + 1,
                    triage=triage,
                    di_json_path=di_json_path,
                )
                page_buffer.append(page_record)
                if len(page_buffer) >= 50:
                    repo.insert_pages(conn, page_buffer)
                    conn.commit()
                    page_buffer.clear()
                    if progress_cb:
                        progress_cb(
                            "pages_committed",
                            page_index + 1,
                            page_count,
                        )

            if page_buffer:
                repo.insert_pages(conn, page_buffer)
                conn.commit()
                if progress_cb:
                    progress_cb(
                        "pages_committed",
                        page_count,
                        page_count,
                    )
    finally:
        pdf.close()

    return doc_id


def ingest_and_chunk(
    pdf_path: str,
    filename: Optional[str] = None,
    force_di_pages: Optional[List[int]] = None,
    macro_max_tokens: int = 8192,
    macro_overlap_tokens: int = 256,
    child_target_tokens: int = 256,
    progress_cb=None,
    force_reprocess: bool = False,
) -> str:
    doc_id = ingest_pdf(
        pdf_path,
        filename=filename,
        force_di_pages=force_di_pages,
        progress_cb=progress_cb,
    )
    with get_connection() as conn:
        if not force_reprocess and repo.count_chunks(conn, doc_id) > 0:
            return doc_id
    _cache_source_pdf(doc_id, pdf_path)
    with get_connection() as conn:
        pages = repo.fetch_pages(conn, doc_id)
    canonical_pages = canonicalize_document(
        doc_id=doc_id,
        pdf_path=pdf_path,
        pages=pages,
        progress_cb=progress_cb,
    )
    if progress_cb:
        progress_cb("embed", 0, len(canonical_pages))
    chunks = late_chunk_embeddings(
        canonical_pages,
        macro_max_tokens=macro_max_tokens,
        macro_overlap_tokens=macro_overlap_tokens,
        child_target_tokens=child_target_tokens,
        progress_cb=progress_cb,
    )
    if chunks:
        with get_connection() as conn:
            repo.insert_chunks(conn, chunks)
            conn.commit()
    return doc_id


def _cache_source_pdf(doc_id: str, pdf_path: str) -> None:
    output_dir = os.path.join(settings.data_dir, doc_id)
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, "source.pdf")
    if os.path.exists(target_path):
        return
    with open(pdf_path, "rb") as src, open(target_path, "wb") as dst:
        dst.write(src.read())


def _build_page_record(
    doc_id: str, page_number: int, triage: TriageDecision, di_json_path: Optional[str]
) -> PageRecord:
    return PageRecord(
        doc_id=doc_id,
        page_number=page_number,
        triage_metrics=triage.metrics,
        triage_decision=triage.decision,
        reason_codes=triage.reason_codes,
        di_json_path=di_json_path,
    )


def _apply_force_di(
    triage: TriageDecision, page_number: int, force_set: Set[int]
) -> TriageDecision:
    if page_number not in force_set:
        return triage
    reason_codes = list(triage.reason_codes)
    if "force_di" not in reason_codes:
        reason_codes.append("force_di")
    return TriageDecision(
        metrics=triage.metrics,
        decision="di_required",
        reason_codes=reason_codes,
    )


def _apply_disable_di(triage: TriageDecision) -> TriageDecision:
    reason_codes = list(triage.reason_codes)
    if "di_disabled" not in reason_codes:
        reason_codes.append("di_disabled")
    return TriageDecision(
        metrics=triage.metrics,
        decision=triage.decision,
        reason_codes=reason_codes,
    )


def _compute_sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _extract_single_page_pdf(pdf: fitz.Document, page_index: int) -> bytes:
    new_pdf = fitz.open()
    new_pdf.insert_pdf(pdf, from_page=page_index, to_page=page_index)
    page_bytes = new_pdf.tobytes()
    new_pdf.close()
    return page_bytes


def _is_invalid_content_length(exc: HttpResponseError) -> bool:
    message = str(exc).lower()
    return "invalidcontentlength" in message or "input image is too large" in message


def _analyze_with_image_fallback(di_client: DIClient, page: fitz.Page):
    for zoom in (1.0, 0.7, 0.5, 0.3):
        image_bytes = _render_page_png(page, zoom=zoom)
        try:
            return di_client.analyze_page_image_bytes(image_bytes)
        except HttpResponseError as exc:
            if _is_invalid_content_length(exc):
                continue
            raise
    raise RuntimeError("Azure DI rejected all fallback image sizes.")


def _render_page_png(page: fitz.Page, zoom: float) -> bytes:
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
    return pix.tobytes("png")


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
