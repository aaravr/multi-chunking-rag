import hashlib
import json
import os
import time
import uuid
from typing import List, Optional, Set

import fitz

from azure.core.exceptions import HttpResponseError

from core.config import settings
from core.contracts import ChunkRecord, DocumentRecord, PageRecord, TriageDecision
from agents.classifier_agent import ClassifierAgent, get_classification_memory
from agents.contracts import ChunkingOutcome, PreprocessorInput
from agents.message_bus import MessageBus
from agents.model_gateway import ModelGateway
from agents.preprocessor_agent import PreprocessorAgent
from embedding.late_chunking import late_chunk_embeddings
from ingestion.canonicalize import canonicalize_document
from core.logging import configure_logging
from ingestion.di_client import DIClient
from ingestion.document_facts import extract_document_facts
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

    # ── Classify the document (§4.8) ─────────────────────────────────
    # Classification runs BEFORE canonicalization so the preprocessor can
    # use document_type to select the optimal chunking strategy.
    classification = None
    resolved_filename = filename or os.path.basename(pdf_path)
    if settings.enable_classifier:
        # Build front-matter from raw page text for classification
        # (canonicalization hasn't run yet — use page triage data)
        classification = _classify_document_from_pages(
            doc_id=doc_id,
            filename=resolved_filename,
            pages=pages,
            pdf_path=pdf_path,
            progress_cb=progress_cb,
        )

    # ── Preprocessor: determine chunking strategy (§4.9) ─────────────
    preprocess_result = None
    if settings.enable_preprocessor:
        preprocess_result = _run_preprocessor(
            doc_id=doc_id,
            filename=resolved_filename,
            pages=pages,
            classification=classification,
            progress_cb=progress_cb,
        )
        if preprocess_result and not preprocess_result.requires_chunking:
            return doc_id
        if preprocess_result:
            strategy = preprocess_result.chunking_strategy
            macro_max_tokens = strategy.macro_max_tokens
            macro_overlap_tokens = strategy.macro_overlap_tokens
            child_target_tokens = strategy.child_target_tokens

    canonical_pages = canonicalize_document(
        doc_id=doc_id,
        pdf_path=pdf_path,
        pages=pages,
        progress_cb=progress_cb,
    )

    # If classifier hasn't run yet (preprocessor disabled), run it now
    # on canonical pages as before.
    if classification is None and settings.enable_classifier:
        classification = _classify_document(
            doc_id=doc_id,
            filename=resolved_filename,
            canonical_pages=canonical_pages,
            page_count=len(canonical_pages),
            progress_cb=progress_cb,
        )

    if progress_cb:
        progress_cb("embed", 0, len(canonical_pages))
    chunk_start_time = time.monotonic()
    chunks = late_chunk_embeddings(
        canonical_pages,
        macro_max_tokens=macro_max_tokens,
        macro_overlap_tokens=macro_overlap_tokens,
        child_target_tokens=child_target_tokens,
        progress_cb=progress_cb,
    )
    chunk_elapsed_ms = (time.monotonic() - chunk_start_time) * 1000

    # Stamp classification on chunks before insert
    if classification and chunks:
        chunks = [
            ChunkRecord(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                page_numbers=c.page_numbers,
                macro_id=c.macro_id,
                child_id=c.child_id,
                chunk_type=c.chunk_type,
                text_content=c.text_content,
                char_start=c.char_start,
                char_end=c.char_end,
                polygons=c.polygons,
                source_type=c.source_type,
                embedding_model=c.embedding_model,
                embedding_dim=c.embedding_dim,
                embedding=c.embedding,
                heading_path=c.heading_path,
                section_id=c.section_id,
                document_type=classification.document_type,
                classification_label=classification.classification_label,
            )
            for c in chunks
        ]

    if chunks:
        with get_connection() as conn:
            repo.insert_chunks(conn, chunks)
            if classification:
                repo.update_document_classification(
                    conn, doc_id,
                    classification.document_type,
                    classification.classification_label,
                )
            if settings.enable_document_facts:
                facts = extract_document_facts(doc_id, chunks)
                repo.upsert_document_facts(conn, facts)
            conn.commit()

    # ── Record chunking outcome for preprocessor learning (§4.9) ─────
    if settings.enable_preprocessor and preprocess_result and chunks:
        _record_chunking_outcome(
            doc_id=doc_id,
            preprocess_result=preprocess_result,
            classification=classification,
            chunks=chunks,
            page_count=len(pages),
            processing_time_ms=chunk_elapsed_ms,
        )

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


def _classify_document(
    doc_id: str,
    filename: str,
    canonical_pages: list,
    page_count: int,
    progress_cb=None,
):
    """Run the classifier agent on the document front matter.

    Extracts text from the first N pages (configured by front_matter_pages)
    and passes it to the ClassifierAgent for classification.
    """
    if progress_cb:
        progress_cb("classify", 0, 1)

    # Build front-matter text from first N canonical pages
    front_matter_limit = settings.front_matter_pages
    front_text_parts = []
    for page in canonical_pages[:front_matter_limit]:
        front_text_parts.append(page.text)
    front_matter_text = "\n\n".join(front_text_parts)

    # Create a lightweight classifier (no LLM gateway in basic mode)
    bus = MessageBus()
    gateway = None
    try:
        gateway = ModelGateway()
    except Exception:
        pass

    classifier = ClassifierAgent(bus=bus, gateway=gateway)

    result = classifier.classify(
        doc_id=doc_id,
        filename=filename,
        front_matter_text=front_matter_text,
        page_count=page_count,
    )

    if progress_cb:
        progress_cb("classify_done", 1, 1)

    return result


def _classify_document_from_pages(
    doc_id: str,
    filename: str,
    pages: list,
    pdf_path: str,
    progress_cb=None,
):
    """Run the classifier using raw page text (before canonicalization).

    Used when the preprocessor is enabled so classification results are
    available before chunking strategy selection.
    """
    if progress_cb:
        progress_cb("classify", 0, 1)

    import fitz
    front_matter_limit = settings.front_matter_pages
    front_text_parts = []
    pdf = fitz.open(pdf_path)
    try:
        for i in range(min(front_matter_limit, pdf.page_count)):
            page = pdf.load_page(i)
            text = page.get_text("text") or ""
            front_text_parts.append(text)
    finally:
        pdf.close()
    front_matter_text = "\n\n".join(front_text_parts)

    bus = MessageBus()
    gateway = None
    try:
        gateway = ModelGateway()
    except Exception:
        pass

    classifier = ClassifierAgent(bus=bus, gateway=gateway)
    result = classifier.classify(
        doc_id=doc_id,
        filename=filename,
        front_matter_text=front_matter_text,
        page_count=len(pages),
    )

    if progress_cb:
        progress_cb("classify_done", 1, 1)

    return result


def _run_preprocessor(
    doc_id: str,
    filename: str,
    pages: list,
    classification,
    progress_cb=None,
):
    """Run the preprocessor agent to determine chunking strategy (§4.9)."""
    if progress_cb:
        progress_cb("preprocess", 0, 1)

    # Build triage summary from page records
    triage_summary = _build_triage_summary(pages)

    bus = MessageBus()
    preprocessor = PreprocessorAgent(bus=bus)

    inp = PreprocessorInput(
        doc_id=doc_id,
        filename=filename,
        page_count=len(pages),
        document_type=classification.document_type if classification else None,
        classification_label=classification.classification_label if classification else None,
        classification_confidence=classification.confidence if classification else 0.0,
        triage_summary=triage_summary,
    )

    result = preprocessor.determine_strategy(inp)

    if progress_cb:
        progress_cb("preprocess_done", 1, 1)

    return result


def _build_triage_summary(pages: list) -> dict:
    """Aggregate page-level triage metrics into a document-level summary."""
    if not pages:
        return {}

    total_text = 0
    total_image_coverage = 0.0
    di_pages = 0

    for p in pages:
        metrics = p.triage_metrics
        total_text += metrics.text_length
        total_image_coverage += metrics.image_coverage_ratio
        if p.triage_decision == "di_required":
            di_pages += 1

    page_count = len(pages)
    return {
        "total_text_length": total_text,
        "avg_text_length": total_text / page_count,
        "avg_image_coverage": total_image_coverage / page_count,
        "di_page_count": di_pages,
        "di_page_ratio": di_pages / page_count,
        "page_count": page_count,
    }


def _record_chunking_outcome(
    doc_id: str,
    preprocess_result,
    classification,
    chunks: list,
    page_count: int,
    processing_time_ms: float,
) -> None:
    """Record chunking outcome for preprocessor learning loop (§4.9)."""
    if not chunks:
        return

    total = len(chunks)
    table_count = sum(1 for c in chunks if c.chunk_type == "table")
    heading_count = sum(1 for c in chunks if c.chunk_type == "heading")
    boilerplate_count = sum(1 for c in chunks if c.chunk_type == "boilerplate")
    avg_tokens = sum(len(c.text_content.split()) for c in chunks) / total

    # Compute a simple quality heuristic:
    # - Penalize very high boilerplate ratio
    # - Penalize very low or very high chunk counts relative to pages
    chunks_per_page = total / max(page_count, 1)
    boilerplate_ratio = boilerplate_count / total
    quality = 1.0
    if boilerplate_ratio > 0.3:
        quality -= 0.3 * boilerplate_ratio
    if chunks_per_page < 1:
        quality -= 0.2
    elif chunks_per_page > 50:
        quality -= 0.1
    quality = max(0.0, min(1.0, quality))

    outcome = ChunkingOutcome(
        doc_id=doc_id,
        strategy_name=preprocess_result.chunking_strategy.strategy_name,
        document_type=classification.document_type if classification else "unknown",
        classification_label=classification.classification_label if classification else "unknown",
        page_count=page_count,
        chunk_count=total,
        avg_chunk_tokens=avg_tokens,
        table_chunk_ratio=table_count / total,
        heading_chunk_ratio=heading_count / total,
        boilerplate_ratio=boilerplate_ratio,
        processing_time_ms=processing_time_ms,
        quality_score=quality,
    )

    bus = MessageBus()
    preprocessor = PreprocessorAgent(bus=bus)
    preprocessor.record_outcome(outcome)
