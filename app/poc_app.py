import os
import sys
import tempfile
import time
import re
from typing import List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from grounding.highlight import build_annotations, build_annotations_with_index
from ingestion.ingest_pipeline import ingest_and_chunk
from core.logging import configure_logging
from storage.schema_contract import check_schema_contract
from retrieval.metadata import detect_fact_name, handle_metadata_query
from retrieval.router import classify_query, search_with_intent_debug
from storage.db import get_connection
from storage import repo
from synthesis.openai_client import (
    synthesize_answer,
    synthesize_coverage_answer,
    synthesize_coverage_attribute,
)
from synthesis.verifier import verify_coverage, verify_coverage_attribute
from core.config import settings


st.set_page_config(page_title="IDP RAG PoC", layout="wide")
st.title("IDP RAG PoC")
configure_logging()
try:
    check_schema_contract()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
if "query_results" not in st.session_state:
    st.session_state.query_results = []
if "annotations" not in st.session_state:
    st.session_state.annotations = []
if "selected_index" not in st.session_state:
    st.session_state.selected_index = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "progress_state" not in st.session_state:
    st.session_state.progress_state = {}
if "scroll_to_page" not in st.session_state:
    st.session_state.scroll_to_page = None
if "scroll_to_annotation" not in st.session_state:
    st.session_state.scroll_to_annotation = None
if "annotation_index_map" not in st.session_state:
    st.session_state.annotation_index_map = {}
if "upload_name" not in st.session_state:
    st.session_state.upload_name = None
if "query_debug" not in st.session_state:
    st.session_state.query_debug = None


def _select_cited_chunk(answer_text: str, chunks: List) -> object:
    if not answer_text:
        return chunks[0] if chunks else None
    indices = re.findall(r"\[C(\d+)\]", answer_text)
    for idx in indices:
        pos = int(idx) - 1
        if 0 <= pos < len(chunks):
            return chunks[pos]
    return chunks[0] if chunks else None


def _save_upload(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def _load_triage_summary(doc_id: str) -> str:
    with get_connection() as conn:
        pages = repo.fetch_pages(conn, doc_id)
    di_pages = [p for p in pages if p.triage_decision == "di_required"]
    native_pages = [p for p in pages if p.triage_decision == "native_only"]
    return f"DI: {len(di_pages)} pages | Native: {len(native_pages)} pages"


def _get_cached_pdf_path(doc_id: str) -> str:
    data_dir = os.getenv("IDP_DATA_DIR", "data")
    return os.path.join(data_dir, doc_id, "source.pdf")


with st.sidebar:
    st.header("Ingestion")
    with get_connection() as conn:
        documents = repo.fetch_documents(conn)
    if documents:
        doc_options = {f"{d.filename} ({d.doc_id[:8]})": d for d in documents}
        selected_label = st.selectbox("Reuse existing document", list(doc_options.keys()))
        if st.button("Load selected"):
            selected = doc_options[selected_label]
            st.session_state.doc_id = selected.doc_id
            cached_path = _get_cached_pdf_path(selected.doc_id)
            if os.path.exists(cached_path):
                st.session_state.pdf_path = cached_path
            else:
                st.warning("Cached PDF not found. Upload the file to view it.")
            st.session_state.upload_name = selected.filename
            st.session_state.query_results = []
            st.session_state.annotations = []

    force_reprocess = st.checkbox("Force reprocess", value=False)
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded:
        st.session_state.pdf_path = _save_upload(uploaded)
        st.session_state.upload_name = uploaded.name
    process = st.button(
        "Process document",
        disabled=st.session_state.is_processing or not st.session_state.pdf_path,
    )
    if process and st.session_state.pdf_path:
        st.session_state.is_processing = True
        status_box = st.status("Processing document...", expanded=True)
        progress = st.progress(0)
        stage_text = st.empty()
        eta_text = st.empty()
        global_eta_text = st.empty()
        st.session_state.progress_state = {
            "stage": None,
            "start_time": None,
            "global_start_time": time.time(),
            "stage_units": {},
            "stage_done": {},
        }

        def _progress(stage, current, total):
            now = time.time()
            if st.session_state.progress_state.get("stage") != stage:
                st.session_state.progress_state["stage"] = stage
                st.session_state.progress_state["start_time"] = now
            if total:
                st.session_state.progress_state["stage_units"][stage] = total
                st.session_state.progress_state["stage_done"][stage] = current
            if total:
                ratio = min(max(current / total, 0.0), 1.0)
            else:
                ratio = 0.0
            progress.progress(ratio)
            stage_text.write(f"{stage}: {current}/{total}")
            if total and current:
                elapsed = now - st.session_state.progress_state.get("start_time", now)
                per_unit = elapsed / max(current, 1)
                remaining = per_unit * (total - current)
                eta_text.write(f"Estimated remaining: {int(remaining)}s")
            else:
                eta_text.write("Estimated remaining: calculating...")

            global_total = sum(st.session_state.progress_state["stage_units"].values())
            global_done = sum(st.session_state.progress_state["stage_done"].values())
            if global_total and global_done:
                global_elapsed = now - st.session_state.progress_state.get(
                    "global_start_time", now
                )
                global_per_unit = global_elapsed / max(global_done, 1)
                global_remaining = global_per_unit * (global_total - global_done)
                global_eta_text.write(
                    f"Global ETA: {int(global_remaining)}s"
                )
            else:
                global_eta_text.write("Global ETA: calculating...")

        try:
            filename = st.session_state.upload_name or os.path.basename(
                st.session_state.pdf_path
            )
            st.session_state.doc_id = ingest_and_chunk(
                st.session_state.pdf_path,
                filename=filename,
                progress_cb=_progress,
                force_reprocess=force_reprocess,
            )
            status_box.update(state="complete")
            st.success(f"Ingested: {st.session_state.doc_id}")
        except Exception as exc:
            status_box.update(state="error")
            st.error(f"Ingestion failed: {exc}")
        finally:
            st.session_state.is_processing = False
    if st.session_state.doc_id:
        st.write(_load_triage_summary(st.session_state.doc_id))


left, right = st.columns([1, 1])

with left:
    if st.session_state.pdf_path:
        pdf_viewer(
            st.session_state.pdf_path,
            width="100%",
            height=900,
            annotations=st.session_state.annotations,
            render_text=True,
            zoom_level=1.0,
            scroll_to_annotation=st.session_state.scroll_to_annotation,
        )
    else:
        st.info("Upload a PDF to preview.")

with right:
    st.header("Extraction Dashboard")
    preset = st.radio("Preset attributes", ["CET1 Ratio", "Net Income", "Risk Exposure"], index=0)
    query_input = st.text_input("Free-form query", value=preset)
    st.checkbox("Enable verifier", value=settings.enable_verifier, key="enable_verifier")
    coverage_mode_options = ["deterministic", "llm_fallback", "llm_always"]
    default_mode = (
        settings.coverage_mode
        if settings.coverage_mode in set(coverage_mode_options)
        else "llm_fallback"
    )
    coverage_mode = st.selectbox(
        "CoverageQuery mode",
        coverage_mode_options,
        index=coverage_mode_options.index(default_mode),
        key="coverage_mode",
    )
    run_query = st.button("Run query", disabled=st.session_state.is_processing or not st.session_state.doc_id)

    if run_query:
        if not st.session_state.doc_id:
            st.warning("Please process a document first.")
        else:
            metadata_fact = detect_fact_name(query_input)
            if metadata_fact:
                answer, results, debug_info = handle_metadata_query(
                    st.session_state.doc_id,
                    query_input,
                    use_cache=settings.enable_document_facts,
                )
                st.session_state.query_results = results
                st.session_state.query_debug = debug_info
                annotations, index_map = build_annotations_with_index(results)
                st.session_state.annotations = annotations
                st.session_state.annotation_index_map = index_map
                st.subheader("Answer")
                st.write(answer)
                if st.session_state.query_debug is not None:
                    with st.expander("Debug details"):
                        st.json(st.session_state.query_debug)
            else:
                intent = classify_query(query_input)
                try:
                    results, debug_info = search_with_intent_debug(
                        st.session_state.doc_id, query_input, top_k=3
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                    st.stop()
                st.session_state.query_results = results
                st.session_state.query_debug = debug_info
                annotations, index_map = build_annotations_with_index(results)
                st.session_state.annotations = annotations
                st.session_state.annotation_index_map = index_map
                os.environ["ENABLE_VERIFIER"] = "true" if st.session_state.enable_verifier else "false"
                try:
                    if intent.intent == "coverage":
                        os.environ["COVERAGE_MODE"] = coverage_mode
                        if intent.coverage_type == "attribute":
                            answer, mode_used = synthesize_coverage_attribute(
                                query_input, results
                            )
                            if st.session_state.enable_verifier:
                                verdict, rationale = verify_coverage_attribute(
                                    query_input, answer, results
                                )
                                answer = f"{answer}\n\nVerifier: {verdict}\n{rationale}"
                        elif intent.coverage_type == "pointer":
                            answer = synthesize_answer(query_input, results)
                            mode_used = "llm"
                        elif intent.coverage_type == "numeric_list":
                            answer, mode_used = synthesize_coverage_answer(
                                query_input,
                                results,
                                mode="llm_always",
                                status_filter=intent.status_filter,
                            )
                            if st.session_state.enable_verifier:
                                verdict, rationale = verify_coverage(
                                    query_input, answer, results
                                )
                                answer = f"{answer}\n\nVerifier: {verdict}\n{rationale}"
                        else:
                            answer, mode_used = synthesize_coverage_answer(
                                query_input,
                                results,
                                mode=coverage_mode,
                                status_filter=intent.status_filter,
                            )
                            if st.session_state.enable_verifier:
                                verdict, rationale = verify_coverage(
                                    query_input, answer, results
                                )
                                answer = f"{answer}\n\nVerifier: {verdict}\n{rationale}"
                    else:
                        answer = synthesize_answer(query_input, results)
                except RuntimeError as exc:
                    answer = f"Synthesis unavailable: {exc}"
                st.subheader("Answer")
                st.write(answer)
                if intent.intent == "coverage":
                    heading_paths = sorted(
                        {c.heading_path for c in results if c.heading_path}
                    )
                    section_ids = sorted(
                        {c.section_id for c in results if c.section_id}
                    )
                    pages = sorted({p for c in results for p in c.page_numbers})
                    page_range = (
                        f"{pages[0]}-{pages[-1]}" if pages else "unknown"
                    )
                    bbox_preview = "unavailable"
                    cited_chunk = _select_cited_chunk(answer, results)
                    if cited_chunk and cited_chunk.polygons:
                        bbox_preview = cited_chunk.polygons[0]
                    st.caption(f"Coverage mode used: {mode_used}")
                    st.caption(
                        "Coverage expansion: "
                        f"heading_path={heading_paths or ['unknown']}; "
                        f"section_id={section_ids or ['unknown']}; "
                        f"page_range={page_range}"
                    )
                    coverage_label = {
                        "attribute": "CoverageAttribute",
                        "numeric_list": "CoverageNumericList",
                        "pointer": "CoveragePointer",
                        "list": "CoverageList",
                    }.get(intent.coverage_type or "list", "CoverageList")
                    st.caption(f"Coverage subtype: {coverage_label}")
                    st.caption(f"Coverage bbox sample: {bbox_preview}")
                    if st.session_state.query_debug is not None:
                        with st.expander("Debug details"):
                            st.json(st.session_state.query_debug)

    if st.session_state.query_results:
        st.subheader("Sources")
        for idx, chunk in enumerate(st.session_state.query_results):
            label = f"Source {idx + 1} (p{','.join(map(str, chunk.page_numbers))})"
            if st.button(label):
                annotations, index_map = build_annotations_with_index([chunk], color="blue")
                st.session_state.annotations = annotations
                st.session_state.annotation_index_map = index_map
                st.session_state.scroll_to_annotation = index_map.get(chunk.chunk_id)
            lineage_label = f"Lineage {idx + 1}"
            if st.button(lineage_label):
                annotations, index_map = build_annotations_with_index([chunk], color="blue")
                st.session_state.annotations = annotations
                st.session_state.annotation_index_map = index_map
                st.session_state.scroll_to_annotation = index_map.get(chunk.chunk_id)
                st.info(
                    " | ".join(
                        [
                            f"chunk_id={chunk.chunk_id}",
                            f"pages={chunk.page_numbers}",
                            f"char={chunk.char_start}-{chunk.char_end}",
                            f"source={chunk.source_type}",
                            f"polygons={len(chunk.polygons)}",
                        ]
                    )
                )
