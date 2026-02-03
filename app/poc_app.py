import os
import sys
import tempfile
import time
from typing import List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from grounding.highlight import build_annotations, build_annotations_with_index
from ingestion.ingest_pipeline import ingest_and_chunk
from retrieval.vector_search import search
from storage.db import get_connection
from storage import repo
from synthesis.openai_client import synthesize_answer


st.set_page_config(page_title="IDP RAG PoC", layout="wide")
st.title("IDP RAG PoC")

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
            st.session_state.query_results = []
            st.session_state.annotations = []

    force_reprocess = st.checkbox("Force reprocess", value=False)
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded:
        st.session_state.pdf_path = _save_upload(uploaded)
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
            st.session_state.doc_id = ingest_and_chunk(
                st.session_state.pdf_path,
                filename=uploaded.name,
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
    run_query = st.button("Run query", disabled=st.session_state.is_processing or not st.session_state.doc_id)

    if run_query:
        if not st.session_state.doc_id:
            st.warning("Please process a document first.")
        else:
            results = search(st.session_state.doc_id, query_input, top_k=3)
            st.session_state.query_results = results
            annotations, index_map = build_annotations_with_index(results)
            st.session_state.annotations = annotations
            st.session_state.annotation_index_map = index_map
            try:
                answer = synthesize_answer(query_input, results)
            except RuntimeError as exc:
                answer = f"Synthesis unavailable: {exc}"
            st.subheader("Answer")
            st.write(answer)

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
