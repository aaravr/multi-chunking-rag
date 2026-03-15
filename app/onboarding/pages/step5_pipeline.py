"""Step 5 — Pipeline Settings."""

import streamlit as st
from app.onboarding.mock_data import PIPELINE_SETTINGS
from app.onboarding.components.layout import (
    render_page_title, render_section_header, render_recommendation,
    render_summary_card, render_action_bar,
)


def render(current_step: int = 5):
    render_page_title(
        "Pipeline Settings",
        "Configure the document processing pipeline — chunking, extraction, retrieval, and verification strategies.",
    )

    col_main, col_side = st.columns([3, 1])

    with col_main:
        render_section_header("Processing Profile")
        c1, c2, c3 = st.columns(3)
        profiles = PIPELINE_SETTINGS["profiles"]
        selected = PIPELINE_SETTINGS["profile"]

        for col, (key, prof) in zip([c1, c2, c3], profiles.items()):
            with col:
                cls = "selected" if key == selected else ""
                icon = "⭐" if key == "recommended" else "⚖️" if key == "balanced" else "🎯"
                st.markdown(f"""
                <div class="mode-card {cls}" style="text-align:left;min-height:180px">
                    <div class="mode-icon">{icon}</div>
                    <div class="mode-title">{prof['label']}</div>
                    <div class="mode-desc">{prof['description']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        active = profiles[selected]
        render_section_header(f"Configuration — {active['label']} Profile")

        rows = [
            ("Chunking Strategy", active["chunking"]),
            ("Extraction Method", active["extraction"]),
            ("Reranking", active["reranking"]),
            ("Retrieval Strategy", active["retrieval"]),
            ("Verification", active["verification"]),
            ("Temperature", str(active["temperature"])),
            ("Confidence Threshold", f"{active['confidence_threshold']:.0%}"),
        ]
        rows_html = ""
        for label, value in rows:
            rows_html += f"""
            <tr>
                <td style="font-weight:500;color:#64748b;width:220px">{label}</td>
                <td style="font-weight:500">{value}</td>
            </tr>"""

        st.markdown(f"""
        <table class="clean-table">
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        render_section_header("Advanced Configuration")
        with st.expander("Fine-tune pipeline parameters", expanded=False):
            ca, cb = st.columns(2)
            with ca:
                st.selectbox("Chunking Strategy", [
                    "Late Chunking — Semantic", "Late Chunking — Sliding Window",
                    "Late Chunking — Clause-Aware", "Late Chunking — Semantic + Clause-Aware",
                ], index=1, key="adv_chunking")
                st.selectbox("Extraction Method", [
                    "Regex-first + LLM fallback", "LLM-only (GPT-4o)",
                    "LLM-only (GPT-4o-mini)", "Regex-only",
                ], index=0, key="adv_extraction")
                st.selectbox("Embedding Model", [
                    "nomic-ai/modernbert-embed-base (768-dim)",
                    "text-embedding-3-small (1536-dim)",
                ], index=0, key="adv_embedding")
            with cb:
                st.slider("Confidence Threshold", 0.50, 0.99, 0.80, 0.01, key="adv_confidence")
                st.slider("Max Macro Chunk Tokens", 2048, 8192, 8192, 256, key="adv_macro")
                st.slider("Child Span Tokens", 128, 512, 256, 32, key="adv_child")

    with col_side:
        render_recommendation(
            "Profile: Balanced",
            "The <strong>Balanced</strong> profile is selected. It uses hybrid retrieval (BM25 + Vector) "
            "with regex-first extraction and LLM fallback. This provides good accuracy while keeping "
            "processing cost moderate."
        )

        render_summary_card("Pipeline Summary", [
            ("Est. Latency", "~3.2s/doc"),
            ("Est. Cost", "~$0.04/doc"),
            ("Verification", "Enabled", "#16a34a"),
            ("Audit Trail", "Full", "#16a34a"),
        ])

        render_recommendation(
            "Invariant: Temperature = 0",
            "Per platform policy (§2.3), all synthesis and extraction LLM calls use "
            "<strong>temperature = 0.0</strong> for deterministic, reproducible outputs. "
            "This is enforced at the Model Gateway level.",
            variant="warning",
        )

    render_action_bar(current_step)
