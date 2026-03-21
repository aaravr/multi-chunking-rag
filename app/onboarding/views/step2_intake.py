"""Step 2 — Document Intake."""

import streamlit as st
from app.onboarding.mock_data import INTAKE_SOURCES, INTAKE_SUMMARY
from app.onboarding.components.layout import (
    render_page_title, render_section_header, render_metric_card,
    render_recommendation, render_badge, render_summary_card,
    render_action_bar,
)


def render(current_step: int = 2):
    render_page_title(
        "Document Intake",
        "Connect document sources, configure sampling, and review corpus quality before processing.",
    )

    col_main, col_side = st.columns([3, 1])

    with col_main:
        render_section_header("Source Configuration")

        source_mode = st.radio(
            "Document Source Mode",
            ["Central Repository", "Direct Upload", "Connector (API)"],
            index=0, horizontal=True, key="intake_mode",
        )

        st.markdown("**Connected Sources**")
        rows_html = ""
        for src in INTAKE_SOURCES:
            status_badge = render_badge("Connected", "pass") if src["status"] == "connected" else render_badge("Uploaded", "info")
            rows_html += f"""
            <tr>
                <td style="font-weight:500">{src['name']}</td>
                <td>{src['type'].title()}</td>
                <td><code style="font-size:0.75rem;background:#f5f5f5;padding:2px 6px;border-radius:3px">{src['path'] or '—'}</code></td>
                <td style="text-align:right;font-weight:600">{src['count']:,}</td>
                <td>{status_badge}</td>
            </tr>"""

        st.markdown(f"""
        <table class="clean-table">
            <thead><tr>
                <th>Source Name</th><th>Type</th><th>Path</th><th style="text-align:right">Documents</th><th>Status</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        render_section_header("Corpus Quality Preview")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card("Total Documents", f"{INTAKE_SUMMARY['total_documents']:,}", "Across all sources")
        with c2:
            render_metric_card("Total Pages", f"{INTAKE_SUMMARY['total_pages']:,}", f"Avg {INTAKE_SUMMARY['avg_pages']} pages/doc")
        with c3:
            render_metric_card("High Quality", f"{INTAKE_SUMMARY['quality_scores']['High (>0.9)']:,}", f"of {INTAKE_SUMMARY['total_documents']:,} documents")
        with c4:
            render_metric_card("Duplicates", str(INTAKE_SUMMARY["duplicates_detected"]), "Auto-detected, flagged for review")

        render_section_header("Breakdown")
        c5, c6 = st.columns(2)
        with c5:
            st.markdown("**File Types**")
            for ft, count in INTAKE_SUMMARY["file_types"].items():
                pct = count / INTAKE_SUMMARY["total_documents"] * 100
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.35rem">
                    <span style="font-size:0.8rem;width:50px;font-weight:500;color:#000000">{ft}</span>
                    <div style="flex:1;height:10px;border-radius:5px;overflow:hidden" class="progress-bar-track">
                        <div class="progress-bar-fill" style="width:{pct}%;background:#E60000"></div>
                    </div>
                    <span style="font-size:0.75rem;color:#666666;width:70px;text-align:right">{count:,} ({pct:.0f}%)</span>
                </div>
                """, unsafe_allow_html=True)

        with c6:
            st.markdown("**Languages**")
            for lang, count in INTAKE_SUMMARY["languages"].items():
                pct = count / INTAKE_SUMMARY["total_documents"] * 100
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.35rem">
                    <span style="font-size:0.8rem;width:60px;font-weight:500;color:#000000">{lang}</span>
                    <div style="flex:1;height:10px;border-radius:5px;overflow:hidden" class="progress-bar-track">
                        <div class="progress-bar-fill" style="width:{pct}%;background:#16a34a"></div>
                    </div>
                    <span style="font-size:0.75rem;color:#666666;width:70px;text-align:right">{count:,} ({pct:.0f}%)</span>
                </div>
                """, unsafe_allow_html=True)

    with col_side:
        render_recommendation(
            "Sample Strategy",
            "A <strong>stratified sample of 50 documents</strong> is recommended for initial evaluation. "
            "This covers all document types and quality tiers while keeping processing time under 5 hours."
        )

        render_summary_card("Est. Processing", [
            ("Time", INTAKE_SUMMARY["estimated_processing_time"]),
            ("Sample Size", f"{INTAKE_SUMMARY['sample_size']} documents"),
            ("Strategy", "Stratified"),
        ])

        render_recommendation(
            "Quality Note",
            f"<strong>{INTAKE_SUMMARY['quality_scores']['Low (<0.7)']}</strong> documents flagged as low quality. "
            "Consider excluding or prioritizing manual review for these during evaluation.",
            variant="warning",
        )

    render_action_bar(current_step)
