"""Step 4 — Ground Truth Configuration."""

import streamlit as st
from app.onboarding.mock_data import GROUND_TRUTH
from app.onboarding.components.layout import (
    render_page_title, render_section_header, render_metric_card,
    render_recommendation, render_badge, render_summary_card,
    render_action_bar,
)


def render(current_step: int = 4):
    render_page_title(
        "Ground Truth & Evaluation Mode",
        "Define how extraction quality will be measured — with labeled ground truth, partial labels, or review-based evaluation.",
    )

    col_main, col_side = st.columns([3, 1])

    with col_main:
        render_section_header("Evaluation Mode")
        c1, c2, c3 = st.columns(3)
        gt_modes = [
            ("📊", "Full Ground Truth", "Complete labeled dataset for all fields and documents. Enables automated accuracy measurement.", False),
            ("📋", "Partial Ground Truth", "Labels available for a subset of documents or fields. Combines automated + review-based evaluation.", True),
            ("👁️", "No Ground Truth", "No pre-existing labels. Quality assessed through expert review and inter-annotator agreement.", False),
        ]
        for col, (icon, title, desc, selected) in zip([c1, c2, c3], gt_modes):
            with col:
                cls = "selected" if selected else ""
                st.markdown(f"""
                <div class="mode-card {cls}" style="min-height:160px">
                    <div class="mode-icon">{icon}</div>
                    <div class="mode-title">{title}</div>
                    <div class="mode-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        render_section_header("Label Coverage")
        c4, c5, c6, c7 = st.columns(4)
        with c4:
            render_metric_card("Labeled Documents", str(GROUND_TRUTH["labeled_documents"]), f"of {GROUND_TRUTH['total_documents']} total")
        with c5:
            render_metric_card("Coverage", f"{GROUND_TRUTH['label_coverage_pct']}%", "Document-level coverage")
        with c6:
            render_metric_card("Unlabeled", str(GROUND_TRUTH["unlabeled_documents"]), "Will use review-based evaluation")
        with c7:
            render_metric_card("Validation", "3 warnings", "Date format inconsistencies")

        render_section_header("Field-Level Coverage")
        rows_html = ""
        for fname, info in GROUND_TRUTH["field_coverage"].items():
            pct = info["pct"]
            bar_color = "#16a34a" if pct >= 90 else "#d97706" if pct >= 60 else "#dc2626"
            badge = render_badge(f"{pct:.0f}%", "pass" if pct >= 90 else "warn" if pct >= 60 else "fail")
            rows_html += f"""
            <tr>
                <td style="font-family:monospace;font-size:0.78rem;font-weight:500">{fname}</td>
                <td style="text-align:right">{info['labeled']}</td>
                <td style="text-align:right">{info['total']}</td>
                <td style="width:200px">
                    <div class="progress-bar-track">
                        <div class="progress-bar-fill" style="width:{pct}%;background:{bar_color}"></div>
                    </div>
                </td>
                <td>{badge}</td>
            </tr>"""

        st.markdown(f"""
        <table class="clean-table">
            <thead><tr>
                <th>Field</th><th style="text-align:right">Labeled</th><th style="text-align:right">Total</th>
                <th>Coverage</th><th>Status</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        render_section_header("Label Source")
        c8, c9 = st.columns(2)
        with c8:
            st.text_input("Source System", value=GROUND_TRUTH["source"], disabled=True, key="gt_source")
            st.text_input("Import Format", value=GROUND_TRUTH["format"], disabled=True, key="gt_format")
        with c9:
            st.text_input("Validation Status", value=GROUND_TRUTH["validation_status"], disabled=True, key="gt_validation")
            st.button("Upload Additional Labels", key="upload_labels", use_container_width=True)

    with col_side:
        render_recommendation(
            "Partial Ground Truth Strategy",
            "With <strong>20.8% label coverage</strong>, evaluation will use a hybrid approach: "
            "<strong>automated accuracy</strong> on the 142 labeled documents and "
            "<strong>expert review</strong> on a sampled subset of unlabeled documents."
        )

        render_recommendation(
            "Coverage Gaps",
            "Fields <strong>analyst_name</strong> (31.7%) and <strong>financial_covenant_breach</strong> "
            "(50.7%) have the lowest label coverage. Quality assessment will rely primarily on "
            "expert review with confidence-based sampling.",
            variant="warning",
        )

        render_summary_card("Evaluation Mix", [
            ("Automated (labeled)", "20.8%", "#16a34a"),
            ("Expert review (sampled)", "~15%", "#E60000"),
            ("Confidence-only", "~64%", "#999999"),
        ])

    render_action_bar(current_step)
