"""Step 6 — Evaluation Plan."""

import streamlit as st
from app.onboarding.mock_data import EVALUATION_PLAN
from app.onboarding.components.layout import (
    render_section_header, render_recommendation, render_badge,
    render_readiness_ring, render_nav_buttons,
)


def render(current_step: int = 6):
    st.markdown("## Evaluation Plan")
    st.markdown('<p style="color:#5D6D7E;font-size:0.85rem;margin-top:-0.5rem">Define quality gates, target deployment mode, evaluation segments, and readiness thresholds.</p>', unsafe_allow_html=True)

    col_main, col_side = st.columns([3, 1])

    with col_main:
        render_section_header("Target Deployment Mode")
        cols = st.columns(5)
        for col, (key, mode) in zip(cols, EVALUATION_PLAN["modes"].items()):
            with col:
                cls = "selected" if key == EVALUATION_PLAN["target_mode"] else ""
                st.markdown(f"""
                <div class="mode-card {cls}" style="min-height:150px">
                    <div class="mode-icon">{mode['icon']}</div>
                    <div class="mode-title">{mode['label']}</div>
                    <div class="mode-desc">{mode['description']}</div>
                    <div style="margin-top:0.5rem;font-size:0.7rem;color:#5D6D7E">Min accuracy: {mode['min_accuracy']:.0%}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        render_section_header("Quality Gates")
        rows_html = ""
        for gate in EVALUATION_PLAN["quality_gates"]:
            if gate["status"] == "pass":
                badge = render_badge("Pass", "pass")
                icon = "✅"
            elif gate["status"] == "warn":
                badge = render_badge("Warning", "warn")
                icon = "⚠️"
            else:
                badge = render_badge("Fail", "fail")
                icon = "❌"

            # Format threshold and current
            if gate["threshold"] <= 1.0 and gate["name"] != "Confidence Calibration (ECE)":
                thresh_str = f"{gate['threshold']:.0%}"
                curr_str = f"{gate['current']:.0%}"
            elif gate["name"] == "Confidence Calibration (ECE)":
                thresh_str = f"≤ {gate['threshold']}"
                curr_str = f"{gate['current']}"
            else:
                thresh_str = f"{gate['threshold']}"
                curr_str = f"{gate['current']}"

            rows_html += f"""
            <tr>
                <td>{icon}</td>
                <td style="font-weight:500">{gate['name']}</td>
                <td style="text-align:center">{thresh_str}</td>
                <td style="text-align:center;font-weight:600">{curr_str}</td>
                <td>{badge}</td>
            </tr>"""

        st.markdown(f"""
        <table class="clean-table">
            <thead><tr>
                <th style="width:30px"></th><th>Quality Gate</th>
                <th style="text-align:center">Threshold</th>
                <th style="text-align:center">Current</th><th>Status</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        render_section_header("Evaluation Segments")
        for seg in EVALUATION_PLAN["segments"]:
            st.markdown(f"""
            <div style="margin-bottom:0.75rem;padding:0.75rem;background:white;border:1px solid #D5D8DC;border-radius:8px">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem">
                    <span style="font-weight:600;font-size:0.85rem">{seg['dimension']}</span>
                    <span style="font-size:0.75rem;color:#5D6D7E">Coverage: {seg['coverage']}</span>
                </div>
                <div style="display:flex;gap:0.4rem;flex-wrap:wrap">
                    {"".join(f'<span style="background:#F4F6F9;padding:0.2rem 0.5rem;border-radius:4px;font-size:0.72rem;color:#2C3E50">{s}</span>' for s in seg['slices'])}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_side:
        render_readiness_ring(EVALUATION_PLAN["readiness_score"])
        st.markdown('<p style="text-align:center;font-size:0.8rem;color:#5D6D7E;margin-top:0.5rem">Readiness Score</p>', unsafe_allow_html=True)
        st.markdown("---")

        render_recommendation(
            "Target: Shadow Deployment",
            "Based on current quality gate results, the workspace is <strong>ready for Shadow deployment</strong> "
            "(5 of 6 gates passing). The Entity Resolution Rate (87%) is below the 90% threshold — "
            "this is acceptable for Shadow mode but must be resolved before Canary or Production."
        )
        st.markdown("---")

        render_recommendation(
            "Segment Coverage",
            "Evaluation covers <strong>3 dimensions</strong> with high slice coverage. "
            "Risk Rating segment coverage (94%) is slightly below 100% — "
            "the 'Doubtful/Loss' slice has limited sample size (6 documents)."
        )

    render_nav_buttons(current_step)
