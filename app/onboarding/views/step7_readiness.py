"""Step 7 — Review & Readiness."""

import streamlit as st
from app.onboarding.mock_data import READINESS
from app.onboarding.components.layout import (
    render_page_title, render_section_header, render_metric_card,
    render_recommendation, render_badge, render_readiness_ring,
    render_summary_card, render_action_bar,
    render_stacked_bar_chart, render_donut_chart, render_trend_chart,
    render_kpi_delta,
)


def render(current_step: int = 7):
    render_page_title(
        "Review & Readiness",
        "Final readiness assessment — accuracy summary, field-level results, risk review, and sign-off tracking.",
    )

    # Top KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric_card("Overall Accuracy", f"{READINESS['overall_accuracy']:.1%}", "Across all fields")
    with c2:
        render_metric_card("Critical Fields", f"{READINESS['critical_field_accuracy']:.1%}", "5 critical fields")
    with c3:
        render_metric_card("Readiness Score", str(READINESS["readiness_score"]), READINESS["readiness_label"])
    with c4:
        render_metric_card("Ground Truth", READINESS["ground_truth_summary"]["coverage"], f"{READINESS['ground_truth_summary']['mode']}")
    with c5:
        render_metric_card("Recommendation", "Shadow", "Target deployment mode")

    col_main, col_side = st.columns([3, 1])

    with col_main:
        render_section_header("Field-Level Results")
        rows_html = ""
        for f in READINESS["field_results"]:
            acc_pct = f["accuracy"] * 100
            conf_pct = f["confidence"] * 100
            badge = render_badge("Pass", "pass") if f["status"] == "pass" else (
                render_badge("Review", "warn") if f["status"] == "warn" else render_badge("Fail", "fail")
            )
            bar_color = "#16a34a" if f["accuracy"] >= 0.90 else "#d97706" if f["accuracy"] >= 0.80 else "#dc2626"
            rows_html += f"""
            <tr>
                <td style="font-family:monospace;font-size:0.78rem;font-weight:500">{f['field']}</td>
                <td style="width:160px">
                    <div style="display:flex;align-items:center;gap:0.4rem">
                        <div style="flex:1" class="progress-bar-track">
                            <div class="progress-bar-fill" style="width:{acc_pct}%;background:{bar_color}"></div>
                        </div>
                        <span style="font-size:0.78rem;font-weight:600;width:42px;text-align:right">{acc_pct:.0f}%</span>
                    </div>
                </td>
                <td style="text-align:center;font-size:0.8rem">{conf_pct:.0f}%</td>
                <td style="text-align:center;font-size:0.8rem">{f['reviewed']}</td>
                <td>{badge}</td>
            </tr>"""

        st.markdown(f"""
        <table class="clean-table">
            <thead><tr>
                <th>Field</th><th>Accuracy</th>
                <th style="text-align:center">Confidence</th>
                <th style="text-align:center">Reviewed</th><th>Status</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Performance by Field (stacked bars) ──────────────────────
        render_section_header("Performance by Field")
        render_stacked_bar_chart(READINESS["field_performance"])

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Accuracy Trend ───────────────────────────────────────────
        render_section_header("Accuracy Trend — Evaluation Runs")
        trend = READINESS["accuracy_trend"]
        scaled = [int(v * 100) for v in trend]
        labels = [f"R{i+1}" for i in range(len(trend))]
        render_trend_chart(scaled, color="#E60000", target=88, height=70, labels=labels)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Business Impact KPIs ─────────────────────────────────────
        render_section_header("Business Impact")
        bi = READINESS["business_impact"]
        k1, k2, k3 = st.columns(3)
        with k1:
            render_kpi_delta("Auto-Processed", f"{bi['auto_processed_pct']:.0%}", bi["handling_time_delta"], "up")
        with k2:
            render_kpi_delta("Avg Handling Time", f"{bi['avg_handling_time_min']} min", bi["handling_time_delta"], "up")
        with k3:
            render_kpi_delta("Exceptions", str(bi["exceptions"]), bi["exceptions_delta"], "down")

        st.markdown("<br>", unsafe_allow_html=True)

        render_section_header("Risks & Mitigations")
        for risk in READINESS["risks"]:
            sev_color = "#d97706" if risk["severity"] == "Medium" else "#999999"
            sev_bg = "#fffbeb" if risk["severity"] == "Medium" else "#fafafa"
            st.markdown(f"""
            <div style="padding:0.85rem;margin-bottom:0.5rem;background:{sev_bg};border:1px solid #e5e5e5;border-left:4px solid {sev_color};border-radius:8px">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem">
                    <span style="font-weight:600;font-size:0.82rem;color:#000000">{risk['severity']} Risk</span>
                </div>
                <div style="font-size:0.8rem;color:#333333;margin-bottom:0.3rem">{risk['description']}</div>
                <div style="font-size:0.78rem;color:#16a34a"><strong>Mitigation:</strong> {risk['mitigation']}</div>
            </div>
            """, unsafe_allow_html=True)

        render_section_header("Sign-Off Tracking")
        for role, info in READINESS["sign_off"].items():
            if info["status"] == "approved":
                icon = '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="6" stroke="#16a34a" stroke-width="1.4"/><polyline points="4,7 6,9.5 10,4.5" stroke="#16a34a" stroke-width="1.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>'
                status_text = f"Approved by {info['name']} on {info['date']}"
                color = "#16a34a"
            else:
                icon = '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="6" stroke="#d97706" stroke-width="1.4" stroke-dasharray="3 2"/><circle cx="7" cy="7" r="1" fill="#d97706"/><line x1="7" y1="4" x2="7" y2="7" stroke="#d97706" stroke-width="1.2" stroke-linecap="round"/></svg>'
                status_text = f"Pending — {info['name']}"
                color = "#d97706"

            st.markdown(f"""
            <div class="checklist-item">
                <span class="checklist-icon">{icon}</span>
                <span class="checklist-text" style="color:{color};font-weight:500">{status_text}</span>
                <span class="checklist-category">{role.title()}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_side:
        render_readiness_ring(READINESS["readiness_score"])
        st.markdown(
            '<p style="text-align:center;font-size:0.8rem;color:#666666;margin-top:0.5rem">Overall Readiness</p>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Readiness Breakdown Donut ─────────────────────────────────
        breakdown = READINESS["readiness_breakdown"]
        total = sum(v for v, _, _ in breakdown)
        donut_segments = [(v / total * 100, c, l) for v, c, l in breakdown]
        render_donut_chart(
            donut_segments,
            center_value=f"{READINESS['readiness_score']}",
            center_label="Score",
            size=130,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        render_recommendation(
            "Readiness Assessment",
            f"The workspace achieves <strong>{READINESS['overall_accuracy']:.1%} overall accuracy</strong> "
            f"and <strong>{READINESS['critical_field_accuracy']:.1%} on critical fields</strong>. "
            "All critical field thresholds are met. Shadow deployment is recommended."
        )

        render_summary_card("Next Actions", [
            ("1", "Obtain compliance sign-off"),
            ("2", "Obtain technology sign-off"),
            ("3", "Review entity resolution gap"),
            ("4", "Proceed to enablement"),
        ])

    render_action_bar(current_step)
