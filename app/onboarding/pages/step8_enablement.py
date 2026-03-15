"""Step 8 — Production Enablement."""

import streamlit as st
from app.onboarding.mock_data import ENABLEMENT, EVALUATION_PLAN
from app.onboarding.components.layout import (
    render_page_title, render_section_header, render_recommendation,
    render_badge, render_summary_card, render_action_bar,
)


def render(current_step: int = 8):
    render_page_title(
        "Production Enablement",
        "Final enablement checklist, governance review, and deployment confirmation.",
    )

    col_main, col_side = st.columns([3, 1])

    with col_main:
        render_section_header("Deployment Mode")
        cols = st.columns(5)
        modes = EVALUATION_PLAN["modes"]
        target = ENABLEMENT["target_mode"]
        for col, (key, mode) in zip(cols, modes.items()):
            with col:
                cls = "selected" if key == target else ""
                st.markdown(f"""
                <div class="mode-card {cls}" style="min-height:120px">
                    <div class="mode-icon">{mode['icon']}</div>
                    <div class="mode-title">{mode['label']}</div>
                    <div class="mode-desc" style="font-size:0.7rem">{mode['description']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        render_section_header("Enablement Checklist")
        categories = {}
        for item in ENABLEMENT["checklist"]:
            cat = item["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(item)

        for cat, items in categories.items():
            st.markdown(
                f'<div style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;'
                f'letter-spacing:0.5px;margin:0.75rem 0 0.25rem 0">{cat}</div>',
                unsafe_allow_html=True,
            )
            for item in items:
                if item["status"] == "complete":
                    icon = "✅"
                    color = "#0f172a"
                else:
                    icon = "⏳"
                    color = "#d97706"
                st.markdown(f"""
                <div class="checklist-item">
                    <span class="checklist-icon">{icon}</span>
                    <span class="checklist-text" style="color:{color}">{item['item']}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        render_section_header("Governance & Compliance")
        gov = ENABLEMENT["governance"]
        rows_html = ""
        for label, value in [
            ("Boundary Key", gov["boundary_key"]),
            ("Data Classification", gov["data_classification"]),
            ("Audit Trail", gov["audit_trail"]),
            ("Retention Policy", gov["retention_policy"]),
        ]:
            rows_html += f"""
            <tr>
                <td style="font-weight:500;color:#64748b;width:200px">{label}</td>
                <td style="font-weight:500">{value}</td>
            </tr>"""

        st.markdown(f"""
        <table class="clean-table">
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # CTA
        complete_count = sum(1 for i in ENABLEMENT["checklist"] if i["status"] == "complete")
        total_count = len(ENABLEMENT["checklist"])

        if complete_count < total_count:
            st.warning(f"⚠️ {total_count - complete_count} checklist items pending. Resolve all items before enabling deployment.")

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.button(
                f"Confirm {target.title()} Enablement",
                key="enable_deployment",
                type="primary",
                use_container_width=True,
                disabled=(complete_count < total_count),
            )

    with col_side:
        complete_pct = int(complete_count / total_count * 100)
        st.markdown(f"""
        <div class="metric-card" style="text-align:center">
            <div class="metric-label">Checklist Progress</div>
            <div class="metric-value">{complete_count}/{total_count}</div>
            <div class="metric-sub">{complete_pct}% complete</div>
            <div class="progress-bar-track" style="margin-top:0.5rem">
                <div class="progress-bar-fill" style="width:{complete_pct}%;background:#16a34a"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        render_recommendation(
            "Pending: 2 Approvals",
            "Compliance and Technology sign-offs are still pending. Contact "
            "<strong>Maria Santos</strong> (Compliance) and <strong>David Park</strong> "
            "(Technology) to complete the review process.",
            variant="warning",
        )

        render_recommendation(
            "Rollback Plan",
            "A rollback plan is in place. If Shadow deployment shows accuracy degradation "
            "below 85%, the system will auto-disable processing and alert the operations owner. "
            "Previous stable configuration (v1.1.0) is preserved for instant rollback."
        )

        render_summary_card("Post-Enablement", [
            ("Mode", "Monitoring & Maintenance"),
            ("Dashboards", "Live"),
            ("Drift Detection", "Enabled"),
            ("Alerting", "Automated"),
        ])

    render_action_bar(current_step)
