"""Step 9 — Post-Go-Live Monitoring (standalone screen, not part of wizard)."""

import streamlit as st
from app.onboarding.mock_data import MONITORING
from app.onboarding.components.layout import (
    render_page_title, render_section_header, render_metric_card,
    render_recommendation, render_badge, render_mini_chart,
    render_summary_card,
)


def render():
    render_page_title(
        "Post-Go-Live Monitoring",
        "Live operational dashboard — volume, accuracy, field health, review burden, alerts, and maintenance actions.",
    )

    # ── Status bar ───────────────────────────────────────────────────
    status_color = "#16a34a" if MONITORING["status"] == "healthy" else "#d97706"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:1.5rem;padding:0.65rem 1.25rem;background:white;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:1rem">
        <div style="display:flex;align-items:center;gap:0.4rem">
            <div style="width:10px;height:10px;border-radius:50%;background:{status_color}"></div>
            <span style="font-weight:600;font-size:0.82rem;color:{status_color}">{MONITORING['status'].upper()}</span>
        </div>
        <span style="font-size:0.78rem;color:#64748b">Mode: <strong style="color:#0f172a">{MONITORING['deployment_mode']}</strong></span>
        <span style="font-size:0.78rem;color:#64748b">Model: <strong style="color:#0f172a">{MONITORING['model_version']}</strong></span>
        <span style="font-size:0.78rem;color:#64748b">Uptime: <strong style="color:#0f172a">{MONITORING['uptime']}</strong></span>
        <span style="font-size:0.78rem;color:#94a3b8;margin-left:auto">Last updated: {MONITORING['last_updated']}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    vol = MONITORING["volume"]
    acc = MONITORING["accuracy"]
    rev = MONITORING["review_burden"]
    with c1:
        render_metric_card("Today's Volume", str(vol["today"]), f"{vol['week']} this week")
    with c2:
        render_metric_card("Overall Accuracy", f"{acc['current']:.1%}", f"Target: {acc['target']:.0%}")
    with c3:
        render_metric_card("Critical Fields", f"{acc['critical_field']:.1%}", "5 critical fields")
    with c4:
        render_metric_card("Auto-Accept Rate", f"{rev['auto_accept_rate']:.1%}", f"{rev['manual_reviews']} manual reviews")
    with c5:
        render_metric_card("Escalations", str(rev["escalations"]), f"Avg review: {rev['avg_review_time_sec']}s")

    col_main, col_side = st.columns([3, 1])

    with col_main:
        # ── Volume Trend ─────────────────────────────────────────────
        render_section_header("Document Volume — Last 14 Days")
        render_mini_chart(vol["trend"], color="#3b82f6")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Accuracy Trend ───────────────────────────────────────────
        render_section_header("Accuracy Trend — Last 14 Days")
        acc_scaled = [int(v * 100) for v in acc["trend"]]
        render_mini_chart(acc_scaled, color="#16a34a")
        target_pct = int(acc["target"] * 100)
        st.markdown(
            f'<div style="font-size:0.72rem;color:#64748b;margin-top:0.25rem">Target line: {target_pct}%</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Field Health ─────────────────────────────────────────────
        render_section_header("Field Health")
        rows_html = ""
        for f in MONITORING["field_health"]:
            acc_pct = f["accuracy"] * 100
            bar_color = "#16a34a" if f["accuracy"] >= 0.90 else "#d97706" if f["accuracy"] >= 0.85 else "#dc2626"

            badge = {
                "healthy": render_badge("Healthy", "pass"),
                "warning": render_badge("Warning", "warn"),
            }.get(f["status"], render_badge("Attention", "fail"))

            drift_icon = "→" if f["drift"] == "stable" else "↑" if f["drift"] == "improving" else "↓"
            drift_color = "#64748b" if f["drift"] == "stable" else "#16a34a" if f["drift"] == "improving" else "#dc2626"

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
                <td style="text-align:center;font-size:0.8rem;color:{drift_color}">{drift_icon} {f['drift'].title()}</td>
                <td style="text-align:center;font-size:0.78rem">{f['volume']:,}</td>
                <td>{badge}</td>
            </tr>"""

        st.markdown(f"""
        <table class="clean-table">
            <thead><tr>
                <th>Field</th><th>Accuracy</th>
                <th style="text-align:center">Drift</th>
                <th style="text-align:center">Volume</th><th>Status</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Alerts & Incidents ───────────────────────────────────────
        render_section_header("Alerts & Incidents")
        for alert in MONITORING["alerts"]:
            if alert["severity"] == "warning":
                alert_cls = "warning"
                icon = "⚠️"
            else:
                alert_cls = "info"
                icon = "ℹ️"

            ack_badge = render_badge("Acknowledged", "pass") if alert["acknowledged"] else render_badge("New", "warn")
            st.markdown(f"""
            <div class="alert-row {alert_cls}">
                <div style="flex:1">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.2rem">
                        <span style="font-size:0.82rem">{icon} {alert['message']}</span>
                        {ack_badge}
                    </div>
                    <div class="alert-timestamp">{alert['timestamp']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Failure / Escalation Queue ───────────────────────────────
        render_section_header("Failure & Escalation Queue")
        rows_html = ""
        for item in MONITORING["failure_queue"]:
            badge = render_badge("Pending", "warn") if item["status"] == "pending" else render_badge("Resolved", "pass")
            rows_html += f"""
            <tr>
                <td style="font-family:monospace;font-size:0.76rem">{item['doc_id']}</td>
                <td style="font-size:0.8rem">{item['error']}</td>
                <td style="font-size:0.72rem;color:#94a3b8;white-space:nowrap">{item['timestamp']}</td>
                <td>{badge}</td>
            </tr>"""

        st.markdown(f"""
        <table class="clean-table">
            <thead><tr>
                <th>Document</th><th>Error</th><th>Time</th><th>Status</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

    with col_side:
        # ── Review Burden ────────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-card" style="text-align:center">
            <div class="metric-label">Review Burden</div>
            <div class="metric-value">{rev['auto_accept_rate']:.0%}</div>
            <div class="metric-sub">Auto-Accept Rate</div>
            <div class="progress-bar-track" style="margin-top:0.5rem">
                <div class="progress-bar-fill" style="width:{int(rev['auto_accept_rate'] * 100)}%;background:#16a34a"></div>
            </div>
            <div style="font-size:0.75rem;color:#64748b;margin-top:0.5rem">
                {rev['auto_accepted']} auto / {rev['manual_reviews']} manual / {rev['escalations']} escalated
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Maintenance Actions ──────────────────────────────────────
        st.markdown('<div class="metric-card"><div class="metric-label">Maintenance Actions</div>', unsafe_allow_html=True)
        for ma in MONITORING["maintenance_actions"]:
            if ma["status"] == "recommended":
                btn_style = "background:#1e40af;color:white"
                label = "Recommended"
            else:
                btn_style = "background:#f1f5f9;color:#475569"
                label = "Available"
            st.markdown(f"""
            <div style="padding:0.5rem 0;border-bottom:1px solid #f1f5f9">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.2rem">
                    <span style="font-size:0.8rem;font-weight:600;color:#0f172a">{ma['action']}</span>
                    <span style="font-size:0.65rem;padding:0.15rem 0.5rem;border-radius:4px;{btn_style};font-weight:600">{label}</span>
                </div>
                <div style="font-size:0.72rem;color:#64748b">{ma['description']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        render_recommendation(
            "Shadow Mode Active",
            "The workspace is running in <strong>Shadow mode</strong>. Extractions run in parallel with "
            "the existing system. Promote to <strong>Canary</strong> "
            "when accuracy exceeds 92% across all critical fields for 7 consecutive days."
        )

        render_summary_card("Model & Run Info", [
            ("Model", MONITORING['model_version']),
            ("Run", MONITORING['run_version']),
            ("Deployment", MONITORING['deployment_mode']),
            ("Uptime", MONITORING['uptime'], "#16a34a"),
        ])
