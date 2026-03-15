"""Shared layout components: header, wizard stepper, navigation, panels."""

import streamlit as st
from app.onboarding.mock_data import WIZARD_STEPS, WORKSPACE


def render_header():
    """Render the enterprise top header bar."""
    st.markdown(f"""
    <div class="top-header">
        <div>
            <h1>IDP Onboarding Studio</h1>
            <div class="header-meta">Enterprise Intelligent Document Processing Platform</div>
        </div>
        <div style="text-align:right">
            <div style="font-size:0.82rem;font-weight:600">{WORKSPACE['workspace_id']}</div>
            <div class="header-meta">{WORKSPACE['department']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_stepper(current_step: int):
    """Render the wizard progress stepper.

    Args:
        current_step: 1-indexed current step number.
    """
    html = '<div class="wizard-stepper">'
    for i, step in enumerate(WIZARD_STEPS):
        n = step["number"]
        if n < current_step:
            cls = "completed"
            icon = "✓"
        elif n == current_step:
            cls = "active"
            icon = str(n)
        else:
            cls = ""
            icon = str(n)

        if i > 0:
            conn_cls = "completed" if n <= current_step else ""
            html += f'<div class="wizard-connector {conn_cls}"></div>'

        html += f"""
        <div class="wizard-step {cls}">
            <div class="step-number">{icon}</div>
            <span>{step['title']}</span>
        </div>"""

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_section_header(title: str):
    """Render a section header with bottom border."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def render_metric_card(label: str, value: str, sub: str = ""):
    """Render a single KPI metric card."""
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_badge(text: str, variant: str = "info"):
    """Return badge HTML. Variants: pass, warn, fail, info, pending."""
    return f'<span class="badge badge-{variant}">{text}</span>'


def render_recommendation(title: str, body: str):
    """Render a recommendation/info panel."""
    st.markdown(f"""
    <div class="recommendation-panel">
        <div class="rec-title">💡 {title}</div>
        <div class="rec-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)


def render_readiness_ring(score: int):
    """Render a circular readiness score gauge."""
    st.markdown(f"""
    <div class="readiness-ring" style="--pct:{score}">
        <div class="readiness-ring-inner">
            <div class="readiness-ring-value">{score}</div>
            <div class="readiness-ring-label">Readiness</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_nav_buttons(current_step: int, total_steps: int = 8):
    """Render Previous / Next navigation buttons."""
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if current_step > 1:
            st.button("← Previous", key=f"prev_{current_step}", use_container_width=True)
    with col3:
        if current_step < total_steps:
            st.button("Next →", key=f"next_{current_step}", type="primary", use_container_width=True)
        elif current_step == total_steps:
            st.button("Confirm", key=f"confirm_{current_step}", type="primary", use_container_width=True)


def render_mini_chart(values: list, color: str = "#AED6F1"):
    """Render a mini inline bar chart from a list of values."""
    if not values:
        return
    max_val = max(values) or 1
    bars = ""
    for v in values:
        h = max(2, int((v / max_val) * 48))
        bars += f'<div class="mini-bar" style="height:{h}px;background:{color}"></div>'
    st.markdown(f'<div class="mini-chart">{bars}</div>', unsafe_allow_html=True)
