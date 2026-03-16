"""Shared layout components: top nav, breadcrumb stepper, sidebar, action bar, panels."""

import streamlit as st
from app.onboarding.mock_data import WIZARD_STEPS, WORKSPACE


# ── Navigation items for the top bar ──────────────────────────────────
NAV_ITEMS = ["Workspaces", "Documents", "Schemas", "Evaluation", "Review", "Production"]


def render_top_nav(active_section: str = "Workspaces"):
    """Render the enterprise top navigation bar."""
    links = ""
    for item in NAV_ITEMS:
        cls = "active" if item == active_section else ""
        links += f'<span class="nav-link {cls}">{item}</span>'

    st.markdown(f"""
    <div class="top-nav">
        <div class="nav-brand">
            <span class="nav-brand-name">Enterprise IDP Onboarding Studio</span>
            <span class="nav-brand-chevron">&#9662;</span>
        </div>
        <div class="nav-links">
            {links}
        </div>
        <div class="nav-right">
            <span class="nav-icon">&#128276;</span>
            <span class="nav-icon">&#9881;</span>
            <div class="nav-avatar">SC</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_breadcrumb(current_step: int):
    """Render the breadcrumb-style step progress indicator.

    Args:
        current_step: 1-indexed current step number.
    """
    html = '<div class="breadcrumb-stepper">'
    for i, step in enumerate(WIZARD_STEPS):
        n = step["number"]
        if i > 0:
            html += '<span class="bc-separator">&middot;</span>'

        if n < current_step:
            html += f'''<span class="bc-step completed">
                <span class="bc-check">&#10003;</span>
                <span>{step['title']}</span>
            </span>'''
        elif n == current_step:
            html += f'''<span class="bc-step active">
                <span class="bc-num active">{n}</span>
                <span>{step['title']}</span>
            </span>'''
        else:
            html += f'''<span class="bc-step">
                <span class="bc-num pending">{n}</span>
                <span>{step['title']}</span>
            </span>'''

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_page_title(title: str, subtitle: str = ""):
    """Render a page title with optional subtitle."""
    sub = f'<div class="page-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="page-title">{title}</div>{sub}', unsafe_allow_html=True)


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
    """Return badge HTML. Variants: pass, warn, fail, info, pending, critical."""
    return f'<span class="badge badge-{variant}">{text}</span>'


def render_recommendation(title: str, body: str, variant: str = ""):
    """Render a recommendation/info panel.

    Args:
        variant: '' (default blue), 'governance' (purple), 'warning' (amber)
    """
    cls = f"recommendation-panel {variant}".strip()
    st.markdown(f"""
    <div class="{cls}">
        <div class="rec-title">{title}</div>
        <div class="rec-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)


def render_summary_card(title: str, rows: list):
    """Render a summary card with key-value rows.

    Args:
        rows: list of (label, value) or (label, value, color) tuples.
    """
    items = ""
    for row in rows:
        label, value = row[0], row[1]
        color = row[2] if len(row) > 2 else "#0f172a"
        items += f'''<div class="summary-row">
            <span class="summary-label">{label}</span>
            <span class="summary-value" style="color:{color}">{value}</span>
        </div>'''

    st.markdown(f"""
    <div class="summary-card">
        <div class="summary-title">{title}</div>
        {items}
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


def render_action_bar(current_step: int, total_steps: int = 8):
    """Render the bottom action bar with Previous / Save Draft / Next."""
    prev_html = ""
    if current_step > 1:
        prev_html = '<span class="action-btn secondary">&lsaquo; Previous</span>'

    if current_step < total_steps:
        next_html = '<span class="action-btn primary">Next &rsaquo;</span>'
    elif current_step == total_steps:
        next_html = '<span class="action-btn primary">Confirm Enablement</span>'
    else:
        next_html = ""

    st.markdown(f"""
    <div class="action-bar">
        {prev_html}
        <span class="action-btn draft">Save Draft</span>
        {next_html}
    </div>
    """, unsafe_allow_html=True)


def render_nav_buttons(current_step: int, total_steps: int = 8):
    """Render Previous / Next navigation using Streamlit buttons for interactivity."""
    render_action_bar(current_step, total_steps)


def render_mini_chart(values: list, color: str = "#93c5fd"):
    """Render a mini inline bar chart from a list of values."""
    if not values:
        return
    max_val = max(values) or 1
    bars = ""
    for v in values:
        h = max(2, int((v / max_val) * 48))
        bars += f'<div class="mini-bar" style="height:{h}px;background:{color}"></div>'
    st.markdown(f'<div class="mini-chart">{bars}</div>', unsafe_allow_html=True)


def render_sidebar_workspace():
    """Render the workspace selector in the sidebar."""
    st.markdown(f"""
    <div class="sidebar-workspace">
        <div class="sidebar-workspace-label">Workspace</div>
        <div class="sidebar-workspace-name">
            {WORKSPACE['name'].split('—')[0].strip()}
            <span style="color:#94a3b8;font-size:0.7rem">&#9662;</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
