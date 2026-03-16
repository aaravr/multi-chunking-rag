"""Shared layout components: top nav, breadcrumb stepper, sidebar, action bar, panels."""

import streamlit as st
from app.onboarding.mock_data import WIZARD_STEPS, WORKSPACE


# ── Navigation items for the top bar ──────────────────────────────────
NAV_ITEMS = ["Workspaces", "Documents", "Schemas", "Evaluation", "Review", "Production"]

# UBS three-keys logo as inline SVG
UBS_LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 28" height="24">'
    # Three keys icon (simplified)
    '<g transform="translate(0,2)">'
    # Key 1 (leftmost)
    '<g fill="#000000">'
    '<circle cx="6" cy="6" r="5" fill="none" stroke="#000" stroke-width="1.8"/>'
    '<rect x="5" y="10" width="2" height="12" rx="0.5"/>'
    '<rect x="7" y="16" width="4" height="1.8" rx="0.5"/>'
    '<rect x="7" y="19.5" width="3" height="1.8" rx="0.5"/>'
    '</g>'
    # Key 2 (middle, offset)
    '<g fill="#000000" transform="translate(7,0)">'
    '<circle cx="6" cy="6" r="5" fill="none" stroke="#000" stroke-width="1.8"/>'
    '<rect x="5" y="10" width="2" height="12" rx="0.5"/>'
    '<rect x="7" y="16" width="4" height="1.8" rx="0.5"/>'
    '<rect x="7" y="19.5" width="3" height="1.8" rx="0.5"/>'
    '</g>'
    # Key 3 (rightmost, offset)
    '<g fill="#000000" transform="translate(14,0)">'
    '<circle cx="6" cy="6" r="5" fill="none" stroke="#000" stroke-width="1.8"/>'
    '<rect x="5" y="10" width="2" height="12" rx="0.5"/>'
    '<rect x="7" y="16" width="4" height="1.8" rx="0.5"/>'
    '<rect x="7" y="19.5" width="3" height="1.8" rx="0.5"/>'
    '</g>'
    '</g>'
    # "UBS" text in red serif
    '<text x="40" y="20" font-family="Georgia,\'Times New Roman\',serif" '
    'font-size="18" font-weight="700" fill="#E60000" letter-spacing="1">UBS</text>'
    '</svg>'
)


def render_top_nav(active_section: str = "Workspaces"):
    """Render the enterprise top navigation bar."""
    links = ""
    for item in NAV_ITEMS:
        cls = "active" if item == active_section else ""
        links += f'<span class="nav-link {cls}">{item}</span>'

    st.markdown(f"""
    <div class="top-nav">
        <div class="nav-brand">
            {UBS_LOGO_SVG}
            <span style="color:#e5e5e5;font-size:1.2rem;font-weight:200;margin:0 0.6rem">|</span>
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
        color = row[2] if len(row) > 2 else "#000000"
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


def render_mini_chart(values: list, color: str = "#ff9999"):
    """Render a mini inline bar chart from a list of values."""
    if not values:
        return
    max_val = max(values) or 1
    bars = ""
    for v in values:
        h = max(2, int((v / max_val) * 48))
        bars += f'<div class="mini-bar" style="height:{h}px;background:{color}"></div>'
    st.markdown(f'<div class="mini-chart">{bars}</div>', unsafe_allow_html=True)


def render_stacked_bar_chart(fields: list):
    """Render a horizontal stacked bar chart for field performance.

    Args:
        fields: list of dicts with keys: name, exact, partial, error
    """
    rows = ""
    for f in fields:
        total = f["exact"] + f["partial"] + f["error"]
        if total == 0:
            total = 1
        e_pct = f["exact"] / total * 100
        p_pct = f["partial"] / total * 100
        r_pct = f["error"] / total * 100
        rows += (
            f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.45rem">'
            f'<span style="font-size:0.75rem;font-weight:500;color:#000000;width:110px;text-align:right;flex-shrink:0">{f["name"]}</span>'
            f'<div style="flex:1;display:flex;height:18px;border-radius:3px;overflow:hidden">'
            f'<div style="width:{e_pct}%;background:#16a34a"></div>'
            f'<div style="width:{p_pct}%;background:#eab308"></div>'
            f'<div style="width:{r_pct}%;background:#ef4444"></div>'
            f'</div></div>'
        )

    legend = (
        '<div style="display:flex;gap:1rem;margin-top:0.5rem;font-size:0.7rem;color:#666666">'
        '<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#16a34a;margin-right:0.3rem;vertical-align:middle"></span>Exact Match</span>'
        '<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#eab308;margin-right:0.3rem;vertical-align:middle"></span>Partial</span>'
        '<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#ef4444;margin-right:0.3rem;vertical-align:middle"></span>Errors</span>'
        '</div>'
    )

    st.html(
        f'<div style="background:white;border:1px solid #e5e5e5;border-radius:10px;padding:1rem 1.25rem">'
        f'{rows}{legend}</div>'
    )


def render_donut_chart(segments: list, center_value: str, center_label: str, size: int = 160):
    """Render a CSS donut chart.

    Args:
        segments: list of (value, color, label) tuples. Values should sum to 100.
        center_value: text in center of donut.
        center_label: sub-label below the value.
        size: pixel size of the donut.
    """
    # Build conic-gradient stops
    stops = []
    cumulative = 0
    for val, color, _label in segments:
        start = cumulative * 3.6
        end = (cumulative + val) * 3.6
        stops.append(f"{color} {start}deg {end}deg")
        cumulative += val

    gradient = ", ".join(stops)
    inner_size = int(size * 0.68)

    legend_items = ""
    for val, color, label in segments:
        legend_items += (
            f'<span style="display:flex;align-items:center;gap:0.3rem">'
            f'<span style="width:8px;height:8px;border-radius:2px;background:{color};flex-shrink:0"></span>'
            f'<span>{label}</span></span>'
        )

    st.html(
        f'<div style="text-align:center">'
        f'<div style="width:{size}px;height:{size}px;border-radius:50%;background:conic-gradient({gradient});display:flex;align-items:center;justify-content:center;margin:0 auto">'
        f'<div style="width:{inner_size}px;height:{inner_size}px;border-radius:50%;background:white;display:flex;flex-direction:column;align-items:center;justify-content:center">'
        f'<span style="font-size:1.5rem;font-weight:700;color:#000000;line-height:1">{center_value}</span>'
        f'<span style="font-size:0.62rem;color:#666666;text-transform:uppercase;letter-spacing:0.5px">{center_label}</span>'
        f'</div></div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:0.5rem 1rem;justify-content:center;margin-top:0.6rem;font-size:0.7rem;color:#333333">'
        f'{legend_items}</div></div>'
    )


def render_trend_chart(values: list, color: str = "#E60000",
                       target: float = None, height: int = 60,
                       labels: list = None):
    """Render a bar chart with optional target line and labels.

    Args:
        values: list of numeric values.
        color: bar color.
        target: optional target value to show as a dashed line.
        height: chart height in pixels.
        labels: optional list of x-axis labels.
    """
    if not values:
        return
    max_val = max(values) or 1
    bars = ""
    for i, v in enumerate(values):
        h = max(2, int((v / max_val) * (height - 4)))
        tooltip = f' title="{labels[i]}: {v}"' if labels else f' title="{v}"'
        bars += f'<div style="flex:1;height:{h}px;background:{color};border-radius:2px;min-width:4px"{tooltip}></div>'

    target_html = ""
    if target is not None:
        target_pos = int((target / max_val) * (height - 4))
        target_html = (
            f'<div style="position:absolute;bottom:{target_pos}px;left:0;right:0;border-top:2px dashed #999999;z-index:1"></div>'
            f'<div style="position:absolute;bottom:{target_pos + 2}px;right:0;font-size:0.6rem;color:#999999;background:#fafafa;padding:0 0.3rem">Target: {target}</div>'
        )

    label_html = ""
    if labels and len(labels) <= 14:
        label_items = ""
        for lbl in labels:
            label_items += f'<span style="flex:1;text-align:center;min-width:8px">{lbl}</span>'
        label_html = f'<div style="display:flex;gap:3px;font-size:0.55rem;color:#999999;margin-top:0.2rem">{label_items}</div>'

    st.html(
        f'<div style="background:white;border:1px solid #e5e5e5;border-radius:10px;padding:0.85rem 1rem">'
        f'<div style="position:relative;height:{height}px;display:flex;align-items:flex-end;gap:3px">'
        f'{target_html}{bars}</div>{label_html}</div>'
    )


def render_kpi_delta(label: str, value: str, delta: str = "",
                     delta_direction: str = "up"):
    """Render a KPI card with delta indicator.

    Args:
        delta_direction: 'up' (green arrow), 'down' (red arrow), or 'neutral'.
    """
    if delta:
        if delta_direction == "up":
            arrow = "↑"
            delta_color = "#16a34a"
        elif delta_direction == "down":
            arrow = "↓"
            delta_color = "#dc2626"
        else:
            arrow = "→"
            delta_color = "#666666"
        delta_html = f'<span style="font-size:0.75rem;font-weight:600;color:{delta_color};margin-left:0.5rem">{arrow} {delta}</span>'
    else:
        delta_html = ""

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div style="display:flex;align-items:baseline">
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_workspace():
    """Render the workspace selector in the sidebar."""
    st.markdown(f"""
    <div class="sidebar-workspace">
        <div class="sidebar-workspace-label">Workspace</div>
        <div class="sidebar-workspace-name">
            {WORKSPACE['name'].split('—')[0].strip()}
            <span style="color:#999999;font-size:0.7rem">&#9662;</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
