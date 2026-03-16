"""Enterprise design tokens and global CSS for the IDP Onboarding Studio.

Visual language inspired by UBS wealth management: clean white backgrounds,
red accent color, sophisticated Swiss typography, minimal borders,
and premium financial services aesthetic.
"""

import streamlit as st

# ── Color Palette (UBS-inspired) ────────────────────────────────────

COLORS = {
    "nav_bg": "#ffffff",           # White — top nav (UBS style)
    "nav_hover": "#f5f5f5",        # Light gray for hover
    "primary": "#E60000",          # UBS Red — primary actions
    "primary_light": "#ff1a1a",    # Lighter red — accent
    "secondary": "#666666",        # Medium gray — secondary text
    "success": "#008a00",          # UBS Green — pass/healthy
    "success_light": "#e6f4e6",    # Light green background
    "warning": "#cc7a00",          # Amber — warnings
    "warning_light": "#fff5e6",    # Light amber background
    "danger": "#E60000",           # Red — errors/critical (same as primary)
    "danger_light": "#ffe6e6",     # Light red background
    "info": "#0063c3",             # UBS Blue — info badges
    "info_light": "#e6f0fa",       # Light blue background
    "surface": "#ffffff",          # White — cards
    "background": "#f5f5f5",       # Light gray — page background
    "border": "#e5e5e5",           # Light border
    "border_strong": "#cccccc",    # Stronger border
    "text_primary": "#000000",     # Black — headings
    "text_secondary": "#666666",   # Medium gray
    "text_muted": "#999999",       # Light gray
    "text_inverse": "#ffffff",     # White text on dark bg
}


def inject_global_css():
    """Inject UBS-inspired global CSS into the Streamlit app."""
    st.markdown("""
    <style>
    /* ── Global Reset & Typography ─────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background: #f5f5f5 !important;
    }

    .main .block-container {
        padding-top: 0 !important;
        padding-bottom: 2rem;
        max-width: 100% !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, div, label, input, select, textarea, button, td, th, code {
        font-family: 'Source Sans 3', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    }

    /* ── Hide ALL Streamlit chrome ────────────────────────────── */
    #MainMenu, footer { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }

    /* ── Top Navigation Bar (UBS white nav) ───────────────────── */
    .top-nav {
        background: #ffffff;
        display: flex;
        align-items: center;
        padding: 0 2rem;
        height: 52px;
        margin: 0 -1.5rem 0 -1.5rem;
        position: relative;
        z-index: 100;
        border-bottom: 1px solid #e5e5e5;
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-right: 2.5rem;
        white-space: nowrap;
    }
    .nav-brand-name {
        color: #000000;
        font-size: 0.92rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    .nav-brand-chevron {
        color: #999999;
        font-size: 0.7rem;
        margin-left: 0.25rem;
    }
    .nav-links {
        display: flex;
        align-items: stretch;
        gap: 0;
        height: 52px;
    }
    .nav-link {
        display: flex;
        align-items: center;
        padding: 0 0.85rem;
        font-size: 0.82rem;
        font-weight: 500;
        color: #666666;
        text-decoration: none;
        border-bottom: 2px solid transparent;
        transition: color 0.15s, border-color 0.15s;
        white-space: nowrap;
    }
    .nav-link:hover { color: #000000; }
    .nav-link.active {
        color: #000000;
        font-weight: 600;
        border-bottom-color: #E60000;
    }
    .nav-right {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin-left: auto;
    }
    .nav-icon {
        color: #666666;
        font-size: 1rem;
        cursor: pointer;
        padding: 0.25rem;
    }
    .nav-icon:hover { color: #000000; }
    .nav-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: #E60000;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 600;
        color: white;
        cursor: pointer;
    }

    /* ── Left Sidebar (in-page column) ──────────────────────────── */
    .sidebar-workspace {
        background: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 4px;
        padding: 0.65rem 0.75rem;
        margin-bottom: 0.75rem;
    }
    .sidebar-workspace-label {
        font-size: 0.6rem;
        font-weight: 600;
        color: #999999;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.15rem;
    }
    .sidebar-workspace-name {
        font-size: 0.82rem;
        font-weight: 600;
        color: #000000;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .sidebar-step-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #E60000;
        color: white;
        padding: 0.25rem 0.65rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    /* Style in-page sidebar buttons */
    [data-testid="stHorizontalBlock"]:first-child .stButton > button {
        text-align: left !important;
        font-size: 0.8rem !important;
        padding: 0.4rem 0.65rem !important;
        border-radius: 4px !important;
    }

    /* ── Breadcrumb Stepper ────────────────────────────────────── */
    .breadcrumb-stepper {
        display: flex;
        align-items: center;
        gap: 0;
        padding: 0.75rem 1.25rem;
        background: white;
        border: 1px solid #e5e5e5;
        border-radius: 4px;
        margin: 0.75rem 0 1rem 0;
        flex-wrap: nowrap;
        overflow-x: auto;
    }
    .bc-step {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        white-space: nowrap;
        font-size: 0.78rem;
        font-weight: 500;
        color: #999999;
    }
    .bc-step.completed { color: #008a00; }
    .bc-step.active {
        color: #E60000;
        font-weight: 600;
    }
    .bc-check {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: #008a00;
        color: white;
        font-size: 0.55rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    .bc-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        font-size: 0.65rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    .bc-num.active {
        background: #E60000;
        color: white;
    }
    .bc-num.pending {
        background: #f5f5f5;
        color: #999999;
        border: 1.5px solid #cccccc;
    }
    .bc-separator {
        margin: 0 0.5rem;
        color: #cccccc;
        font-size: 0.85rem;
        flex-shrink: 0;
    }

    /* ── Page Title ────────────────────────────────────────────── */
    .page-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .page-subtitle {
        font-size: 0.82rem;
        color: #666666;
        margin-bottom: 1rem;
        line-height: 1.4;
    }

    /* ── Cards ─────────────────────────────────────────────────── */
    .metric-card {
        background: white;
        border: 1px solid #e5e5e5;
        border-radius: 4px;
        padding: 1.15rem;
        margin-bottom: 0.65rem;
    }
    .metric-card .metric-label {
        font-size: 0.68rem;
        font-weight: 600;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }
    .metric-card .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #000000;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .metric-card .metric-sub {
        font-size: 0.75rem;
        color: #666666;
        margin-top: 0.2rem;
    }

    /* ── Section Headers ───────────────────────────────────────── */
    .section-header {
        font-size: 0.92rem;
        font-weight: 600;
        color: #000000;
        margin: 1.25rem 0 0.65rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e5e5e5;
    }

    /* ── Status Badges ─────────────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 3px;
        font-size: 0.67rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .badge-pass { background: #e6f4e6; color: #006600; }
    .badge-warn { background: #fff5e6; color: #995c00; }
    .badge-fail { background: #ffe6e6; color: #cc0000; }
    .badge-info { background: #e6f0fa; color: #004d99; }
    .badge-pending { background: #f5f5f5; color: #666666; }
    .badge-critical { background: #ffe6e6; color: #cc0000; }

    /* ── Recommendation Panel ──────────────────────────────────── */
    .recommendation-panel {
        background: #fafafa;
        border: 1px solid #e5e5e5;
        border-left: 4px solid #E60000;
        border-radius: 4px;
        padding: 0.85rem 1rem;
        margin: 0.65rem 0;
    }
    .recommendation-panel.governance {
        border-left-color: #7928ca;
    }
    .recommendation-panel.warning {
        border-left-color: #cc7a00;
        background: #fff9f0;
    }
    .recommendation-panel .rec-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #000000;
        margin-bottom: 0.35rem;
    }
    .recommendation-panel .rec-body {
        font-size: 0.78rem;
        color: #333333;
        line-height: 1.5;
    }

    /* ── Data Tables ───────────────────────────────────────────── */
    .clean-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.78rem;
        background: white;
        border: 1px solid #e5e5e5;
        border-radius: 4px;
        overflow: hidden;
    }
    .clean-table th {
        text-align: left;
        padding: 0.6rem 0.75rem;
        background: #f5f5f5;
        color: #666666;
        font-weight: 600;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 2px solid #e5e5e5;
    }
    .clean-table td {
        padding: 0.55rem 0.75rem;
        border-bottom: 1px solid #f0f0f0;
        color: #000000;
        vertical-align: middle;
    }
    .clean-table tr:last-child td { border-bottom: none; }
    .clean-table tr:hover td { background: #fafafa; }

    /* ── Mode / Selection Cards ────────────────────────────────── */
    .mode-card {
        background: white;
        border: 2px solid #e5e5e5;
        border-radius: 4px;
        padding: 1.15rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
        min-height: 140px;
    }
    .mode-card:hover {
        border-color: #cccccc;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    .mode-card.selected {
        border-color: #E60000;
        background: #fff5f5;
        box-shadow: 0 0 0 1px #E60000;
    }
    .mode-card .mode-icon {
        font-size: 1.75rem;
        margin-bottom: 0.4rem;
    }
    .mode-card .mode-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #000000;
        margin-bottom: 0.25rem;
    }
    .mode-card .mode-desc {
        font-size: 0.7rem;
        color: #666666;
        line-height: 1.4;
    }

    /* ── Checklist ─────────────────────────────────────────────── */
    .checklist-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid #f0f0f0;
        font-size: 0.8rem;
    }
    .checklist-item:last-child { border-bottom: none; }
    .checklist-icon { font-size: 0.9rem; flex-shrink: 0; }
    .checklist-text { color: #000000; }
    .checklist-category {
        margin-left: auto;
        font-size: 0.65rem;
        color: #999999;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        background: #f5f5f5;
        padding: 0.12rem 0.45rem;
        border-radius: 3px;
    }

    /* ── Progress Ring ─────────────────────────────────────────── */
    .readiness-ring {
        width: 130px;
        height: 130px;
        border-radius: 50%;
        background: conic-gradient(#E60000 0deg, #E60000 calc(var(--pct) * 3.6deg), #e5e5e5 calc(var(--pct) * 3.6deg));
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
    }
    .readiness-ring-inner {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: white;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .readiness-ring-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #E60000;
        line-height: 1;
    }
    .readiness-ring-label {
        font-size: 0.65rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.12rem;
    }

    /* ── Alert Rows ────────────────────────────────────────────── */
    .alert-row {
        display: flex;
        align-items: flex-start;
        gap: 0.65rem;
        padding: 0.7rem 0.85rem;
        border-radius: 4px;
        margin-bottom: 0.45rem;
        font-size: 0.78rem;
        border: 1px solid;
    }
    .alert-row.warning { background: #fff9f0; border-color: #ffe0b2; }
    .alert-row.info { background: #f5f9ff; border-color: #c2d9f2; }
    .alert-row.critical { background: #fff5f5; border-color: #ffcccc; }
    .alert-timestamp {
        font-size: 0.68rem;
        color: #999999;
        margin-top: 0.2rem;
    }

    /* ── Mini Charts ───────────────────────────────────────────── */
    .mini-chart {
        display: flex;
        align-items: flex-end;
        gap: 3px;
        height: 50px;
        margin-top: 0.5rem;
    }
    .mini-bar {
        flex: 1;
        border-radius: 2px 2px 0 0;
        min-width: 8px;
        transition: height 0.2s;
    }

    /* ── Bottom Action Bar ─────────────────────────────────────── */
    .action-bar {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 0.65rem;
        padding: 0.85rem 1.25rem;
        background: white;
        border: 1px solid #e5e5e5;
        border-radius: 4px;
        margin-top: 1.25rem;
    }
    .action-btn {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.5rem 1.15rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        cursor: pointer;
        border: none;
        transition: all 0.15s;
        text-decoration: none;
    }
    .action-btn.secondary {
        background: white;
        color: #333333;
        border: 1px solid #cccccc;
    }
    .action-btn.secondary:hover { background: #f5f5f5; }
    .action-btn.draft {
        background: #000000;
        color: white;
    }
    .action-btn.draft:hover { background: #333333; }
    .action-btn.primary {
        background: #E60000;
        color: white;
    }
    .action-btn.primary:hover { background: #cc0000; }

    /* ── Summary / Info Card ───────────────────────────────────── */
    .summary-card {
        background: #fafafa;
        border: 1px solid #e5e5e5;
        border-radius: 4px;
        padding: 0.85rem;
        margin: 0.65rem 0;
    }
    .summary-card .summary-title {
        font-size: 0.68rem;
        font-weight: 600;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }
    .summary-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.25rem 0;
        border-bottom: 1px solid #e5e5e5;
        font-size: 0.78rem;
    }
    .summary-row:last-child { border-bottom: none; }
    .summary-label { color: #666666; }
    .summary-value { font-weight: 600; color: #000000; }

    /* ── Progress Bars ─────────────────────────────────────────── */
    .progress-bar-track {
        background: #e5e5e5;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s;
    }

    /* ── Button overrides ──────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background-color: #E60000 !important;
        border-color: #E60000 !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #cc0000 !important;
    }
    .stButton > button[kind="secondary"] {
        border-color: #e5e5e5 !important;
        color: #333333 !important;
        font-weight: 500 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #f5f5f5 !important;
    }

    /* ── Selectbox & Input Styling ─────────────────────────────── */
    .stSelectbox > div > div, .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-color: #e5e5e5 !important;
        border-radius: 4px !important;
    }
    </style>
    """, unsafe_allow_html=True)
