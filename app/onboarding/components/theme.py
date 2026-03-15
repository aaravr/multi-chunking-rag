"""Enterprise design tokens and global CSS for the IDP Onboarding Studio.

Visual language inspired by enterprise SaaS platforms: dark top nav bar,
left sidebar with workspace selector, breadcrumb stepper, bottom action bar,
clean data tables with status badges, and professional card layouts.
"""

import streamlit as st

# ── Color Palette ────────────────────────────────────────────────────

COLORS = {
    "nav_bg": "#1e293b",           # Dark navy — top nav & sidebar
    "nav_hover": "#334155",        # Slightly lighter navy for hover
    "primary": "#1e40af",          # Strong blue — primary actions
    "primary_light": "#3b82f6",    # Bright blue — active/accent
    "secondary": "#475569",        # Slate — secondary text/elements
    "success": "#16a34a",          # Green — pass/healthy
    "success_light": "#dcfce7",    # Light green background
    "warning": "#d97706",          # Amber — warnings
    "warning_light": "#fef3c7",    # Light amber background
    "danger": "#dc2626",           # Red — errors/critical
    "danger_light": "#fee2e2",     # Light red background
    "info": "#2563eb",             # Blue — info badges
    "info_light": "#dbeafe",       # Light blue background
    "surface": "#ffffff",          # White — cards
    "background": "#f1f5f9",       # Cool gray — page background
    "border": "#e2e8f0",           # Light border
    "border_strong": "#cbd5e1",    # Stronger border
    "text_primary": "#0f172a",     # Near-black
    "text_secondary": "#64748b",   # Medium gray
    "text_muted": "#94a3b8",       # Light gray
    "text_inverse": "#ffffff",     # White text on dark bg
}


def inject_global_css():
    """Inject enterprise-grade global CSS into the Streamlit app."""
    st.markdown("""
    <style>
    /* ── Global Reset & Typography ─────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background: #f1f5f9 !important;
    }

    .main .block-container {
        padding-top: 0 !important;
        padding-bottom: 2rem;
        max-width: 1440px;
    }

    h1, h2, h3, h4, h5, h6, p, span, div, label, input, select, textarea, button, td, th, code {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    /* ── Top Navigation Bar ────────────────────────────────────── */
    .top-nav {
        background: #1e293b;
        display: flex;
        align-items: center;
        padding: 0 1.5rem;
        height: 48px;
        margin: -1rem calc(-1rem - 5%) 0 calc(-1rem - 5%);
        padding-left: calc(1.5rem + 5%);
        padding-right: calc(1.5rem + 5%);
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-right: 2rem;
        white-space: nowrap;
    }
    .nav-brand-name {
        color: #ffffff;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    .nav-brand-chevron {
        color: #94a3b8;
        font-size: 0.7rem;
        margin-left: 0.25rem;
    }
    .nav-links {
        display: flex;
        align-items: stretch;
        gap: 0;
        height: 48px;
    }
    .nav-link {
        display: flex;
        align-items: center;
        padding: 0 1rem;
        font-size: 0.8rem;
        font-weight: 500;
        color: #94a3b8;
        text-decoration: none;
        border-bottom: 2px solid transparent;
        transition: color 0.15s, border-color 0.15s;
        white-space: nowrap;
    }
    .nav-link:hover {
        color: #e2e8f0;
    }
    .nav-link.active {
        color: #ffffff;
        border-bottom-color: #3b82f6;
    }
    .nav-right {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-left: auto;
    }
    .nav-icon {
        color: #94a3b8;
        font-size: 1rem;
        cursor: pointer;
        padding: 0.25rem;
    }
    .nav-icon:hover { color: #e2e8f0; }
    .nav-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: #3b82f6;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 600;
        color: white;
        cursor: pointer;
    }

    /* ── Left Sidebar Panel ─────────────────────────────────────── */
    .sidebar-workspace {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 1rem;
    }
    .sidebar-workspace-label {
        font-size: 0.65rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }
    .sidebar-workspace-name {
        font-size: 0.85rem;
        font-weight: 600;
        color: #1e293b;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .sidebar-step-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #1e40af;
        color: white;
        padding: 0.3rem 0.75rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    .sidebar-nav-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-size: 0.82rem;
        color: #475569;
        cursor: pointer;
        transition: background 0.15s;
        margin-bottom: 0.15rem;
    }
    .sidebar-nav-item:hover {
        background: #f1f5f9;
    }
    .sidebar-nav-item.active {
        background: #eff6ff;
        color: #1e40af;
        font-weight: 600;
    }
    .sidebar-nav-item .nav-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ── Breadcrumb Stepper ────────────────────────────────────── */
    .breadcrumb-stepper {
        display: flex;
        align-items: center;
        gap: 0;
        padding: 0.85rem 1.25rem;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        margin: 1rem 0 1.25rem 0;
        flex-wrap: nowrap;
        overflow-x: auto;
    }
    .bc-step {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        white-space: nowrap;
        font-size: 0.8rem;
        font-weight: 500;
        color: #94a3b8;
    }
    .bc-step.completed {
        color: #16a34a;
    }
    .bc-step.active {
        color: #1e40af;
        font-weight: 600;
    }
    .bc-check {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #16a34a;
        color: white;
        font-size: 0.6rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    .bc-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        font-size: 0.7rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    .bc-num.active {
        background: #1e40af;
        color: white;
    }
    .bc-num.pending {
        background: #f1f5f9;
        color: #94a3b8;
        border: 1.5px solid #cbd5e1;
    }
    .bc-separator {
        margin: 0 0.6rem;
        color: #cbd5e1;
        font-size: 0.7rem;
        flex-shrink: 0;
    }

    /* ── Page Title ────────────────────────────────────────────── */
    .page-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .page-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 1.25rem;
        line-height: 1.4;
    }

    /* ── Cards ─────────────────────────────────────────────────── */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
    }
    .metric-card .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    .metric-card .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .metric-card .metric-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* ── Section Headers ───────────────────────────────────────── */
    .section-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: #0f172a;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .section-header:first-child {
        margin-top: 0.5rem;
    }

    /* ── Status Badges ─────────────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 12px;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .badge-pass { background: #dcfce7; color: #15803d; }
    .badge-warn { background: #fef3c7; color: #b45309; }
    .badge-fail { background: #fee2e2; color: #dc2626; }
    .badge-info { background: #dbeafe; color: #1d4ed8; }
    .badge-pending { background: #f1f5f9; color: #64748b; }
    .badge-critical { background: #fce7f3; color: #be185d; }

    /* ── Recommendation Panel ──────────────────────────────────── */
    .recommendation-panel {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }
    .recommendation-panel.governance {
        border-left-color: #8b5cf6;
    }
    .recommendation-panel.warning {
        border-left-color: #d97706;
        background: #fffbeb;
    }
    .recommendation-panel .rec-icon {
        font-size: 0.85rem;
        margin-right: 0.4rem;
    }
    .recommendation-panel .rec-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.4rem;
    }
    .recommendation-panel .rec-body {
        font-size: 0.8rem;
        color: #475569;
        line-height: 1.55;
    }

    /* ── Data Tables ───────────────────────────────────────────── */
    .clean-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.8rem;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
    }
    .clean-table th {
        text-align: left;
        padding: 0.65rem 0.85rem;
        background: #f8fafc;
        color: #64748b;
        font-weight: 600;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 2px solid #e2e8f0;
    }
    .clean-table td {
        padding: 0.6rem 0.85rem;
        border-bottom: 1px solid #f1f5f9;
        color: #1e293b;
        vertical-align: middle;
    }
    .clean-table tr:last-child td { border-bottom: none; }
    .clean-table tr:hover td { background: #f8fafc; }

    /* ── Mode / Selection Cards ────────────────────────────────── */
    .mode-card {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
        min-height: 140px;
    }
    .mode-card:hover {
        border-color: #93c5fd;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.08);
    }
    .mode-card.selected {
        border-color: #1e40af;
        background: #eff6ff;
        box-shadow: 0 0 0 1px #1e40af;
    }
    .mode-card .mode-icon {
        font-size: 1.75rem;
        margin-bottom: 0.5rem;
    }
    .mode-card .mode-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.3rem;
    }
    .mode-card .mode-desc {
        font-size: 0.72rem;
        color: #64748b;
        line-height: 1.45;
    }

    /* ── Checklist ─────────────────────────────────────────────── */
    .checklist-item {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.55rem 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 0.82rem;
    }
    .checklist-item:last-child { border-bottom: none; }
    .checklist-icon { font-size: 1rem; flex-shrink: 0; }
    .checklist-text { color: #1e293b; }
    .checklist-category {
        margin-left: auto;
        font-size: 0.68rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        background: #f1f5f9;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
    }

    /* ── Progress Ring ─────────────────────────────────────────── */
    .readiness-ring {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        background: conic-gradient(#1e40af 0deg, #1e40af calc(var(--pct) * 3.6deg), #e2e8f0 calc(var(--pct) * 3.6deg));
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
    }
    .readiness-ring-inner {
        width: 108px;
        height: 108px;
        border-radius: 50%;
        background: white;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .readiness-ring-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e40af;
        line-height: 1;
    }
    .readiness-ring-label {
        font-size: 0.68rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.15rem;
    }

    /* ── Alert Rows ────────────────────────────────────────────── */
    .alert-row {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-size: 0.8rem;
        border: 1px solid;
    }
    .alert-row.warning { background: #fffbeb; border-color: #fde68a; }
    .alert-row.info { background: #eff6ff; border-color: #bfdbfe; }
    .alert-row.critical { background: #fef2f2; border-color: #fecaca; }
    .alert-timestamp {
        font-size: 0.7rem;
        color: #94a3b8;
        margin-top: 0.25rem;
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
        gap: 0.75rem;
        padding: 1rem 1.25rem;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        margin-top: 1.5rem;
    }
    .action-btn {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.55rem 1.25rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        cursor: pointer;
        border: none;
        transition: all 0.15s;
        text-decoration: none;
    }
    .action-btn.secondary {
        background: white;
        color: #475569;
        border: 1px solid #cbd5e1;
    }
    .action-btn.secondary:hover {
        background: #f8fafc;
        border-color: #94a3b8;
    }
    .action-btn.draft {
        background: #1e40af;
        color: white;
    }
    .action-btn.draft:hover {
        background: #1e3a8a;
    }
    .action-btn.primary {
        background: #0f172a;
        color: white;
    }
    .action-btn.primary:hover {
        background: #1e293b;
    }

    /* ── Summary / Info Card ───────────────────────────────────── */
    .summary-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.75rem 0;
    }
    .summary-card .summary-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .summary-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.3rem 0;
        border-bottom: 1px solid #e2e8f0;
        font-size: 0.8rem;
    }
    .summary-row:last-child { border-bottom: none; }
    .summary-label { color: #64748b; }
    .summary-value { font-weight: 600; color: #0f172a; }

    /* ── Sidebar Styling (Streamlit) ──────────────────────────── */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] .stButton > button {
        text-align: left !important;
        color: #475569 !important;
        background: transparent !important;
        border: none !important;
        font-size: 0.82rem !important;
        padding: 0.45rem 0.75rem !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #f1f5f9 !important;
        color: #1e293b !important;
    }

    /* ── Button overrides ──────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background-color: #1e40af !important;
        border-color: #1e40af !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1e3a8a !important;
        border-color: #1e3a8a !important;
    }
    .stButton > button[kind="secondary"] {
        border-color: #cbd5e1 !important;
        color: #475569 !important;
    }

    /* ── Progress Bars ─────────────────────────────────────────── */
    .progress-bar-track {
        background: #e2e8f0;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s;
    }

    /* ── Hide Streamlit branding ──────────────────────────────── */
    #MainMenu, footer { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }

    /* ── Selectbox & Input Styling ─────────────────────────────── */
    .stSelectbox > div > div, .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-color: #e2e8f0 !important;
        border-radius: 6px !important;
    }
    .stSelectbox > div > div:focus-within, .stTextInput > div > div:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }
    </style>
    """, unsafe_allow_html=True)
