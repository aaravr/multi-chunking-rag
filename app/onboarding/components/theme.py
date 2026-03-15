"""Enterprise design tokens and global CSS for the IDP Onboarding Studio.

Visual language: clean white surfaces, dark header, restrained palette,
strong hierarchy. Suitable for banking / operations / compliance environments.
"""

import streamlit as st

# ── Color Palette ────────────────────────────────────────────────────

COLORS = {
    "primary": "#1B4F72",       # Deep enterprise blue
    "primary_light": "#2E86C1", # Accent blue
    "secondary": "#2C3E50",     # Dark slate
    "success": "#1E8449",       # Muted green
    "warning": "#D4AC0D",       # Amber
    "danger": "#C0392B",        # Muted red
    "info": "#2874A6",          # Steel blue
    "surface": "#FFFFFF",       # White card
    "background": "#F4F6F9",    # Light cool gray
    "border": "#D5D8DC",        # Light border
    "text_primary": "#1C2833",  # Near-black
    "text_secondary": "#5D6D7E",# Medium gray
    "text_muted": "#ABB2B9",    # Light gray
}


def inject_global_css():
    """Inject enterprise-grade global CSS into the Streamlit app."""
    st.markdown("""
    <style>
    /* ── Global Reset & Typography ─────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .main .block-container {
        padding-top: 1rem;
        max-width: 1400px;
    }

    h1, h2, h3, h4, h5, h6, p, span, div, label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    /* ── Top Header Bar ────────────────────────────────────────── */
    .top-header {
        background: linear-gradient(135deg, #1B4F72 0%, #154360 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .top-header h1 {
        color: white !important;
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .top-header .header-meta {
        color: rgba(255,255,255,0.75);
        font-size: 0.8rem;
    }

    /* ── Wizard Stepper ────────────────────────────────────────── */
    .wizard-stepper {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0;
        margin-bottom: 1.5rem;
        padding: 0.75rem 0;
        background: white;
        border-radius: 8px;
        border: 1px solid #D5D8DC;
    }
    .wizard-step {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 500;
        color: #ABB2B9;
        transition: all 0.2s;
        white-space: nowrap;
    }
    .wizard-step.active {
        background: #EBF5FB;
        color: #1B4F72;
        font-weight: 600;
    }
    .wizard-step.completed {
        color: #1E8449;
    }
    .wizard-step .step-number {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 700;
        border: 2px solid #D5D8DC;
        flex-shrink: 0;
    }
    .wizard-step.active .step-number {
        background: #1B4F72;
        color: white;
        border-color: #1B4F72;
    }
    .wizard-step.completed .step-number {
        background: #1E8449;
        color: white;
        border-color: #1E8449;
    }
    .wizard-connector {
        width: 24px;
        height: 2px;
        background: #D5D8DC;
        flex-shrink: 0;
    }
    .wizard-connector.completed {
        background: #1E8449;
    }

    /* ── Cards ─────────────────────────────────────────────────── */
    .metric-card {
        background: white;
        border: 1px solid #D5D8DC;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
    }
    .metric-card .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #5D6D7E;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }
    .metric-card .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1C2833;
        line-height: 1.2;
    }
    .metric-card .metric-sub {
        font-size: 0.78rem;
        color: #5D6D7E;
        margin-top: 0.25rem;
    }

    /* ── Section Headers ───────────────────────────────────────── */
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #1C2833;
        margin: 1.25rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #EBF5FB;
    }

    /* ── Status Badges ─────────────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .badge-pass { background: #D5F5E3; color: #1E8449; }
    .badge-warn { background: #FEF9E7; color: #B7950B; }
    .badge-fail { background: #FADBD8; color: #C0392B; }
    .badge-info { background: #D6EAF8; color: #2874A6; }
    .badge-pending { background: #F2F3F4; color: #5D6D7E; }

    /* ── Recommendation Panel ──────────────────────────────────── */
    .recommendation-panel {
        background: #EBF5FB;
        border: 1px solid #AED6F1;
        border-left: 4px solid #2E86C1;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }
    .recommendation-panel .rec-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #1B4F72;
        margin-bottom: 0.4rem;
    }
    .recommendation-panel .rec-body {
        font-size: 0.8rem;
        color: #2C3E50;
        line-height: 1.5;
    }

    /* ── Data Tables ───────────────────────────────────────────── */
    .clean-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.8rem;
    }
    .clean-table th {
        text-align: left;
        padding: 0.6rem 0.75rem;
        background: #F4F6F9;
        color: #5D6D7E;
        font-weight: 600;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 2px solid #D5D8DC;
    }
    .clean-table td {
        padding: 0.55rem 0.75rem;
        border-bottom: 1px solid #EAECEE;
        color: #1C2833;
        vertical-align: middle;
    }
    .clean-table tr:hover { background: #FAFBFC; }

    /* ── Mode Cards ────────────────────────────────────────────── */
    .mode-card {
        background: white;
        border: 2px solid #D5D8DC;
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
        min-height: 140px;
    }
    .mode-card:hover { border-color: #AED6F1; }
    .mode-card.selected {
        border-color: #1B4F72;
        background: #EBF5FB;
    }
    .mode-card .mode-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .mode-card .mode-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #1C2833;
        margin-bottom: 0.25rem;
    }
    .mode-card .mode-desc {
        font-size: 0.72rem;
        color: #5D6D7E;
        line-height: 1.4;
    }

    /* ── Checklist ─────────────────────────────────────────────── */
    .checklist-item {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid #F4F6F9;
        font-size: 0.82rem;
    }
    .checklist-icon { font-size: 1rem; flex-shrink: 0; }
    .checklist-text { color: #1C2833; }
    .checklist-category {
        margin-left: auto;
        font-size: 0.7rem;
        color: #ABB2B9;
        text-transform: uppercase;
    }

    /* ── Progress Ring ─────────────────────────────────────────── */
    .readiness-ring {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        background: conic-gradient(#1B4F72 0deg, #1B4F72 calc(var(--pct) * 3.6deg), #EAECEE calc(var(--pct) * 3.6deg));
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
    }
    .readiness-ring-inner {
        width: 110px;
        height: 110px;
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
        color: #1B4F72;
        line-height: 1;
    }
    .readiness-ring-label {
        font-size: 0.7rem;
        color: #5D6D7E;
        text-transform: uppercase;
    }

    /* ── Alert Rows ────────────────────────────────────────────── */
    .alert-row {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        font-size: 0.8rem;
    }
    .alert-row.warning { background: #FEF9E7; border-left: 3px solid #D4AC0D; }
    .alert-row.info { background: #EBF5FB; border-left: 3px solid #2E86C1; }
    .alert-row.critical { background: #FADBD8; border-left: 3px solid #C0392B; }
    .alert-timestamp {
        font-size: 0.7rem;
        color: #ABB2B9;
        margin-top: 0.25rem;
    }

    /* ── Chart placeholder ─────────────────────────────────────── */
    .mini-chart {
        display: flex;
        align-items: flex-end;
        gap: 3px;
        height: 50px;
        margin-top: 0.5rem;
    }
    .mini-bar {
        flex: 1;
        background: #AED6F1;
        border-radius: 2px 2px 0 0;
        min-width: 8px;
    }

    /* ── Sidebar ───────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #1C2833 !important;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: rgba(255,255,255,0.85) !important;
    }

    /* ── Button overrides ──────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background-color: #1B4F72;
        border-color: #1B4F72;
    }

    /* ── Hide Streamlit artifacts for screenshots ──────────────── */
    .screenshot-mode #MainMenu, .screenshot-mode footer,
    .screenshot-mode header[data-testid="stHeader"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
